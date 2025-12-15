import cv2
from deepface import DeepFace
import threading
import time
import os
import glob
import numpy as np
import json


class FaceAnalyzer:
    def __init__(self):
        self.cap = None
        self.analysis_result = {}
        self.analyzing = False
        self.last_analysis_time = 0
        self.analysis_interval = 2.0  # Analizar cada 2 segundos

        self.faces_dir = 'rostros_registrados'
        os.makedirs(self.faces_dir, exist_ok=True)

        # Cargar embeddings existentes
        self.person_embeddings = []
        for npy in glob.glob(os.path.join(self.faces_dir, 'face_*.npy')):
            try:
                emb = np.load(npy)
                self.person_embeddings.append(emb)
            except Exception:
                pass

        # Parámetros
        self.min_sharpness = 40.0
        self.min_brightness = 40
        self.max_brightness = 220
        self.save_consecutive_required = 6
        self.save_time_window = 8.0  # segundos
        # Si no llegan suficientes frames consecutivos, permitimos guardar por tiempo
        self.min_save_count_time = 3

        # Embedding thresholds
        self.use_cosine = True
        self.embedding_cosine_thresh = 0.28
        self.candidate_cosine_thresh = 0.22
        self.embedding_threshold = 3.8
        self.candidate_similarity_thresh = 2.5
        self.candidate_iou_thresh = 0.6

        # Verbosity control
        self.verbose = False

        # Tracker simple
        self.tracks = {}  # id -> {bbox, last_seen, count, embedding, saved}
        self.next_track_id = 1
        self.track_max_age = 4.0
        # Pre-cargar modelo de embeddings (Facenet) para evitar cargarlo cada vez
        try:
            self.facenet_model = DeepFace.build_model('Facenet')
            if self.verbose:
                print('Modelo Facenet cargado en memoria')
        except Exception as e:
            self.facenet_model = None
            if self.verbose:
                print(f'No se pudo pre-cargar Facenet: {e}')
    def _iou(self, boxA, boxB):
        if boxA is None or boxB is None:
            return 0.0
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        denom = float(boxAArea + boxBArea - interArea)
        if denom == 0:
            return 0.0
        return interArea / denom

    def _cosine_distance(self, a, b):
        a = np.array(a)
        b = np.array(b)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 1.0
        return 1.0 - (np.dot(a, b) / (na * nb))

    def analyze_face(self, frame):
        """Analiza un frame y, si corresponde, guarda un rostro único en disco."""
        try:
            results = DeepFace.analyze(
                frame,
                actions=['age', 'gender', 'emotion'],
                enforce_detection=False,
                detector_backend='opencv',
                silent=True,
            )
            # Pre-cargar modelo de embeddings (Facenet) para evitar cargarlo cada vez
            self.facenet_model = DeepFace.build_model('Facenet')
            print('Modelo Facenet cargado en memoria')
            if isinstance(results, list):
                self.analysis_result = results[0] if results else {}
            else:
                self.analysis_result = results

            faces = results if isinstance(results, list) else [results]
            if not faces:
                return

            # Seleccionar rostro más grande
            face = max(
                faces,
                key=lambda f: f.get('region', {}).get('w', 0) * f.get('region', {}).get('h', 0),
            )
            region = face.get('region', {})
            x, y, w, h = (
                region.get('x', 0),
                region.get('y', 0),
                region.get('w', 0),
                region.get('h', 0),
            )
            if w <= 0 or h <= 0:
                return

            face_img = frame[y : y + h, x : x + w]
            if face_img.size == 0:
                return

            face_img_resized = cv2.resize(face_img, (160, 160))

            # Calidad
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            brightness = int(gray_face.mean())
            if sharpness < self.min_sharpness or not (
                self.min_brightness <= brightness <= self.max_brightness
            ):
                print(f"Rostro descartado por calidad (sharp={sharpness:.1f}, bright={brightness})")
                return

            # Obtener embedding
            try:
                rep = DeepFace.represent(
                    face_img_resized, model_name='Facenet', enforce_detection=False, detector_backend='opencv'
                )
                embedding = rep[0]['embedding'] if isinstance(rep, list) and rep else rep['embedding']
            except Exception as e:
                print(f"Error obteniendo embedding: {e}")
                return

            # Comparar con person_embeddings
            is_duplicate = False
            for emb_reg in self.person_embeddings:
                try:
                    if self.use_cosine:
                        dist = self._cosine_distance(embedding, emb_reg)
                        if dist < self.embedding_cosine_thresh:
                            is_duplicate = True
                            break
                    else:
                        dist = np.linalg.norm(np.array(embedding) - emb_reg)
                        if dist < self.embedding_threshold:
                            is_duplicate = True
                            break
                except Exception:
                    continue

            if is_duplicate:
                if self.verbose:
                    print("Rostro ya registrado (duplicado)")
                return

            # Tracker: asignar a track existente por IoU o crear nuevo
            current_time = time.time()
            current_bbox = (x, y, w, h)
            # limpiar tracks viejos
            for tid in list(self.tracks.keys()):
                if current_time - self.tracks[tid]['last_seen'] > self.track_max_age:
                    del self.tracks[tid]

            best_id = None
            best_iou = 0.0
            for tid, tr in self.tracks.items():
                try:
                    i = self._iou(tr['bbox'], current_bbox)
                except Exception:
                    i = 0.0
                if i > best_iou:
                    best_iou = i
                    best_id = tid

            if best_id is not None and best_iou >= self.candidate_iou_thresh:
                tr = self.tracks[best_id]
                tr['last_seen'] = current_time
                tr['bbox'] = current_bbox
                # comparar embeddings
                if self.use_cosine:
                    cd_dist = self._cosine_distance(embedding, tr['embedding'])
                    similar = cd_dist < self.candidate_cosine_thresh
                else:
                    cd_dist = np.linalg.norm(np.array(embedding) - tr['embedding'])
                    similar = cd_dist < self.candidate_similarity_thresh
                if similar:
                    tr['count'] += 1
                    tr['embedding'] = (
                        np.array(tr['embedding']) * (tr['count'] - 1) + np.array(embedding)
                    ) / tr['count']
                    print(
                        f"Track {best_id} incrementado ({tr['count']}), iou={best_iou:.2f}, cd_dist={cd_dist:.2f}"
                    )
                else:
                    tr['embedding'] = embedding
                    tr['count'] = 1
                    print(f"Track {best_id} reiniciado (embedding diferente), cd_dist={cd_dist:.2f}")
                track_ref = tr
            else:
                tid = self.next_track_id
                self.next_track_id += 1
                self.tracks[tid] = {
                    'bbox': current_bbox,
                    'last_seen': current_time,
                    'first_seen': current_time,
                    'count': 1,
                    'embedding': embedding,
                    'saved': False,
                }
                track_ref = self.tracks[tid]
                print(f"Nuevo track {tid} iniciado")

            # Guardar si el track alcanzó el mínimo requerido
            # Condición de guardado:
            # 1) alcanzó el número requerido de apariciones consecutivas
            # OR
            # 2) ha pasado la ventana de tiempo `save_time_window` desde la primera aparición
            #    y ha alcanzado al menos `min_save_count_time` apariciones
            should_save = False
            if track_ref.get('count', 0) >= self.save_consecutive_required:
                should_save = True
            else:
                age = current_time - track_ref.get('first_seen', current_time)
                if age >= self.save_time_window and track_ref.get('count', 0) >= self.min_save_count_time:
                    should_save = True

            if should_save and not track_ref.get('saved', False):
                timestamp = int(time.time() * 1000)
                filename = f"face_{timestamp}.jpg"
                filepath = os.path.join(self.faces_dir, filename)
                cv2.imwrite(filepath, face_img)
                np.save(filepath.replace('.jpg', '.npy'), np.array(track_ref['embedding']))
                datos = {
                    'edad': face.get('age', None),
                    'genero': face.get('dominant_gender', None),
                    'emocion': face.get('dominant_emotion', None),
                }
                with open(filepath.replace('.jpg', '.json'), 'w', encoding='utf-8') as f:
                    json.dump(datos, f, ensure_ascii=False, indent=2)
                print(f"Rostro único guardado para track tras {track_ref['count']} apariciones: {filename} y datos")
                track_ref['saved'] = True
                try:
                    self.person_embeddings.append(np.array(track_ref['embedding']))
                except Exception:
                    pass

        except Exception as e:
            print(f"Error en analyze_face: {e}")
        finally:
            # permitir nuevos análisis
            self.analyzing = False

    def draw_results(self, frame, face_locations):
        """Dibuja los resultados en el frame y retorna el frame modificado"""
        for (x, y, w, h) in face_locations:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if self.analysis_result:
            edad = self.analysis_result.get('age', 'N/A')
            genero = self.analysis_result.get('dominant_gender', 'N/A')
            emocion = self.analysis_result.get('dominant_emotion', 'N/A')
            genero_es = 'Hombre' if genero == 'Man' else 'Mujer' if genero == 'Woman' else genero
            text = f"Edad: {edad}  Género: {genero_es}  Emoción: {emocion}"
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame

    def run(self, camera_index=0, rtsp_url=None, rtsp_user=None, rtsp_pass=None):
        # Seleccionar fuente: si se proporcionó rtsp_url, construir URL con credenciales opcionales
        source = camera_index
        if rtsp_url:
            rtsp_full = rtsp_url
            try:
                if rtsp_user:
                    # insertar credenciales si no están ya en la URL
                    if '://' in rtsp_url:
                        scheme, rest = rtsp_url.split('://', 1)
                        if '@' not in rest:
                            rtsp_full = f"{scheme}://{rtsp_user}:{rtsp_pass}@{rest}"
                source = rtsp_full
                print(f"Conectando a cámara RTSP: {rtsp_url}")
            except Exception:
                source = rtsp_url
                print("Conectando a cámara RTSP (sin insertar credenciales automáticamente)")

        # Crear VideoCapture con la fuente seleccionada
        self.cap = cv2.VideoCapture(source)
        # Reducir buffer para streams RTSP
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        except Exception:
            pass
        # Ajustes básicos
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            print("Error: No se pudo abrir la cámara")
            return

        # Cargar detector de rostros de OpenCV
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        print("Aplicación iniciada correctamente!")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error al leer frame")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )

            current_time = time.time()
            if len(faces) > 0 and not self.analyzing:
                if current_time - self.last_analysis_time > self.analysis_interval:
                    self.analyzing = True
                    self.last_analysis_time = current_time
                    threading.Thread(target=self.analyze_face, args=(frame.copy(),), daemon=True).start()

            frame = self.draw_results(frame, faces)

            fps_text = f'Rostros detectados: {len(faces)}'
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Reconocimiento Facial - Presiona Q para salir', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print("Aplicación cerrada")


if __name__ == "__main__":
    print("=" * 50)
    print("APLICACIÓN DE RECONOCIMIENTO FACIAL")
    print("Detecta: Rostros, Edad, Género y Emociones")
    print("=" * 50)
    print()

    analyzer = FaceAnalyzer()
    # Si quieres usar la cámara IP que indicaste, se llaman estos parámetros.
    # Cambia o borra las credenciales si prefieres ingresarlas manualmente.
    analyzer.run(
        camera_index=0,
        rtsp_url="rtsp://192.168.1.213:554/dev.hik-connect.com/channels/101/",
        rtsp_user="admin",
        rtsp_pass="Nodo2023",
    )
# """
# Aplicación Web de Detección Facial con Edad y Género
# Optimizada para CPU - Intel Core i7-10700

# Instalación requerida:
# pip install deepface
# pip install opencv-python
# pip install tf-keras
# pip install flask
# pip install pillow

# USO: python main.py
# Luego abre tu navegador en: http://localhost:5000
# """

# from flask import Flask, render_template, Response, jsonify
# import cv2
# from deepface import DeepFace
# import threading
# import json
# import base64
# import numpy as np
# from datetime import datetime
# import os

# app = Flask(__name__)

# class FaceDetector:
#     def __init__(self):
#         self.frame_skip = 2
#         self.frame_count = 0
#         self.last_analysis = []
#         self.current_frame = None
#         self.running = False
#         self.cap = None
#         self.lock = threading.Lock()
#         self.stats = {
#             'total_faces': 0,
#             'detections': []
#         }
        
#     def analyze_frame(self, frame):
#         """Analiza un frame y retorna información facial"""
#         try:
#             results = DeepFace.analyze(
#                 frame,
#                 actions=['age', 'gender', 'emotion'],
#                 enforce_detection=False,
#                 detector_backend='opencv',
#                 silent=True
#             )
            
#             if isinstance(results, list):
#                 return results
#             else:
#                 return [results]
                
#         except Exception as e:
#             return []
    
#     def draw_info(self, frame, analysis_results):
#         """Dibuja la información en el frame"""
#         faces_detected = len(analysis_results)
        
#         for idx, result in enumerate(analysis_results):
#             region = result.get('region', {})
#             x = region.get('x', 0)
#             y = region.get('y', 0)
#             w = region.get('w', 100)
#             h = region.get('h', 100)
            
#             # Rectángulo verde alrededor del rostro
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
#             # Información
#             age = result.get('age', 'N/A')
#             gender = result.get('dominant_gender', 'N/A')
#             emotion = result.get('dominant_emotion', 'N/A')
#             gender_conf = result.get('gender', {})
            
#             confidence = gender_conf.get(gender, 0) if isinstance(gender_conf, dict) else 0
            
#             # Textos
#             info = [
#                 f"Rostro #{idx + 1}",
#                 f"Edad: {int(age)} anos",
#                 f"Genero: {gender} ({confidence:.1f}%)",
#                 f"Emocion: {emotion}"
#             ]
            
#             # Dibujar cada línea
#             text_y = y - 10
#             for i, text in enumerate(info):
#                 text_y_pos = text_y - (len(info) - i) * 22
                
#                 (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
#                 # Fondo negro
#                 cv2.rectangle(
#                     frame,
#                     (x, text_y_pos - th - 3),
#                     (x + tw + 4, text_y_pos + 3),
#                     (0, 0, 0),
#                     -1
#                 )
                
#                 # Texto blanco
#                 cv2.putText(
#                     frame,
#                     text,
#                     (x + 2, text_y_pos),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5,
#                     (255, 255, 255),
#                     1
#                 )
        
#         # Info general
#         cv2.putText(
#             frame,
#             f"Rostros detectados: {faces_detected}",
#             (10, 30),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7,
#             (0, 255, 0),
#             2
#         )
        
#         return frame
    
#     def start_camera(self, camera_index=0):
#         """Inicia la cámara"""
#         if self.cap is not None:
#             self.cap.release()
            
#         self.cap = cv2.VideoCapture(camera_index)
        
#         if not self.cap.isOpened():
#             raise Exception("No se pudo abrir la cámara")
        
#         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#         self.cap.set(cv2.CAP_PROP_FPS, 30)
        
#         self.running = True
#         return True
    
#     def stop_camera(self):
#         """Detiene la cámara"""
#         self.running = False
#         if self.cap is not None:
#             self.cap.release()
#             self.cap = None
    
#     def get_frame(self):
#         """Obtiene y procesa un frame"""
#         if not self.running or self.cap is None:
#             return None
        
#         ret, frame = self.cap.read()
#         if not ret:
#             return None
        
#         with self.lock:
#             # Procesar solo cada N frames
#             if self.frame_count % self.frame_skip == 0:
#                 self.last_analysis = self.analyze_frame(frame)
                
#                 # Actualizar estadísticas
#                 if self.last_analysis:
#                     self.stats['total_faces'] = len(self.last_analysis)
#                     for result in self.last_analysis:
#                         detection = {
#                             'timestamp': datetime.now().isoformat(),
#                             'age': int(result.get('age', 0)),
#                             'gender': result.get('dominant_gender', 'N/A'),
#                             'emotion': result.get('dominant_emotion', 'N/A')
#                         }
#                         self.stats['detections'].append(detection)
#                         # Mantener solo últimos 100 registros
#                         if len(self.stats['detections']) > 100:
#                             self.stats['detections'].pop(0)
            
#             # Dibujar información
#             if self.last_analysis:
#                 frame = self.draw_info(frame, self.last_analysis)
            
#             self.frame_count += 1
#             self.current_frame = frame.copy()
        
#         return frame

# # Instancia global
# detector = FaceDetector()

# def generate_frames():
#     """Generador de frames para streaming"""
#     while True:
#         frame = detector.get_frame()
#         if frame is None:
#             break
        
#         ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
#         if not ret:
#             continue
            
#         frame_bytes = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# @app.route('/')
# def index():
#     """Página principal"""
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     """Stream de video"""
#     return Response(
#         generate_frames(),
#         mimetype='multipart/x-mixed-replace; boundary=frame'
#     )

# @app.route('/start_camera', methods=['POST'])
# def start_camera():
#     """Inicia la cámara"""
#     try:
#         detector.start_camera(0)
#         return jsonify({'success': True, 'message': 'Cámara iniciada'})
#     except Exception as e:
#         return jsonify({'success': False, 'message': str(e)})

# @app.route('/stop_camera', methods=['POST'])
# def stop_camera():
#     """Detiene la cámara"""
#     detector.stop_camera()
#     return jsonify({'success': True, 'message': 'Cámara detenida'})

# @app.route('/capture', methods=['POST'])
# def capture():
#     """Captura el frame actual"""
#     if detector.current_frame is not None:
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"captura_{timestamp}.jpg"
        
#         # Crear carpeta si no existe
#         os.makedirs('capturas', exist_ok=True)
#         filepath = os.path.join('capturas', filename)
        
#         cv2.imwrite(filepath, detector.current_frame)
#         return jsonify({'success': True, 'filename': filename})
#     return jsonify({'success': False, 'message': 'No hay frame disponible'})

# @app.route('/stats')
# def get_stats():
#     """Obtiene estadísticas"""
#     return jsonify(detector.stats)

# if __name__ == "__main__":
#     print("="*60)
#     print("DETECTOR FACIAL - EDAD Y GÉNERO (Versión Web)")
#     print("="*60)
#     print("\n🚀 Iniciando servidor web...")
#     print("\n✅ Servidor iniciado correctamente")
#     print("\n📱 Abre tu navegador en:")
#     print("   http://localhost:5000")
#     print("   http://127.0.0.1:5000")
#     print("\n⏹️  Presiona Ctrl+C para detener el servidor\n")
    
#     app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)

# # from flask import Flask, render_template, Response, jsonify
# # import cv2
# # from deepface import DeepFace
# # import threading
# # import json
# # import base64
# # import numpy as np
# # from datetime import datetime
# # import os

# # app = Flask(__name__)

# # class FaceDetector:
# #     def __init__(self):
# #         self.frame_skip = 2
# #         self.frame_count = 0
# #         self.last_analysis = []
# #         self.current_frame = None
# #         self.running = False
# #         self.cap = None
# #         self.lock = threading.Lock()
# #         self.stats = {
# #             'total_faces': 0,
# #             'detections': []
# #         }
        
# #     def analyze_frame(self, frame):
# #         """Analiza un frame y retorna información facial"""
# #         try:
# #             results = DeepFace.analyze(
# #                 frame,
# #                 actions=['age', 'gender', 'emotion'],
# #                 enforce_detection=False,
# #                 detector_backend='opencv',
# #                 silent=True
# #             )
            
# #             if isinstance(results, list):
# #                 return results
# #             else:
# #                 return [results]
                
# #         except Exception as e:
# #             return []
    
# #     def draw_info(self, frame, analysis_results):
# #         """Dibuja la información en el frame"""
# #         faces_detected = len(analysis_results)
        
# #         for idx, result in enumerate(analysis_results):
# #             region = result.get('region', {})
# #             x = region.get('x', 0)
# #             y = region.get('y', 0)
# #             w = region.get('w', 100)
# #             h = region.get('h', 100)
            
# #             # Rectángulo verde alrededor del rostro
# #             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
# #             # Información
# #             age = result.get('age', 'N/A')
# #             gender = result.get('dominant_gender', 'N/A')
# #             emotion = result.get('dominant_emotion', 'N/A')
# #             gender_conf = result.get('gender', {})
            
# #             confidence = gender_conf.get(gender, 0) if isinstance(gender_conf, dict) else 0
            
# #             # Textos
# #             info = [
# #                 f"Rostro #{idx + 1}",
# #                 f"Edad: {int(age)} anos",
# #                 f"Genero: {gender} ({confidence:.1f}%)",
# #                 f"Emocion: {emotion}"
# #             ]
            
# #             # Dibujar cada línea
# #             text_y = y - 10
# #             for i, text in enumerate(info):
# #                 text_y_pos = text_y - (len(info) - i) * 22
                
# #                 (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
# #                 # Fondo negro
# #                 cv2.rectangle(
# #                     frame,
# #                     (x, text_y_pos - th - 3),
# #                     (x + tw + 4, text_y_pos + 3),
# #                     (0, 0, 0),
# #                     -1
# #                 )
                
# #                 # Texto blanco
# #                 cv2.putText(
# #                     frame,
# #                     text,
# #                     (x + 2, text_y_pos),
# #                     cv2.FONT_HERSHEY_SIMPLEX,
# #                     0.5,
# #                     (255, 255, 255),
# #                     1
# #                 )
        
# #         # Info general
# #         cv2.putText(
# #             frame,
# #             f"Rostros detectados: {faces_detected}",
# #             (10, 30),
# #             cv2.FONT_HERSHEY_SIMPLEX,
# #             0.7,
# #             (0, 255, 0),
# #             2
# #         )
        
# #         return frame
    
# #     def start_camera(self, camera_index=0):
# #         """Inicia la cámara"""
# #         if self.cap is not None:
# #             self.cap.release()
            
# #         self.cap = cv2.VideoCapture(camera_index)
        
# #         if not self.cap.isOpened():
# #             raise Exception("No se pudo abrir la cámara")
        
# #         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# #         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# #         self.cap.set(cv2.CAP_PROP_FPS, 30)
        
# #         self.running = True
# #         return True
    
# #     def stop_camera(self):
# #         """Detiene la cámara"""
# #         self.running = False
# #         if self.cap is not None:
# #             self.cap.release()
# #             self.cap = None
    
# #     def get_frame(self):
# #         """Obtiene y procesa un frame"""
# #         if not self.running or self.cap is None:
# #             return None
        
# #         ret, frame = self.cap.read()
# #         if not ret:
# #             return None
        
# #         with self.lock:
# #             # Procesar solo cada N frames
# #             if self.frame_count % self.frame_skip == 0:
# #                 self.last_analysis = self.analyze_frame(frame)
                
# #                 # Actualizar estadísticas
# #                 if self.last_analysis:
# #                     self.stats['total_faces'] = len(self.last_analysis)
# #                     for result in self.last_analysis:
# #                         detection = {
# #                             'timestamp': datetime.now().isoformat(),
# #                             'age': int(result.get('age', 0)),
# #                             'gender': result.get('dominant_gender', 'N/A'),
# #                             'emotion': result.get('dominant_emotion', 'N/A')
# #                         }
# #                         self.stats['detections'].append(detection)
# #                         # Mantener solo últimos 100 registros
# #                         if len(self.stats['detections']) > 100:
# #                             self.stats['detections'].pop(0)
            
# #             # Dibujar información
# #             if self.last_analysis:
# #                 frame = self.draw_info(frame, self.last_analysis)
            
# #             self.frame_count += 1
# #             self.current_frame = frame.copy()
        
# #         return frame

# # # Instancia global
# # detector = FaceDetector()

# # def generate_frames():
# #     """Generador de frames para streaming"""
# #     while True:
# #         frame = detector.get_frame()
# #         if frame is None:
# #             break
        
# #         ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
# #         if not ret:
# #             continue
            
# #         frame_bytes = buffer.tobytes()
# #         yield (b'--frame\r\n'
# #                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# # @app.route('/')
# # def index():
# #     """Página principal"""
# #     return render_template('index.html')

# # @app.route('/video_feed')
# # def video_feed():
# #     """Stream de video"""
# #     return Response(
# #         generate_frames(),
# #         mimetype='multipart/x-mixed-replace; boundary=frame'
# #     )

# # @app.route('/start_camera', methods=['POST'])
# # def start_camera():
# #     """Inicia la cámara"""
# #     try:
# #         detector.start_camera(0)
# #         return jsonify({'success': True, 'message': 'Cámara iniciada'})
# #     except Exception as e:
# #         return jsonify({'success': False, 'message': str(e)})

# # @app.route('/stop_camera', methods=['POST'])
# # def stop_camera():
# #     """Detiene la cámara"""
# #     detector.stop_camera()
# #     return jsonify({'success': True, 'message': 'Cámara detenida'})

# # @app.route('/capture', methods=['POST'])
# # def capture():
# #     """Captura el frame actual"""
# #     if detector.current_frame is not None:
# #         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# #         filename = f"captura_{timestamp}.jpg"
        
# #         # Crear carpeta si no existe
# #         os.makedirs('capturas', exist_ok=True)
# #         filepath = os.path.join('capturas', filename)
        
# #         cv2.imwrite(filepath, detector.current_frame)
# #         return jsonify({'success': True, 'filename': filename})
# #     return jsonify({'success': False, 'message': 'No hay frame disponible'})

# # @app.route('/stats')
# # def get_stats():
# #     """Obtiene estadísticas"""
# #     return jsonify(detector.stats)
# # # """
# # # Aplicación de Detección Facial con Edad y Género
# # # Optimizada para CPU - Intel Core i7-10700

# # # Instalación requerida:
# # # pip install deepface
# # # pip install opencv-python
# # # pip install tf-keras
# # # """

# # # import cv2
# # # from deepface import DeepFace
# # # import time

# # # class FaceDetector:
    def __init__(self):
        self.cap = None
        self.analysis_result = {}
        self.analyzing = False
        self.last_analysis_time = 0
        self.analysis_interval = 1.0  # Analizar cada 1 segundo
        self.embeddings = []  # Lista de embeddings únicos (por compatibilidad)
        self.faces_dir = 'rostros_registrados'
        import os
        import glob
        import numpy as np
        os.makedirs(self.faces_dir, exist_ok=True)
        # Cargar embeddings existentes en memoria como lista de personas
        self.person_embeddings = []
        for npy in glob.glob(os.path.join(self.faces_dir, 'face_*.npy')):
            try:
                emb = np.load(npy)
                self.person_embeddings.append(emb)
            except Exception:
                pass
        # Mecanismo de candidato para evitar guardar inmediatamente
        self.candidate_embedding = None
        self.candidate_count = 0
        self.candidate_first_seen = 0
        self.prev_bbox = None
        # Parámetros de calidad y umbrales (ajustables)
        self.min_sharpness = 40.0
        self.min_brightness = 40
        self.max_brightness = 220
        self.save_consecutive_required = 4
        self.save_time_window = 8.0  # segundos
        self.embedding_threshold = 3.8
        self.candidate_similarity_thresh = 2.5
        self.candidate_iou_thresh = 0.6
        # Preferir similaridad coseno sobre L2 (más estable ante variaciones)
        self.use_cosine = True
        self.embedding_cosine_thresh = 0.30
        self.candidate_cosine_thresh = 0.25
        # Tracker simple por ID (centroid/IoU)
        self.tracks = {}  # id -> {bbox, last_seen, count, embedding, saved}
        self.next_track_id = 1
        self.track_max_age = 4.0
# # #             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
# # #             # Extraer información
# # #             age = result.get('age', 'N/A')
# # #             gender = result.get('dominant_gender', 'N/A')
# # #             emotion = result.get('dominant_emotion', 'N/A')
# # #             gender_confidence = result.get('gender', {})
            
# # #             # Calcular confianza de género
# # #             if isinstance(gender_confidence, dict):
# # #                 confidence = gender_confidence.get(gender, 0)
# # #                 gender_text = f"{gender} ({confidence:.1f}%)"
# # #             else:
# # #                 gender_text = gender
            
# # #             # Preparar texto
# # #             info_text = [
# # #                 f"Edad: {int(age)} anos",
# # #                 f"Genero: {gender_text}",
# # #                 f"Emocion: {emotion}"
# # #             ]
            
# # #             # Dibujar fondo para el texto
# # #             text_y = y - 10
# # #             for i, text in enumerate(info_text):
# # #                 text_y_pos = text_y - (len(info_text) - i) * 25
                
# # #                 # Fondo semitransparente
# # #                 (text_width, text_height), _ = cv2.getTextSize(
# # #                     text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
# # #                 )
# # #                 cv2.rectangle(
# # #                     frame,
# # #                     (x, text_y_pos - text_height - 5),
# # #                     (x + text_width + 5, text_y_pos + 5),
# # #                     (0, 0, 0),
# # #                     -1
# # #                 )
                
# # #                 # Texto
# # #                 cv2.putText(
# # #                     frame,
# # #                     text,
# # #                     (x + 2, text_y_pos),
# # #                     cv2.FONT_HERSHEY_SIMPLEX,
# # #                     0.6,
# # #                     (255, 255, 255),
# # #                     2
# # #                 )
        
# # #         return frame
    
# # #     def run(self, source=0):
# # #         """
# # #         Ejecuta la detección en tiempo real
# # #         source: 0 para webcam, o ruta de video/imagen
# # #         """
# # #         # Abrir fuente de video
# # #         cap = cv2.VideoCapture(source)
        
# # #         if not cap.isOpened():
# # #             print("Error: No se pudo abrir la cámara")
# # #             return
        
# # #         # Configurar resolución (menor = más rápido)
# # #         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# # #         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
# # #         print("Presiona 'q' para salir")
# # #         print("Presiona 's' para guardar captura")
# # #         print("Presiona 'p' para pausar/reanudar")
        
# # #         paused = False
# # #         fps_time = time.time()
        
# # #         while True:
# # #             if not paused:
# # #                 ret, frame = cap.read()
                
# # #                 if not ret:
# # #                     print("Error: No se pudo leer el frame")
# # #                     break
                
# # #                 # Procesar solo cada N frames
# # #                 if self.frame_count % self.frame_skip == 0:
# # #                     self.last_analysis = self.analyze_frame(frame)
                
# # #                 # Dibujar información en el frame
# # #                 if self.last_analysis:
# # #                     frame = self.draw_info(frame, self.last_analysis)
                
# # #                 # Calcular FPS
# # #                 current_time = time.time()
# # #                 fps = 1 / (current_time - fps_time)
# # #                 fps_time = current_time
                
# # #                 # Mostrar FPS
# # #                 cv2.putText(
# # #                     frame,
# # #                     f"FPS: {fps:.1f}",
# # #                     (10, 30),
# # #                     cv2.FONT_HERSHEY_SIMPLEX,
# # #                     0.7,
# # #                     (0, 255, 255),
# # #                     2
# # #                 )
                
# # #                 # Instrucciones
# # #                 cv2.putText(
# # #                     frame,
# # #                     "q: Salir | s: Guardar | p: Pausar",
# # #                     (10, frame.shape[0] - 10),
# # #                     cv2.FONT_HERSHEY_SIMPLEX,
# # #                     0.5,
# # #                     (255, 255, 255),
# # #                     1
# # #                 )
                
# # #                 self.frame_count += 1
            
# # #             # Mostrar frame
# # #             cv2.imshow('Detector Facial - Edad y Genero', frame)
            
# # #             # Capturar teclas
# # #             key = cv2.waitKey(1) & 0xFF
            
# # #             if key == ord('q'):
# # #                 break
# # #             elif key == ord('s'):
# # #                 filename = f"captura_{int(time.time())}.jpg"
# # #                 cv2.imwrite(filename, frame)
# # #                 print(f"Captura guardada: {filename}")
# # #             elif key == ord('p'):
# # #                 paused = not paused
# # #                 print("Pausado" if paused else "Reanudado")
        
# # #         # Liberar recursos
# # #         cap.release()
# # #         cv2.destroyAllWindows()
# # #         print("Aplicación cerrada")

# # # def main():
# # #     """Función principal"""
# # #     print("="*50)
# # #     print("DETECTOR FACIAL - EDAD Y GÉNERO")
# # #     print("="*50)
# # #     print("\nOpciones:")
# # #     print("1. Webcam en tiempo real")
# # #     print("2. Analizar video desde archivo")
# # #     print("3. Analizar imagen")
    
# # #     choice = input("\nSelecciona una opción (1-3): ").strip()
    
# # #     detector = FaceDetector()
    
# # #     if choice == "1":
# # #         print("\nIniciando webcam...")
# # #         detector.run(source=0)
    
# # #     elif choice == "2":
# # #         video_path = input("Ingresa la ruta del video: ").strip()
# # #         print(f"\nAnalizando video: {video_path}")
# # #         detector.run(source=video_path)
    
# # #     elif choice == "3":
# # #         img_path = input("Ingresa la ruta de la imagen: ").strip()
# # #         print(f"\nAnalizando imagen: {img_path}")
        
# # #         # Leer imagen
# # #         frame = cv2.imread(img_path)
# # #         if frame is None:
# # #             print("Error: No se pudo cargar la imagen")
# # #             return
        
# # #         # Analizar
# # #         results = detector.analyze_frame(frame)
        
# # #         # Dibujar información
# # #         if results:
# # #             frame = detector.draw_info(frame, results)
            
# # #             # Mostrar
# # #             cv2.imshow('Resultado - Presiona cualquier tecla para salir', frame)
# # #             cv2.waitKey(0)
# # #             cv2.destroyAllWindows()
            
# # #             # Guardar
# # #             output_path = f"resultado_{int(time.time())}.jpg"
# # #             cv2.imwrite(output_path, frame)
# # #             print(f"\nResultado guardado: {output_path}")
# # #         else:
# # #             print("No se detectaron rostros en la imagen")
    
# # #     else:
# # #         print("Opción no válida")

# # # if __name__ == "__main__":
# # #     main()
# # # # from deepface import DeepFace
# # # # import cv2

# # # # # Analizar una imagen
# # # # result = DeepFace.analyze(
# # # #     img_path="foto.jpg",
# # # #     actions=['age', 'gender', 'emotion', 'race'],
# # # #     enforce_detection=False
# # # # )

# # # # print(f"Edad: {result[0]['age']}")
# # # # print(f"Género: {result[0]['dominant_gender']}")
# # # # print(f"Emoción: {result[0]['dominant_emotion']}")