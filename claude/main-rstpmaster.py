import cv2
from deepface import DeepFace
import threading
import time
import os
import glob
import numpy as np
import json
import subprocess
import shutil
import signal


class FaceAnalyzer:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = None

        self.analysis_result = {}
        self.analyzing = False
        self.last_analysis_time = 0
        self.analysis_interval = 1.0

        self.faces_dir = 'rostros_registrados'
        os.makedirs(self.faces_dir, exist_ok=True)

        # =================== EMBEDDINGS REGISTRADOS ===================
        self.person_embeddings = []
        for npy in glob.glob(os.path.join(self.faces_dir, 'face_*.npy')):
            try:
                self.person_embeddings.append(np.load(npy))
            except:
                pass

        # =================== CALIDAD ===================
        self.min_sharpness = 40.0
        self.min_brightness = 40
        self.max_brightness = 220

        # =================== GUARDADO ===================
        self.save_consecutive_required = 6

        # =================== EMBEDDING THRESHOLDS ===================
        self.embedding_cosine_thresh = 0.28
        self.candidate_cosine_thresh = 0.22
        self.candidate_iou_thresh = 0.6

        # =================== TRACKER ===================
        self.tracks = {}
        self.next_track_id = 1
        self.track_max_age = 4.0

        self.lock = threading.Lock()

    # ===============================================================
    # UTILIDADES
    # ===============================================================

    def _iou(self, A, B):
        xA = max(A[0], B[0])
        yA = max(A[1], B[1])
        xB = min(A[0] + A[2], B[0] + B[2])
        yB = min(A[1] + A[3], B[1] + B[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        union = A[2]*A[3] + B[2]*B[3] - inter
        return inter / union if union > 0 else 0

    def _cosine(self, a, b):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        return 1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # ===============================================================
    # ANALISIS
    # ===============================================================

    def analyze_face(self, frame):
        try:
            faces = DeepFace.analyze(
                frame,
                actions=['age', 'gender', 'emotion'],
                detector_backend='opencv',
                enforce_detection=False,
                silent=True
            )

            if not faces:
                return

            face = max(faces, key=lambda f: f['region']['w'] * f['region']['h'])
            r = face['region']
            x, y, w, h = r['x'], r['y'], r['w'], r['h']
            face_img = frame[y:y+h, x:x+w]

            if face_img.size == 0:
                return

            # =================== CALIDAD ===================
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
            bright = gray.mean()

            if sharp < self.min_sharpness or not self.min_brightness <= bright <= self.max_brightness:
                return

            face160 = cv2.resize(face_img, (160, 160))

            # =================== EMBEDDING ===================
            rep = DeepFace.represent(
                face160,
                model_name='Facenet',
                enforce_detection=False
            )

            emb = np.array(rep[0]['embedding'], dtype=np.float32)

            # =================== DUPLICADO GLOBAL ===================
            for e in self.person_embeddings:
                if self._cosine(emb, e) < self.embedding_cosine_thresh:
                    return

            now = time.time()
            bbox = (x, y, w, h)

            with self.lock:
                # limpiar tracks viejos
                self.tracks = {
                    k: v for k, v in self.tracks.items()
                    if now - v['last'] < self.track_max_age
                }

                best_id, best_iou = None, 0.0
                for tid, t in self.tracks.items():
                    i = self._iou(t['bbox'], bbox)
                    if i > best_iou:
                        best_id, best_iou = tid, i

                if best_id and best_iou >= self.candidate_iou_thresh:
                    t = self.tracks[best_id]
                    if self._cosine(emb, t['emb']) < self.candidate_cosine_thresh:
                        t['count'] += 1
                        t['emb'] = (t['emb'] + emb) * 0.5
                    else:
                        t['count'] = 1
                        t['emb'] = emb.copy()
                    t['last'] = now
                else:
                    self.tracks[self.next_track_id] = {
                        'bbox': bbox,
                        'emb': emb.copy(),
                        'count': 1,
                        'last': now,
                        'saved': False
                    }
                    best_id = self.next_track_id
                    self.next_track_id += 1

                t = self.tracks[best_id]

                # =================== GUARDADO ===================
                if t['count'] >= self.save_consecutive_required and not t['saved']:
                    ts = int(time.time() * 1000)
                    base = os.path.join(self.faces_dir, f"face_{ts}")

                    cv2.imwrite(base + ".jpg", face_img)
                    np.save(base + ".npy", t['emb'])

                    json.dump(
                        {
                            'edad': face.get('age'),
                            'genero': face.get('dominant_gender'),
                            'emocion': face.get('dominant_emotion')
                        },
                        open(base + ".json", "w", encoding="utf-8"),
                        indent=2,
                        ensure_ascii=False
                    )

                    self.person_embeddings.append(t['emb'].copy())
                    t['saved'] = True

                    print(f"✅ Rostro único guardado: {base}")

        except Exception as e:
            print(f"❌ Error analyze_face: {e}")
        finally:
            self.analyzing = False

    # ===============================================================
    # RTSP LOOP
    # ===============================================================

    def connect(self):
        # Intentar abrir directamente con backend FFMPEG
        try:
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # pequeña espera para que se estabilice
            time.sleep(0.3)
            if self.cap.isOpened():
                return
        except Exception:
            pass

        # Intentar abrir con backend por defecto
        try:
            self.cap = cv2.VideoCapture(self.rtsp_url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            time.sleep(0.3)
            if self.cap.isOpened():
                return
        except Exception:
            pass

        # Si no se pudo abrir, intentar re-stream con ffmpeg a un puerto local
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path:
            self.start_ffmpeg_restream(ffmpeg_path)
            # abrir stream local
            local_url = 'tcp://127.0.0.1:10000'
            try:
                self.cap = cv2.VideoCapture(local_url, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                time.sleep(0.3)
            except Exception:
                pass

    def run(self):
        self.connect()

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        print("📡 Conectado a RTSP (intentando establecer)")

        reconnect_backoff = 1.0
        lost_frames = 0

        while True:
            if self.cap is None or not self.cap.isOpened():
                print("🔄 Reconectando RTSP...")
                time.sleep(reconnect_backoff)
                reconnect_backoff = min(reconnect_backoff * 2, 16)
                self.connect()
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                lost_frames += 1
                if lost_frames % 10 == 1:
                    print(f"⚠ Frame perdido (contador={lost_frames})")
                # si hay muchos frames perdidos forzar reconexión
                if lost_frames >= 8:
                    print("🔁 Muchos frames perdidos — reiniciando conexión")
                    try:
                        if self.cap:
                            self.cap.release()
                    except:
                        pass
                    self.stop_ffmpeg_restream()
                    self.cap = None
                    time.sleep(1.0)
                else:
                    time.sleep(0.05)
                continue
            else:
                # frame leído correctamente
                lost_frames = 0
                reconnect_backoff = 1.0

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

            now = time.time()
            if (
                len(faces) > 0
                and not self.analyzing
                and now - self.last_analysis_time > self.analysis_interval
            ):
                self.analyzing = True
                self.last_analysis_time = now
                threading.Thread(
                    target=self.analyze_face,
                    args=(frame.copy(),),
                    daemon=True
                ).start()

            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"Rostros detectados: {len(faces)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow("RTSP Facial", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    # ===============================================================
    # FFMPEG RESTREAM (fallback)
    # ===============================================================
    def start_ffmpeg_restream(self, ffmpeg_path):
        # Lanza un proceso ffmpeg que re-emite el stream RTSP localmente por TCP
        if getattr(self, '_ffmpeg_proc', None) is not None:
            return

        cmd = [
            ffmpeg_path,
            '-rtsp_transport', 'tcp',
            '-i', self.rtsp_url,
            '-c:v', 'copy',
            '-f', 'mpegts',
            'tcp://127.0.0.1:10000'
        ]

        try:
            self._ffmpeg_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            print('🔁 ffmpeg restream iniciado (tcp://127.0.0.1:10000)')
        except Exception as e:
            print(f'❌ No se pudo iniciar ffmpeg restream: {e}')
            self._ffmpeg_proc = None

    def stop_ffmpeg_restream(self):
        p = getattr(self, '_ffmpeg_proc', None)
        if p is None:
            return
        try:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
                time.sleep(0.3)
                if p.poll() is None:
                    p.terminate()
                    time.sleep(0.2)
                    if p.poll() is None:
                        p.kill()
        except Exception:
            pass
        self._ffmpeg_proc = None


# ===============================================================
# MAIN
# ===============================================================

if __name__ == "__main__":
    RTSP_URL = "rtsp://admin:Nodo2023@192.168.1.213:554/dev.hik-connect.com/channels/101"
    FaceAnalyzer(RTSP_URL).run()
