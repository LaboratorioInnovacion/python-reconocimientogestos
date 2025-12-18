# """
# Sistema de Análisis y Deduplicación de Rostros para Camlytics v3
# Procesa capturas completas, extrae múltiples rostros y evita duplicados
# Monitorea: C:\ProgramData\Camlytics_v3\Data\StorageData\Snapshot\
# """

import os
import time
import json
import sqlite3
import pickle
from datetime import datetime
from pathlib import Path
import shutil
import cv2
import numpy as np
import face_recognition
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class FaceDatabase:
    """Maneja la base de datos de rostros únicos"""
    
    def __init__(self, db_path='camlytics_faces.db', encodings_dir='face_encodings'):
        self.db_path = db_path
        self.encodings_dir = Path(encodings_dir)
        self.encodings_dir.mkdir(exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Inicializa la base de datos SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS unique_faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT UNIQUE,
                first_seen TIMESTAMP,
                last_seen TIMESTAMP,
                detection_count INTEGER DEFAULT 1,
                encoding_file TEXT,
                sample_image TEXT,
                face_crop TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT,
                source_image TEXT,
                face_location TEXT,
                timestamp TIMESTAMP,
                is_duplicate BOOLEAN,
                similarity_score REAL,
                FOREIGN KEY (person_id) REFERENCES unique_faces(person_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT UNIQUE,
                faces_found INTEGER,
                new_faces INTEGER,
                duplicate_faces INTEGER,
                processed_at TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stats (
                id INTEGER PRIMARY KEY,
                total_unique INTEGER,
                total_detections INTEGER,
                images_processed INTEGER,
                last_update TIMESTAMP
            )
        ''')
        
        # Inicializar stats si no existe
        cursor.execute('SELECT COUNT(*) FROM stats')
        if cursor.fetchone()[0] == 0:
            cursor.execute('''
                INSERT INTO stats (id, total_unique, total_detections, images_processed, last_update)
                VALUES (1, 0, 0, 0, ?)
            ''', (datetime.now(),))
        
        conn.commit()
        conn.close()
    
    def save_encoding(self, person_id, encoding):
        """Guarda el encoding facial"""
        encoding_file = self.encodings_dir / f"{person_id}.pkl"
        with open(encoding_file, 'wb') as f:
            pickle.dump(encoding, f)
        return str(encoding_file)
    
    def load_encoding(self, encoding_file):
        """Carga un encoding facial"""
        with open(encoding_file, 'rb') as f:
            return pickle.load(f)
    
    def get_all_encodings(self):
        """Obtiene todos los encodings de la base de datos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT person_id, encoding_file FROM unique_faces')
        
        encodings = []
        person_ids = []
        
        for person_id, encoding_file in cursor.fetchall():
            try:
                encoding = self.load_encoding(encoding_file)
                encodings.append(encoding)
                person_ids.append(person_id)
            except Exception as e:
                print(f"Error cargando encoding para {person_id}: {e}")
        
        conn.close()
        return encodings, person_ids
    
    def add_unique_face(self, encoding, source_image, face_location, face_crop_path=None):
        """Agrega un nuevo rostro único"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Obtener el siguiente ID
        cursor.execute('SELECT COUNT(*) FROM unique_faces')
        count = cursor.fetchone()[0] + 1
        person_id = f"PERSON_{count:05d}"
        
        # Guardar encoding
        encoding_file = self.save_encoding(person_id, encoding)
        
        # Insertar en la base de datos
        now = datetime.now()
        cursor.execute('''
            INSERT INTO unique_faces 
            (person_id, first_seen, last_seen, detection_count, encoding_file, sample_image, face_crop)
            VALUES (?, ?, ?, 1, ?, ?, ?)
        ''', (person_id, now, now, encoding_file, source_image, face_crop_path))
        
        # Actualizar estadísticas
        cursor.execute('''
            UPDATE stats 
            SET total_unique = total_unique + 1,
                total_detections = total_detections + 1,
                last_update = ?
            WHERE id = 1
        ''', (now,))
        
        conn.commit()
        conn.close()
        
        return person_id
    
    def update_face(self, person_id):
        """Actualiza la información de un rostro existente"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now()
        cursor.execute('''
            UPDATE unique_faces 
            SET last_seen = ?, detection_count = detection_count + 1
            WHERE person_id = ?
        ''', (now, person_id))
        
        cursor.execute('''
            UPDATE stats 
            SET total_detections = total_detections + 1,
                last_update = ?
            WHERE id = 1
        ''', (now,))
        
        conn.commit()
        conn.close()
    
    def log_detection(self, person_id, source_image, face_location, is_duplicate, similarity_score):
        """Registra una detección"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO detections 
            (person_id, source_image, face_location, timestamp, is_duplicate, similarity_score)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (person_id, source_image, json.dumps(face_location), datetime.now(), is_duplicate, similarity_score))
        
        conn.commit()
        conn.close()
    
    def log_processed_image(self, image_path, faces_found, new_faces, duplicate_faces):
        """Registra una imagen procesada"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO processed_images 
            (image_path, faces_found, new_faces, duplicate_faces, processed_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (image_path, faces_found, new_faces, duplicate_faces, datetime.now()))
        
        cursor.execute('''
            UPDATE stats 
            SET images_processed = images_processed + 1,
                last_update = ?
            WHERE id = 1
        ''', (datetime.now(),))
        
        conn.commit()
        conn.close()
    
    def get_stats(self):
        """Obtiene estadísticas"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT total_unique, total_detections, images_processed FROM stats WHERE id = 1')
        unique, detections, images = cursor.fetchone()
        
        cursor.execute('SELECT COUNT(*) FROM detections WHERE is_duplicate = 1')
        duplicates = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_unique': unique,
            'total_detections': detections,
            'images_processed': images,
            'duplicates_filtered': duplicates
        }
    
    def export_report(self, output_file='camlytics_report.json'):
        """Exporta un reporte completo en JSON"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Obtener todas las personas únicas
        cursor.execute('''
            SELECT person_id, first_seen, last_seen, detection_count, sample_image
            FROM unique_faces
            ORDER BY detection_count DESC
        ''')
        
        unique_faces = []
        for row in cursor.fetchall():
            unique_faces.append({
                'person_id': row[0],
                'first_seen': row[1],
                'last_seen': row[2],
                'detection_count': row[3],
                'sample_image': row[4]
            })
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'statistics': self.get_stats(),
            'unique_faces': unique_faces
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        conn.close()
        print(f"📄 Reporte exportado a: {output_file}")
        return output_file


class FaceAnalyzer:
    """Analiza y compara rostros de imágenes completas"""
    
    def __init__(self, database, similarity_threshold=0.6, output_crops_dir='face_crops'):
        self.db = database
        self.similarity_threshold = similarity_threshold
        self.output_crops_dir = Path(output_crops_dir)
        self.output_crops_dir.mkdir(exist_ok=True)
    
    def save_face_crop(self, image, face_location, person_id, image_name):
        """Guarda un recorte del rostro detectado"""
        try:
            top, right, bottom, left = face_location
            
            # Agregar margen
            margin = 20
            height, width = image.shape[:2]
            top = max(0, top - margin)
            bottom = min(height, bottom + margin)
            left = max(0, left - margin)
            right = min(width, right + margin)
            
            face_crop = image[top:bottom, left:right]
            
            # Guardar
            crop_filename = f"{person_id}_{Path(image_name).stem}_crop.jpg"
            crop_path = self.output_crops_dir / crop_filename
            cv2.imwrite(str(crop_path), cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))
            
            return str(crop_path)
        except Exception as e:
            print(f"   ⚠️  Error guardando recorte: {e}")
            return None
    
    def extract_face_encodings(self, image_path):
        """Extrae todos los rostros de una imagen completa"""
        try:
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image, model='hog')
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            return image, face_locations, face_encodings
        
        except Exception as e:
            print(f"❌ Error procesando {Path(image_path).name}: {e}")
            return None, [], []
    
    def find_match(self, face_encoding):
        """Busca si el rostro ya existe en la base de datos"""
        encodings, person_ids = self.db.get_all_encodings()
        
        if len(encodings) == 0:
            return None, 0.0
        
        # Comparar con todos los rostros conocidos
        face_distances = face_recognition.face_distance(encodings, face_encoding)
        
        # Encontrar el más similar
        min_distance = np.min(face_distances)
        similarity_score = 1 - min_distance
        
        if min_distance < self.similarity_threshold:
            best_match_idx = np.argmin(face_distances)
            return person_ids[best_match_idx], similarity_score
        
        return None, similarity_score
    
    def process_image(self, image_path):
        """Procesa una imagen completa de Camlytics con múltiples rostros"""
        filename = Path(image_path).name
        print(f"\n📸 Procesando: {filename}")
        
        # Extraer todos los rostros
        image, face_locations, face_encodings = self.extract_face_encodings(image_path)
        
        if image is None:
            return None
        
        faces_found = len(face_encodings)
        print(f"   👥 Rostros detectados: {faces_found}")
        
        if faces_found == 0:
            print(f"   ⚠️  No se detectaron rostros en esta imagen")
            self.db.log_processed_image(image_path, 0, 0, 0)
            return {'faces': [], 'new': 0, 'duplicates': 0}
        
        results = []
        new_count = 0
        duplicate_count = 0
        
        # Procesar cada rostro detectado
        for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations), 1):
            print(f"   Rostro {i}/{faces_found}:", end=" ")
            
            # Buscar coincidencia
            match_id, similarity = self.find_match(face_encoding)
            
            if match_id:
                # Rostro duplicado
                print(f"🔄 DUPLICADO - {match_id} (similitud: {similarity:.2%})")
                self.db.update_face(match_id)
                self.db.log_detection(match_id, image_path, face_location, True, similarity)
                duplicate_count += 1
                
                results.append({
                    'status': 'duplicate',
                    'person_id': match_id,
                    'similarity': similarity,
                    'location': face_location
                })
            else:
                # Nuevo rostro único
                crop_path = self.save_face_crop(image, face_location, f"TEMP_{i}", filename)
                person_id = self.db.add_unique_face(face_encoding, image_path, face_location, crop_path)
                
                # Renombrar el crop con el ID correcto
                if crop_path:
                    new_crop_path = str(crop_path).replace(f"TEMP_{i}", person_id)
                    if Path(crop_path).exists():
                        shutil.move(crop_path, new_crop_path)
                        crop_path = new_crop_path
                
                print(f"✅ NUEVA PERSONA - {person_id}")
                self.db.log_detection(person_id, image_path, face_location, False, similarity)
                new_count += 1
                
                results.append({
                    'status': 'new',
                    'person_id': person_id,
                    'similarity': similarity,
                    'location': face_location,
                    'crop_path': crop_path
                })
        
        # Registrar imagen procesada
        self.db.log_processed_image(image_path, faces_found, new_count, duplicate_count)
        
        print(f"   📊 Resumen: {new_count} nuevos, {duplicate_count} duplicados")
        
        return {
            'faces': results,
            'new': new_count,
            'duplicates': duplicate_count,
            'total': faces_found
        }


class CamlyticsMonitor(FileSystemEventHandler):
    """Monitorea la carpeta de Camlytics"""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.valid_extensions = {'.jpg', '.jpeg', '.png'}
        self.processing = set()
        self.processed_files = set()
    
    def on_created(self, event):
        """Se ejecuta cuando se crea un archivo nuevo"""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Solo procesar imágenes (ignorar .meta)
        if file_path.suffix.lower() not in self.valid_extensions:
            return
        
        # Evitar procesar dos veces
        if str(file_path) in self.processing or str(file_path) in self.processed_files:
            return
        
        self.processing.add(str(file_path))
        
        # Esperar a que Camlytics termine de escribir el archivo
        time.sleep(2)
        
        try:
            if not file_path.exists():
                return
            
            # Procesar la imagen
            result = self.analyzer.process_image(str(file_path))
            
            if result:
                self.processed_files.add(str(file_path))
        
        except Exception as e:
            print(f"❌ Error: {e}")
        
        finally:
            self.processing.discard(str(file_path))


class FaceCounterApp:
    """Aplicación principal"""
    
    def __init__(self, watch_folder, similarity_threshold=0.6):
        self.watch_folder = Path(watch_folder)
        
        if not self.watch_folder.exists():
            print(f"⚠️  ADVERTENCIA: La carpeta no existe: {self.watch_folder}")
            print(f"   Creando carpeta...")
            self.watch_folder.mkdir(parents=True, exist_ok=True)
        
        print("=" * 80)
        print("🚀 SISTEMA DE CONTEO DE ROSTROS - CAMLYTICS V3")
        print("=" * 80)
        
        # Inicializar componentes
        self.db = FaceDatabase()
        self.analyzer = FaceAnalyzer(self.db, similarity_threshold)
        self.monitor = CamlyticsMonitor(self.analyzer)
        
        print(f"📂 Carpeta: {self.watch_folder}")
        print(f"🎯 Umbral de similitud: {similarity_threshold}")
        print(f"💾 Base de datos: camlytics_faces.db")
        print(f"📁 Recortes guardados en: face_crops/")
        print("=" * 80)
    
    def process_existing_images(self):
        """Procesa imágenes existentes en la carpeta"""
        image_files = []
        for ext in self.monitor.valid_extensions:
            image_files.extend(self.watch_folder.glob(f"*{ext}"))
        
        if image_files:
            print(f"\n📦 Encontradas {len(image_files)} imágenes existentes")
            print(f"⏳ Procesando...\n")
            
            for i, img_path in enumerate(image_files, 1):
                print(f"\n[{i}/{len(image_files)}]", end=" ")
                self.analyzer.process_image(str(img_path))
            
            print("\n\n✅ Procesamiento de imágenes existentes completado")
        else:
            print(f"\n📭 No hay imágenes existentes en la carpeta")
    
    def show_stats(self):
        """Muestra estadísticas"""
        stats = self.db.get_stats()
        print("\n" + "=" * 80)
        print("📊 ESTADÍSTICAS DEL SISTEMA")
        print("=" * 80)
        print(f"   👤 Personas únicas detectadas:  {stats['total_unique']}")
        print(f"   📸 Total de detecciones:        {stats['total_detections']}")
        print(f"   🔄 Duplicados filtrados:        {stats['duplicates_filtered']}")
        print(f"   🖼️  Imágenes procesadas:         {stats['images_processed']}")
        
        if stats['total_unique'] > 0:
            avg_detections = stats['total_detections'] / stats['total_unique']
            print(f"   📈 Promedio detecciones/persona: {avg_detections:.1f}")
        
        print("=" * 80 + "\n")
    
    def run(self):
        """Ejecuta el sistema"""
        # Procesar imágenes existentes
        self.process_existing_images()
        self.show_stats()
        
        # Iniciar monitor en tiempo real
        observer = Observer()
        observer.schedule(self.monitor, str(self.watch_folder), recursive=False)
        observer.start()
        
        print("👀 MONITOR ACTIVO - Esperando nuevas capturas de Camlytics...")
        print("   Presiona Ctrl+C para detener\n")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n🛑 Deteniendo monitor...")
            observer.stop()
        
        observer.join()
        
        # Estadísticas finales
        self.show_stats()
        self.db.export_report()
        print("✅ Sistema detenido - Reporte generado")


if __name__ == "__main__":
    # ============= CONFIGURACIÓN =============
    
    # Ruta de Camlytics (ajusta el GUID a tu cámara específica)
    CAMLYTICS_PATH = r"C:\ProgramData\Camlytics_v3\Data\StorageData\Snapshot\2ace267d-3576-437e-a31f-1daabeea9ba7"
    
    # Umbral de similitud
    # 0.5 = MUY estricto (menos duplicados, pero puede contar la misma persona 2 veces)
    # 0.6 = EQUILIBRADO (recomendado)
    # 0.7 = Permisivo (más chance de agrupar personas diferentes)
    SIMILARITY_THRESHOLD = 0.6
    
    # ========================================
    
    app = FaceCounterApp(
        watch_folder=CAMLYTICS_PATH,
        similarity_threshold=SIMILARITY_THRESHOLD
    )
    
    app.run()