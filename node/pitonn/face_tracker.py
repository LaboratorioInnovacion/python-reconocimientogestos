import os
import json
import cv2
import numpy as np
from deepface import DeepFace
from datetime import datetime

SNAPSHOT_DIR = r"C:/ProgramData/Camlytics_v3/Data/StorageData/Snapshot"
OUTPUT_JSON = "people.json"

MODEL_NAME = "Facenet"
DETECTOR = "retinaface"
DISTANCE_THRESHOLD = 0.58
MIN_FACE_SIZE = 80  # px

people = []
person_id_counter = 1


def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_embedding(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        h, w, _ = img.shape

        detections = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend=DETECTOR,
            enforce_detection=True
        )

        if not detections:
            return None

        face = detections[0]
        fa = face["facial_area"]

        face_w = fa["w"]
        face_h = fa["h"]

        if face_w < MIN_FACE_SIZE or face_h < MIN_FACE_SIZE:
            return None  # cara demasiado chica

        rep = DeepFace.represent(
            img_path=image_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            enforce_detection=True,
            align=True
        )

        return np.array(rep[0]["embedding"])

    except Exception:
        return None


def find_person(embedding):
    for person in people:
        dist = cosine_distance(embedding, person["embedding"])
        if dist < DISTANCE_THRESHOLD:
            return person
    return None


print(f"Observando imágenes en: {SNAPSHOT_DIR}")

for root, dirs, files in os.walk(SNAPSHOT_DIR):
    for file in files:
        if not file.lower().endswith(".jpg"):
            continue

        image_path = os.path.join(root, file)
        embedding = get_embedding(image_path)

        if embedding is None:
            continue

        person = find_person(embedding)

        if person:
            person["photos"].append(image_path)
            person["last_seen"] = datetime.now().isoformat()
        else:
            people.append({
                "id": person_id_counter,
                "photos": [image_path],
                "embedding": embedding,
                "first_seen": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat()
            })
            person_id_counter += 1


# limpiar embeddings antes de guardar
for p in people:
    p["embedding"] = p["embedding"].tolist()

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(people, f, indent=2, ensure_ascii=False)

print("\nRESULTADO FINAL")
for p in people:
    print(f"Persona {p['id']} | Fotos: {len(p['photos'])}")

print(f"\nArchivo generado: {OUTPUT_JSON}")

# import os
# import json
# import cv2
# import numpy as np
# from deepface import DeepFace
# from datetime import datetime

# SNAPSHOT_DIR = r"C:/ProgramData/Camlytics_v3/Data/StorageData/Snapshot"
# OUTPUT_JSON = "people.json"

# MODEL_NAME = "Facenet"
# DISTANCE_THRESHOLD = 0.65   # ajustar si hace falta

# people = []
# person_id_counter = 1


# def cosine_distance(a, b):
#     return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# def get_embedding(image_path):
#     try:
#         img = cv2.imread(image_path)
#         if img is None:
#             return None

#         result = DeepFace.represent(
#             img_path=image_path,
#             model_name=MODEL_NAME,
#             enforce_detection=False
#         )
#         return np.array(result[0]["embedding"])
#     except Exception as e:
#         print(f"Error procesando {image_path}: {e}")
#         return None


# def find_person(embedding):
#     for person in people:
#         dist = cosine_distance(embedding, person["embedding"])
#         if dist < DISTANCE_THRESHOLD:
#             return person
#     return None


# print(f"Observando imágenes en: {SNAPSHOT_DIR}")

# for root, dirs, files in os.walk(SNAPSHOT_DIR):
#     for file in files:
#         if not file.lower().endswith(".jpg"):
#             continue

#         image_path = os.path.join(root, file)
#         embedding = get_embedding(image_path)

#         if embedding is None:
#             continue

#         person = find_person(embedding)

#         if person:
#             person["photos"].append(image_path)
#             person["last_seen"] = datetime.now().isoformat()
#         else:
#             new_person = {
#                 "id": person_id_counter,
#                 "photos": [image_path],
#                 "embedding": embedding.tolist(),
#                 "first_seen": datetime.now().isoformat(),
#                 "last_seen": datetime.now().isoformat()
#             }
#             people.append(new_person)
#             person_id_counter += 1


# # quitar embeddings antes de guardar
# for p in people:
#     del p["embedding"]

# with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
#     json.dump(people, f, indent=2, ensure_ascii=False)

# print("\nRESULTADO FINAL")
# for p in people:
#     print(f"Persona {p['id']} | Fotos: {len(p['photos'])}")

# print(f"\nArchivo generado: {OUTPUT_JSON}")

# # import os
# # import cv2
# # import json
# # from deepface import DeepFace

# # SNAPSHOT_DIR = r"C:\ProgramData\Camlytics_v3\Data\StorageData\Snapshot"
# # DB_PATH = "people_db.json"

# # people = []

# # if os.path.exists(DB_PATH):
# #     with open(DB_PATH, "r") as f:
# #         people = json.load(f)

# # def is_same_person(img_path, person):
# #     try:
# #         result = DeepFace.verify(
# #             img1_path=img_path,
# #             img2_path=person["images"][0],
# #             enforce_detection=False
# #         )
# #         return result["verified"]
# #     except:
# #         return False

# # def process_image(img_path):
# #     global people

# #     for person in people:
# #         if is_same_person(img_path, person):
# #             person["images"].append(img_path)
# #             return

# #     # Nueva persona
# #     people.append({
# #         "id": len(people) + 1,
# #         "images": [img_path]
# #     })

# # # Recorre carpetas de Camlytics
# # for root, dirs, files in os.walk(SNAPSHOT_DIR):
# #     for file in files:
# #         if file.lower().endswith(".jpg"):
# #             full_path = os.path.join(root, file)
# #             process_image(full_path)

# # # Guardar resultados
# # with open(DB_PATH, "w") as f:
# #     json.dump(people, f, indent=2)

# # print("✔ Personas detectadas:", len(people))
# # for p in people:
# #     print(f"Persona {p['id']} → {len(p['images'])} fotos")
