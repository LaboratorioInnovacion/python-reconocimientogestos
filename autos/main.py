from ultralytics import YOLO
import cv2
from yt_dlp import YoutubeDL
import time
import numpy as np

# ---------- CONFIGURACIÓN GENERAL ----------

YOUTUBE_URL = "https://www.youtube.com/watch?v=NK3S_T0Sabk"

# Modo calibracion de semáforos:
MODO_CALIBRACION = False   # True para seguir marcando ROIs de semáforos

# ZONAS DONDE REALMENTE HAY SEMÁFOROS
SEMAFORO_ZONES = [
    {"roi": (1846, 446, 1876, 499), "tipo": "3", "id": "S_CEN3"},
    {"roi": (1328, 300, 1359, 367), "tipo": "3", "id": "S_CEN2"},
    {"roi": (1149, 285, 1177, 353), "tipo": "3", "id": "S_CEN1"},
    {"roi": (924, 266, 955, 330),   "tipo": "3", "id": "S_IZQ2"},
    {"roi": (225, 582, 259, 673),   "tipo": "3", "id": "S_IZQ1"},
    {"roi": (1846, 502, 1875, 545), "tipo": "2", "id": "S_DER"},
]

# LÍNEAS DE CRUCE POR DIRECCIÓN
# sense:
#   1  -> detecta cruce cuando pasa de "lado negativo" a "lado positivo"
#  -1  -> igual, pero con la línea orientada al revés
LANES = [
    {
        "name": "recto",
        "p1": (1026, 652),
        "p2": (1607, 662),
        "semaforos": ["S_CEN1", "S_CEN2"],
        "sense": 1,
    },
    {
        "name": "izquierda",
        "p1": (269, 723),
        "p2": (325, 659),
        "semaforos": ["S_IZQ1", "S_IZQ2"],
        "sense": 1,   # si sigue fallando, probá cambiando a -1
    },
    {
        "name": "derecha",
        "p1": (1700, 709),
        "p2": (1920, 729),
        "semaforos": ["S_DER"],
        "sense": 1,
    },
]

# Traducción de clases YOLO -> español
class_map = {
    "person": "persona",
    "car": "auto",
    "truck": "camión",
    "bus": "colectivo",
    "motorcycle": "moto",
    "bicycle": "bicicleta",
    "traffic light": "semáforo",
}

prev_time = 0
click_points = []  # para ROIs de semáforos en modo calibración

# memoria para tracking y cruces
last_side = {}     # (track_id, lane_name) -> side
counts = {
    "recto": {"ROJO": 0, "AMARILLO": 0, "VERDE": 0},
    "izquierda": {"ROJO": 0, "AMARILLO": 0, "VERDE": 0},
    "derecha": {"ROJO": 0, "AMARILLO": 0, "VERDE": 0},
}


# ---------- FUNCIONES AUXILIARES ----------

def get_stream_url(youtube_url: str) -> str:
    ydl_opts = {
        "format": "best",
        "noplaylist": True,
        "quiet": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info["url"]


def detectar_color_semaforo(roi, tipo="3"):
    if roi.size == 0:
        return "Semaforo"

    h, w, _ = roi.shape

    if tipo == "3":
        third = h // 3
        if third == 0:
            return "Semaforo"
        top    = roi[0:third]
        middle = roi[third:2 * third]
        bottom = roi[2 * third:h]
        segmentos = {"ROJO": top, "AMARILLO": middle, "VERDE": bottom}
    else:
        half = h // 2
        if half == 0:
            return "Semaforo"
        top    = roi[0:half]
        bottom = roi[half:h]
        segmentos = {"ROJO": top, "VERDE": bottom}

    brillo_segmentos = {}
    for nombre, seg in segmentos.items():
        if seg.size == 0:
            brillo_segmentos[nombre] = 0
            continue

        hsv = cv2.cvtColor(seg, cv2.COLOR_BGR2HSV)
        _, s_seg, v_seg = cv2.split(hsv)
        mask = (v_seg > 150) & (s_seg > 60)

        if np.count_nonzero(mask) == 0:
            brillo_segmentos[nombre] = 0
        else:
            brillo_segmentos[nombre] = float(np.mean(v_seg[mask]))

    nombre_mas_brillante = max(brillo_segmentos, key=brillo_segmentos.get)
    max_brillo = brillo_segmentos[nombre_mas_brillante]

    if max_brillo < 120:
        return "Semaforo"

    return nombre_mas_brillante


def mouse_callback(event, x, y, flags, param):
    global click_points
    if event == cv2.EVENT_LBUTTONDOWN:
        click_points.append((x, y))
        print(f"Click en: ({x}, {y})")

        if len(click_points) == 2:
            (x1, y1), (x2, y2) = click_points
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            print(f"ROI sugerido -> ({x1}, {y1}, {x2}, {y2})")
            print('  {"roi": (%d, %d, %d, %d), "tipo": "3", "id": "S_NUEVO"}\n' %
                  (x1, y1, x2, y2))
            click_points = []


def point_side(px, py, x1, y1, x2, y2):
    """Devuelve signo de en qué lado de la línea (x1,y1)-(x2,y2) está el punto."""
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


# ---------- PROGRAMA PRINCIPAL ----------

def main():
    global prev_time, last_side

    print("Obteniendo URL del stream de YouTube...")
    stream_url = get_stream_url(YOUTUBE_URL)

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("❌ No se pudo abrir el stream de YouTube.")
        return

    print("Cargando modelo YOLO 11…")
    model = YOLO("yolo11n.pt")  # o "yolov8s.pt" si no tenés yolo11

    # nombres en español
    original_names = model.model.names
    model.model.names = {
        cls_id: class_map.get(eng_name, eng_name)
        for cls_id, eng_name in original_names.items()
    }

    window_name = "YOLO11_Semaforos_YouTube"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    if MODO_CALIBRACION:
        print("🛠 MODO CALIBRACION ACTIVADO")
    else:
        print("✅ MODO PRODUCCION (contando cruces)")

    while True:
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time

        ret, frame = cap.read()
        if not ret:
            print("⚠️ No se pudo leer frame del stream, saliendo...")
            break

        # 1) Estimamos color actual de cada semáforo
        semaforos_color = {}
        colormap = {
            "ROJO": (0, 0, 255),
            "VERDE": (0, 255, 0),
            "AMARILLO": (0, 255, 255),
            "Semaforo": (255, 255, 255),
        }

        for zona in SEMAFORO_ZONES:
            x1, y1, x2, y2 = zona["roi"]
            tipo = zona.get("tipo", "3")
            sid = zona.get("id", "S")
            roi = frame[y1:y2, x1:x2]
            color = detectar_color_semaforo(roi, tipo=tipo)
            semaforos_color[sid] = color

        # 2) Detección + tracking de autos
        results = model.track(
            frame,
            verbose=False,
            conf=0.4,
            persist=True,
            tracker="bytetrack.yaml",
        )
        annotated = results[0].plot(line_width=1, font_size=10).copy()

        # 3) Dibujar rectángulos de semáforos y su color
        for zona in SEMAFORO_ZONES:
            x1, y1, x2, y2 = zona["roi"]
            sid = zona.get("id", "S")
            color = semaforos_color.get(sid, "Semaforo")
            cv2.rectangle(annotated, (x1, y1), (x2, y2), colormap[color], 2)
            cv2.putText(
                annotated,
                f"{sid}:{color}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                colormap[color],
                2,
            )

        # 4) Dibujar líneas de cruce
        for lane in LANES:
            x1, y1 = lane["p1"]
            x2, y2 = lane["p2"]
            cv2.line(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(
                annotated,
                lane["name"],
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

        # 5) Procesar cada vehículo detectado
        if results and hasattr(results[0], "boxes"):
            for box in results[0].boxes:
                if box.cls is None:
                    continue
                cls_id = int(box.cls[0])
                name = model.model.names[cls_id]

                if name != "auto":
                    continue

                if box.id is None:
                    continue
                track_id = int(box.id[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                cv2.circle(annotated, (cx, cy), 4, (0, 255, 255), -1)
                cv2.putText(
                    annotated,
                    str(track_id),
                    (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )

                for lane in LANES:
                    ln = lane["name"]
                    x1_l, y1_l = lane["p1"]
                    x2_l, y2_l = lane["p2"]
                    sense = lane.get("sense", 1)

                    side_raw = point_side(cx, cy, x1_l, y1_l, x2_l, y2_l)
                    side_now = sense * side_raw
                    key = (track_id, ln)

                    if key in last_side:
                        side_prev = last_side[key]
                        # condición de cruce ajustada por "sense"
                        if side_prev < 0 and side_now >= 0:
                            colores = [
                                semaforos_color.get(sid, "Semaforo")
                                for sid in lane["semaforos"]
                            ]
                            if "ROJO" in colores:
                                c = "ROJO"
                            elif "AMARILLO" in colores:
                                c = "AMARILLO"
                            elif "VERDE" in colores:
                                c = "VERDE"
                            else:
                                c = "Semaforo"

                            if c in ("ROJO", "AMARILLO", "VERDE"):
                                counts[ln][c] += 1
                                print(
                                    f"CRUCE {ln} en {c} - track_id={track_id}  total={counts[ln]}"
                                )

                    last_side[key] = side_now

        # 6) Mostrar contadores
        y_base = 90
        for lane_name, lane_counts in counts.items():
            text = (
                f"{lane_name}: R={lane_counts['ROJO']}"
                f"  A={lane_counts['AMARILLO']}"
                f"  V={lane_counts['VERDE']}"
            )
            cv2.putText(
                annotated,
                text,
                (10, y_base),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            y_base += 25

        if MODO_CALIBRACION:
            cv2.putText(
                annotated,
                "MODO CALIBRACION (clic para ROIs semaforos)",
                (10, y_base + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        cv2.imshow(window_name, annotated)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
