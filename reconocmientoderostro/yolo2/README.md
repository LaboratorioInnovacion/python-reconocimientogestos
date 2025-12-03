# Censo YOLO + DeepFace + SORT

Sistema de conteo y análisis de personas en tiempo real usando:
- **YOLO v8** (detección de personas)
- **DeepFace** (análisis de edad y género)
- **SORT** (seguimiento de identidades)

## 📋 Requisitos del Sistema

- **Python:** 3.10
- **RAM:** 8 GB mínimo
- **GPU:** Opcional (mejora rendimiento)
- **Windows PowerShell** o terminal compatible

## 🚀 Instalación Rápida

### 1. Crear ambiente virtual con Python 3.10

```powershell
py -3.10 -m venv .venv310
```

### 2. Activar ambiente virtual

```powershell
. .\.venv310\Scripts\Activate.ps1
```

Si ves error de permisos:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

### 3. Instalar dependencias

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

**Tiempo estimado:** 5-10 minutos (depende de conexión y PC)

## ▶️ Ejecutar el Proyecto

### Comando básico
```powershell
python census_yolo_deepface_sort.py
```

### Controles
- **ESC** → Salir del programa
- La cámara debe estar conectada y habilitada

## 📊 Salida del Programa

Mientras se ejecuta, verás en pantalla:
- **Caja verde** alrededor de cada persona detectada
- **ID único** para cada persona rastreada
- **Contador en vivo** de hombres y mujeres
- **Distribución por edad** (0-12, 13-20, 21-35, 36-50, 51+)
- **Logs en consola** con detalles de cada detección

Ejemplo de log:
```
[10:30:45] ID=1 | Man | 28 años | 21-35
[10:30:47] ID=2 | Woman | 35 años | 36-50
```

## 📁 Estructura de Archivos

```
yolo2/
├── census_yolo_deepface_sort.py  (script principal)
├── sort.py                        (algoritmo SORT)
├── yolov8n.pt                     (modelo YOLO preentrenado)
├── requirements.txt               (dependencias)
└── README.md                      (este archivo)
```

## ⚙️ Configuración Personalizada

Puedes editar estos parámetros en `census_yolo_deepface_sort.py`:

```python
# Línea 11: Fuente de video (0 = cámara web, o ruta de video/IP)
cap = cv2.VideoCapture(0)

# Línea 12: Parámetros del tracker SORT
tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)
# max_age: frames que espera antes de descartar un ID
# min_hits: detecciones mínimas antes de contar una persona
# iou_threshold: umbral de coincidencia espacial
```

## 🔧 Solucionar Problemas

### Error: "No module named 'deepface'"
```powershell
python -m pip install --upgrade deepface
```

### Error: "No module named 'tf_keras'"
```powershell
python -m pip install tf-keras
```

### Error: "Numpy is not available"
```powershell
python -m pip install numpy==1.26.4 --force-reinstall
```

### La cámara no se abre
- Verifica que la cámara esté conectada y habilitada en Windows
- Prueba cambiar `VIDEO_SOURCE = 0` a `VIDEO_SOURCE = 1`
- O usa un archivo de video: `VIDEO_SOURCE = "video.mp4"`

### Rendimiento lento
- Reduce resolución de entrada (línea después de `ret, frame = cap.read()`)
- Aumenta intervalo de análisis (cambia `ANALYZE_EVERY_N_FRAMES`)
- Desactiva análisis de edad/gender si solo necesitas conteo

## 💡 Mejoras Futuras

- Exportar datos a CSV o base de datos
- Alertas por eventos (p.ej., conglomeración)
- Dashboard web en tiempo real
- Soporte para múltiples cámaras
- Análisis de emociones y otros atributos

## 📝 Notas Importantes

- **Primera ejecución:** Los modelos se descargarán automáticamente (requiere conexión)
- **Precisión de género:** Varía según iluminación, ángulo y calidad de cámara
- **Rendimiento:** Optimizado para Intel i5 + 8GB RAM (tu hardware)
- **Evitar duplicados:** El sistema usa IDs únicos para no contar 2 veces la misma persona

## 🐛 Reportar Errores

Si encuentras problemas:
1. Copia el mensaje de error completo
2. Revisa la sección "Solucionar Problemas"
3. Intenta reinstalar dependencias desde `requirements.txt`

## 📚 Referencias

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [DeepFace](https://github.com/serengp/deepface)
- [SORT Tracker](https://github.com/abewley/sort)

---

**Versión:** 1.0  
**Última actualización:** 3 de diciembre de 2025
