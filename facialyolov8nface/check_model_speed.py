import onnxruntime as ort
import numpy as np
import time
import sys
import os

model_path = sys.argv[1] if len(sys.argv) > 1 else "det_500m.onnx"
if not os.path.exists(model_path):
    print(f"Modelo no encontrado: {model_path}")
    sys.exit(1)

print(f"Cargando: {model_path}")
sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"]) 
inputs = sess.get_inputs()
outputs = sess.get_outputs()
print("Inputs:")
for i in inputs:
    print(f" - {i.name}: shape={i.shape}, type={i.type}")
print("Outputs:")
for o in outputs:
    print(f" - {o.name}: shape={o.shape}, type={o.type}")

# Build a dummy input replacing dynamic dims
def make_input_from_shape(shape):
    shape = list(shape)
    for idx, s in enumerate(shape):
        if s is None or (isinstance(s, str) and s.lower() == 'none'):
            # batch dim -> 1, channel dim -> 3, spatial dims -> 640
            if idx == 0:
                shape[idx] = 1
            elif idx == 1:
                shape[idx] = 3
            else:
                shape[idx] = 640
    return np.random.rand(*shape).astype(np.float32)

input_feed = {}
for i in inputs:
    try:
        shp = tuple([dim if isinstance(dim, int) else None for dim in i.shape])
        arr = make_input_from_shape(shp)
    except Exception:
        arr = np.random.rand(1,3,640,640).astype(np.float32)
    # If model expects NHWC or other, the random data still tests runtime cost
    input_feed[i.name] = arr

# Warmup
print("Realizando warmup...")
for _ in range(2):
    sess.run(None, input_feed)

# Timed runs
runs = 5
times = []
print(f"Ejecutando {runs} inferencias...")
for _ in range(runs):
    t0 = time.time()
    sess.run(None, input_feed)
    t1 = time.time()
    times.append(t1 - t0)

print(f"Tiempos: {times}")
print(f"Promedio: {sum(times)/len(times):.3f}s per inference")
print("Hecho.")
