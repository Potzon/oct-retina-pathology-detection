import cv2
import numpy as np

# ========= 1) Cargar y preprocesar =========
img = cv2.imread('OCT_Dataset/NO/no_1391081_2.jpg', cv2.IMREAD_GRAYSCALE)
assert img is not None, "No se pudo leer la imagen"  # Cargar en escala de grises es adecuado para Canny [web:7][web:19]

# Desenfoque Gaussiano previo a Canny para reducir ruido
blur = cv2.GaussianBlur(img, (5,5), 0)  # Pre-blur recomendado antes de cv2.Canny [web:13]

# Canny para realce de bordes global
# Ajusta thresholds según tu contraste; L2gradient=True usa norma L2 más precisa [web:2][web:5][web:11]
t1, t2 = 50, 150
edges_global = cv2.Canny(blur, t1, t2, apertureSize=3, L2gradient=True)  # Parámetros estándar de Canny [web:1][web:2]

# ========= 2) Energía de borde (para el límite inferior) =========
# En el código original se usaba Sobel vertical para bordes horizontales; aquí derivamos una "energía" compatible
# a partir de Canny global, suavizada y normalizada a [0,1].
edges_float = edges_global.astype(np.float32)
# Suavizado ligero para continuidad en la energía
edges_blur = cv2.GaussianBlur(edges_float, (5,5), 0)  # ayuda a continuidad en DP [web:13]
# Normalizar a [0,1]
edge = (edges_blur - edges_blur.min()) / (edges_blur.max() - edges_blur.min() + 1e-6)  # Normalización usual [web:8]

# Coste base: menor es mejor (inverso del borde)
base_cost = 1.0 - edge  # en [0,1], coherente con DP que minimiza costo [web:8]

H, W = base_cost.shape

# ========= 3) Programación dinámica con continuidad =========
# Parámetros
jmp = 10      # salto vertical máximo por columna (px)
lam1 = 0.3    # penalización por salto |y - y_prev|
lam2 = 0.2    # penalización de curvatura |y - 2*y_prev + y_prevprev|

# Acumuladores DP
C = np.full((H, W), np.inf, dtype=np.float32)
P = np.full((H, W), -1, dtype=np.int16)   # backpointer: y_prev

# Inicializa primera columna
C[:, 0] = base_cost[:, 0]

for x in range(1, W):
    for y in range(H):
        y_min = max(0, y - jmp)
        y_max = min(H, y + jmp + 1)
        prev_range = np.arange(y_min, y_max, dtype=np.int32)

        # coste transición
        jump_pen = lam1 * np.abs(prev_range - y).astype(np.float32)
        cand = C[prev_range, x-1] + base_cost[y, x] + jump_pen

        # curvatura cuando hay x-2
        if x >= 2:
            ypp = P[prev_range, x-1]  # y_prevprev de cada candidato
            valid = ypp >= 0
            curv = np.zeros_like(cand, dtype=np.float32)
            curv[valid] = lam2 * np.abs(y - 2*prev_range[valid] + ypp[valid]).astype(np.float32)
            cand += curv

        k = int(np.argmin(cand))
        C[y, x] = cand[k]
        P[y, x] = prev_range[k]

# Backtracking del mejor camino
y_end = int(np.argmin(C[:, -1]))
bottom_path = np.full(W, np.nan, dtype=np.float32)
y = y_end
for x in range(W-1, -1, -1):
    bottom_path[x] = y
    if x > 0:
        y = int(P[y, x])

# Suavizado ligero (Savitzky-Golay si está disponible; si no, media móvil)
try:
    from scipy.signal import savgol_filter
    bottom_smooth = savgol_filter(bottom_path, 31 if W >= 31 else (W//2*2+1), 3)
except Exception:
    k = 15
    kernel = np.ones(k, dtype=np.float32)/k
    bottom_smooth = np.convolve(bottom_path, kernel, mode='same')

# ========= 4) Detección de línea superior sencilla (por Canny por columna) =========
# Reutilizamos edges_global ya calculado por Canny, coherente con su salida binaria 0/255 [web:5]
edges = edges_global
top_line = np.full(W, np.nan)
for x in range(W):
    col = np.where(edges[:, x] > 0)[0]
    if len(col) > 0:
        top_line[x] = col[0]

# Suavizado superior
if np.isfinite(top_line).any():
    # Interpola huecos NaN
    xs = np.arange(W)
    mask = np.isfinite(top_line)
    top_line[~mask] = np.interp(xs[~mask], xs[mask], top_line[mask])
    try:
        from scipy.signal import savgol_filter
        top_smooth = savgol_filter(top_line, 31 if W >= 31 else (W//2*2+1), 3)
    except Exception:
        k = 15
        kernel = np.ones(k, dtype=np.float32)/k
        top_smooth = np.convolve(top_line, kernel, mode='same')
else:
    top_smooth = np.full(W, np.nan)

# ========= 5) Detección de fóvea en la línea superior =========
def detect_fovea_on_red_line(red_y, center_frac=(0.35, 0.65)):
    a, b = center_frac
    cs, ce = int(W*a), int(W*b)
    region = red_y[cs:ce]
    if len(region) > 0 and not np.all(np.isnan(region)):
        max_idx = int(np.nanargmax(region))
        fx = cs + max_idx
        fy = int(red_y[fx])
        return fx, fy
    return None, None

fovea_x, fovea_y = detect_fovea_on_red_line(top_smooth)

# ========= 6) Visualización =========
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Dibuja superior (rojo)
for x in range(W):
    y = int(top_smooth[x]) if not np.isnan(top_smooth[x]) else None
    if y is not None:
        cv2.circle(img_color, (x, y), 1, (0, 0, 255), -1)

# Dibuja inferior continuo (verde)
for x in range(W):
    y = int(bottom_smooth[x]) if not np.isnan(bottom_smooth[x]) else None
    if y is not None:
        cv2.circle(img_color, (x, y), 1, (0, 255, 0), -1)

# Fóvea
if fovea_x is not None and fovea_y is not None:
    cv2.circle(img_color, (fovea_x, fovea_y), 8, (255, 255, 0), 2)
    cv2.circle(img_color, (fovea_x, fovea_y), 3, (255, 255, 0), -1)
    cv2.putText(img_color, 'Fovea', (fovea_x - 25, fovea_y - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    print(f"Fovea detectada en: X={fovea_x}, Y={fovea_y}")

# Mostrar
cv2.imshow('Canny (global)', edges_global)
cv2.imshow('Segmentacion OCT (continuidad en inferior)', img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
