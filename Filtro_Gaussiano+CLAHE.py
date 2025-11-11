import cv2
import numpy as np

# === 1. Cargar imagen en escala de grises ===
img = cv2.imread('OCT_Dataset/NO/no_1391081_3.jpg', cv2.IMREAD_GRAYSCALE)

# === 2. Preprocesamiento ===
blur = cv2.GaussianBlur(img, (5, 5), 0)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(blur)

# === 3. Gradiente morfológico ===
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
morph_gradient = cv2.morphologyEx(enhanced, cv2.MORPH_GRADIENT, kernel)

# === 4. Detección de bordes por columna ===
height, width = morph_gradient.shape
top_line = np.full(width, np.nan)
bottom_line = np.full(width, np.nan)
vitreous_line = np.full(width, np.nan)  # <-- Nueva capa: vítreo

for x in range(width):
    column = morph_gradient[:, x]

    # Umbral dinámico (por percentil alto)
    threshold = np.percentile(column, 98)
    y_candidates = np.where(column > threshold)[0]

    if len(y_candidates) > 0:
        top_line[x] = y_candidates[0]       # capa superior (retina)
        bottom_line[x] = y_candidates[-1]   # capa inferior

        # --- NUEVO: Buscar capa del vítreo ---
        # Si hay un borde fuerte por encima de la capa superior, considerarlo vítreo
        pre_region = column[:int(top_line[x]) - 5] if top_line[x] > 5 else []
        if len(pre_region) > 0:
            local_thresh = np.percentile(pre_region, 95)
            vit_candidates = np.where(pre_region > local_thresh)[0]
            if len(vit_candidates) > 0:
                vitreous_line[x] = vit_candidates[-1]  # borde más cercano al top_line

# === 5. Suavizado de líneas ===
def smooth(signal, window=15):
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode='same')

top_smooth = smooth(top_line)
bottom_smooth = smooth(bottom_line)
vitreous_smooth = smooth(vitreous_line)


# === 6. Detección de fóvea ===
def detect_fovea_on_red_line(top_line, window_width=50):
    center_start = int(len(top_line) * 0.35)
    center_end = int(len(top_line) * 0.65)
    center_region = top_line[center_start:center_end]
    
    if len(center_region) > 0 and not np.all(np.isnan(center_region)):
        max_idx = np.nanargmax(center_region)
        fovea_x = center_start + max_idx
        fovea_y = int(top_line[fovea_x])
        return fovea_x, fovea_y
    return None, None

fovea_x, fovea_y = detect_fovea_on_red_line(top_smooth)

# === 7. Visualización ===
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for x in range(width):
    if not np.isnan(vitreous_smooth[x]):
        y = int(vitreous_smooth[x])
        cv2.circle(img_color, (x, y), 1, (255, 0, 255), -1)  # magenta = vítreo
    if not np.isnan(top_smooth[x]):
        y = int(top_smooth[x])
        cv2.circle(img_color, (x, y), 1, (0, 0, 255), -1)    # rojo = capa superior
    if not np.isnan(bottom_smooth[x]):
        y = int(bottom_smooth[x])
        cv2.circle(img_color, (x, y), 1, (0, 255, 0), -1)    # verde = capa inferior

# Dibujar fóvea
if fovea_x is not None and fovea_y is not None:
    cv2.circle(img_color, (fovea_x, fovea_y), 8, (255, 255, 0), 2)
    cv2.circle(img_color, (fovea_x, fovea_y), 3, (255, 255, 0), -1)
    cv2.putText(img_color, 'Fovea', (fovea_x - 25, fovea_y - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    print(f"Fovea detectada en: X={fovea_x}, Y={fovea_y}")

# Mostrar imagen
cv2.imshow('Analisis OCT Macular con Capa del Vitreo', img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
