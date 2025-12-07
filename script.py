import cv2
import numpy as np

# Cargar imagen en escala de grises
img = cv2.imread('OCT_Dataset/NO/no_1425855_1.jpg', cv2.IMREAD_GRAYSCALE)

# Preprocesamiento ===
blur = cv2.GaussianBlur(img, (5, 5), 0)

# Filtros Canny: 1 para 2 capas de retina + fóvea y 1 para vitreo 
# Los valores de umbral pueden ajustarse según la calidad de la imagen OCT
edges = cv2.Canny(blur, 0, 400)
filtroVitreo = cv2.Canny(blur, 25, 80)

# Detección de bordes por columna
height, width = edges.shape
top_line = np.full(width, np.nan)
bottom_line = np.full(width, np.nan)

for x in range(width):
    column = edges[:, x]
    y_candidates = np.where(column > 0)[0]
    if len(y_candidates) > 0:
        top_line[x] = y_candidates[0]      # borde superior
        bottom_line[x] = y_candidates[-1]  # borde inferior

# Vitreo
height2, width2 = filtroVitreo.shape
vitreo_line = np.full(width2, np.nan)

for x in range(width2):
    column = filtroVitreo[:, x]
    y_candidates = np.where(column > 0)[0]
    if len(y_candidates) > 0:
        vitreo_line[x] = y_candidates[0]  # borde vitreo

# Suavizado de líneas
def smooth(signal, window=15):
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode='same')

top_smooth = smooth(top_line)
bottom_smooth = smooth(bottom_line)
vitreo_smooth = smooth(vitreo_line)

# Detección de fóvea
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

# Detección de agujero macular alrededor de la fóvea
def detectar_agujero_macular_fovea(top_line, bottom_line, vitreo_line,
                                   fovea_x, window_radius=40, thickness_ratio=0.3):
    start = max(fovea_x - window_radius, 0)
    end   = min(fovea_x + window_radius, len(top_line))

    # grosor en la ventana central
    thickness = bottom_line[start:end] - top_line[start:end]
    thickness = thickness[~np.isnan(thickness)]

    if len(thickness) == 0:
        return None, None

    min_th = np.min(thickness)
    mean_th = np.mean(thickness)

    # agujero si el mínimo es muy pequeño comparado con la retina periférica
    if min_th < thickness_ratio * mean_th:
        x_min = start + np.argmin(thickness)
        print(f"[ALERTA] Posible agujero macular en X={x_min}")
        return x_min, min_th

    print("NO hay agujero macular")
    return None, None

agujero_x, agujero_grosor = detectar_agujero_macular_fovea(
    top_smooth,        
    bottom_smooth,      
    vitreo_smooth,      
    fovea_x,            
    window_radius=40,    
    thickness_ratio=0.3  
)

# Visualización
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for x in range(width):
    if not np.isnan(top_smooth[x]):
        y = int(top_smooth[x]) + 1
        cv2.circle(img_color, (x, y), 1, (0, 0, 255), -1)    # rojo = capa superior
    if not np.isnan(bottom_smooth[x]):
        y = int(bottom_smooth[x])
        cv2.circle(img_color, (x, y), 1, (0, 255, 0), -1)    # verde = capa inferior
    if not np.isnan(vitreo_smooth[x]):
        y = int(vitreo_smooth[x]) - 1  # para poder diferenciar con la capa roja 
        cv2.circle(img_color, (x, y), 1, (255, 0, 0), -1)    # verde = capa inferior

# Dibujar fóvea
if fovea_x is not None and fovea_y is not None:
    cv2.circle(img_color, (fovea_x, fovea_y), 8, (255, 255, 0), 2)
    cv2.circle(img_color, (fovea_x, fovea_y), 3, (255, 255, 0), -1)
    cv2.putText(img_color, 'Fovea', (fovea_x - 25, fovea_y - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    print(f"Fóvea detectada en: X={fovea_x}, Y={fovea_y}")

# Mostrar resultados
cv2.imshow('Original', img)
cv2.imshow('Bordes de Canny para retina y fovea', edges)
cv2.imshow('Borde de Canny para vitreo', filtroVitreo)
cv2.imshow('Analisis OCT con Canny', img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
