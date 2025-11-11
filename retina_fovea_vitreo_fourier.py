import cv2
import numpy as np

# ---------------- Parámetros ----------------
INPUT = 'OCT_Dataset/NO/no_9663705_2.jpg'

# Parámetros del filtro en frecuencia TOMADOS del segundo script
FOURIER_MODE = 'lowpass'   # 'lowpass' reduce speckle; 'highboost' realza bordes
SIGMA_FREQ = 20
HIGHBOOST_ALPHA = 1.5

# ---------- Utilidad: filtro de Fourier (del segundo script) ----------
def fourier_filter(gray, mode='lowpass', sigma=20, alpha=1.5):
    f32 = gray.astype(np.float32)
    dft = cv2.dft(f32, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft, axes=(0, 1))

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    x = np.linspace(-cx, cx - 1, w, dtype=np.float32)
    y = np.linspace(-cy, cy - 1, h, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    gauss = np.exp(-(X**2 + Y**2) / (2.0 * sigma**2)).astype(np.float32)

    if mode == 'lowpass':
        mask = np.repeat(gauss[:, :, None], 2, axis=2)
        filt = dft_shift * mask
    elif mode == 'highboost':
        low = np.repeat(gauss[:, :, None], 2, axis=2)
        high = 1.0 - low
        filt = dft_shift * (low + alpha * high)
    else:
        raise ValueError("mode debe ser 'lowpass' o 'highboost'")

    ishift = np.fft.ifftshift(filt, axes=(0, 1))
    idft = cv2.idft(ishift)
    mag = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])
    out = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return out

# === 1. Cargar imagen en escala de grises ===
img = cv2.imread(INPUT, cv2.IMREAD_GRAYSCALE)
assert img is not None, f'No se pudo leer {INPUT}'

# === 2. Preprocesamiento (del primer script) ===
blur = cv2.GaussianBlur(img, (5, 5), 0)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(blur)

# === 2b. SOLO para detectar la capa de abajo: crear rama con Fourier (del segundo script) ===
fourier_img = fourier_filter(enhanced, mode=FOURIER_MODE, sigma=SIGMA_FREQ, alpha=HIGHBOOST_ALPHA)

# === 3. Gradiente morfológico (primer script) ===
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
morph_gradient = cv2.morphologyEx(enhanced, cv2.MORPH_GRADIENT, kernel)

# === 3b. Gradiente morfológico en la rama Fourier para capa inferior (segundo script) ===
morph_gradient_bottom = cv2.morphologyEx(fourier_img, cv2.MORPH_GRADIENT, kernel)

# === 4. Detección por columnas ===
h, w = morph_gradient.shape
top_line = np.full(w, np.nan, dtype=np.float32)
vitreous_line = np.full(w, np.nan, dtype=np.float32)

# These are computed from the Fourier branch exclusively
bottom_line = np.full(w, np.nan, dtype=np.float32)

for x in range(w):
    # Capa superior y vítreo: usar el gradiente del primer pipeline
    col = morph_gradient[:, x]
    thr = np.percentile(col, 98)
    y_idx = np.where(col > thr)[0]

    if len(y_idx) > 0:
        top_line[x] = y_idx[0]

        # vítreo por encima del top_line
        if top_line[x] > 5:
            pre = col[:int(top_line[x]) - 5]
            if pre.size > 0:
                lthr = np.percentile(pre, 95)
                vit = np.where(pre > lthr)[0]
                if vit.size > 0:
                    vitreous_line[x] = vit[-1]

    # Capa inferior: usar el gradiente de la rama Fourier (segundo script)
    col_b = morph_gradient_bottom[:, x]
    thr_b = np.percentile(col_b, 98)
    y_idx_b = np.where(col_b > thr_b)[0]
    if len(y_idx_b) > 0:
        bottom_line[x] = y_idx_b[-1]

# === 5. Suavizado de líneas (como el segundo script, robusto a NaNs) ===
def smooth(signal, window=15):
    s = signal.copy().astype(np.float32)
    n = np.isnan(s)
    if np.any(~n):
        idx = np.arange(len(s))
        s[n] = np.interp(idx[n], idx[~n], s[~n])
    ker = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(s, ker, mode='same')

top_smooth = smooth(top_line, 15)
bottom_smooth = smooth(bottom_line, 15)
vitreous_smooth = smooth(vitreous_line, 21)

# === 6. Fóvea (primer script) ===
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

# === 7. Visualización y guardado ===
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for x in range(w):
    if not np.isnan(vitreous_smooth[x]):
        y = int(vitreous_smooth[x])
        cv2.circle(img_color, (x, y), 1, (255, 0, 255), -1)  # vítreo: magenta
    if not np.isnan(top_smooth[x]):
        y = int(top_smooth[x])
        cv2.circle(img_color, (x, y), 1, (0, 0, 255), -1)    # capa superior: rojo
    if not np.isnan(bottom_smooth[x]):
        y = int(bottom_smooth[x])
        cv2.circle(img_color, (x, y), 1, (0, 255, 0), -1)    # capa inferior: verde

if fovea_x is not None and fovea_y is not None:
    cv2.circle(img_color, (fovea_x, fovea_y), 8, (255, 255, 0), 2)
    cv2.circle(img_color, (fovea_x, fovea_y), 3, (255, 255, 0), -1)
    cv2.putText(img_color, 'Fovea', (fovea_x - 25, fovea_y - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

# Mostrar
cv2.imshow('Analisis OCT: superior+vitreo (primer) + inferior (Fourier)', img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
