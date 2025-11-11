import cv2
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def analyze_oct_macular_scan(image_path, output_folder='./results'):
    """
    Análisis completo de escaneo OCT macular:
    - Detección de fóvea (incluso con tracción o distorsión)
    - Detección de desprendimiento vítreo posterior (DVP)
    - Visualización integrada de ambos hallazgos
    """
    
    # Crear carpeta de resultados
    import os
    os.makedirs(output_folder, exist_ok=True)
    
    # Cargar imagen
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    print("=" * 60)
    print("ANÁLISIS DE OCT MACULAR")
    print("=" * 60)
    
    # ==================== PREPROCESAMIENTO ====================
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    edges = cv2.Canny(enhanced, 20, 60)
    
    # ==================== DETECCIÓN DE BORDES RETINIANOS ====================
    print("\n[1/4] Detectando bordes retinianos...")
    
    # Borde superior (ILM)
    top_boundary = []
    for x in range(width):
        column = gray[:, x]
        for y in range(height):
            if column[y] > 25:
                top_boundary.append((x, y))
                break
    
    # Borde inferior (RPE)
    bottom_boundary = []
    for x in range(width):
        column = gray[:, x]
        for y in range(height - 1, -1, -1):
            if column[y] > 40:
                bottom_boundary.append((x, y))
                break
    
    if len(top_boundary) == 0 or len(bottom_boundary) == 0:
        print("❌ Error: No se detectaron bordes retinianos claros")
        return None, None, None, None
    
    top_boundary = np.array(top_boundary)
    bottom_boundary = np.array(bottom_boundary)
    print(f"✓ Bordes detectados: ILM y RPE identificados")
    
    # ==================== DETECCIÓN DE FÓVEA (MÉTODO MULTIMODAL) ====================
    print("\n[2/4] Detectando fóvea (análisis multimodal)...")
    
    # MÉTODO 1: Centro geométrico horizontal
    horizontal_center = width // 2
    
    # MÉTODO 2: Grosor retiniano mínimo
    thickness_profile = []
    for i in range(min(len(top_boundary), len(bottom_boundary))):
        thickness = bottom_boundary[i][1] - top_boundary[i][1]
        thickness_profile.append(thickness)
    
    if len(thickness_profile) > 50:
        window = 30
        smoothed_thickness = signal.savgol_filter(thickness_profile, window, 3)
        
        center_start = len(smoothed_thickness) // 3
        center_end = 2 * len(smoothed_thickness) // 3
        center_region = smoothed_thickness[center_start:center_end]
        
        min_idx = np.argmin(center_region)
        fovea_x_thickness = center_start + min_idx
    else:
        fovea_x_thickness = horizontal_center
    
    # MÉTODO 3: Punto de máxima curvatura (tracción)
    y_coords = top_boundary[:, 1]
    if len(y_coords) > 50:
        smoothed_top = signal.savgol_filter(y_coords, 25, 3)
        second_deriv = np.gradient(np.gradient(smoothed_top))
        
        center_start = len(smoothed_top) // 3
        center_end = 2 * len(smoothed_top) // 3
        center_deriv = second_deriv[center_start:center_end]
        
        if np.max(np.abs(center_deriv)) > 0.01:
            traction_idx = center_start + np.argmax(np.abs(center_deriv))
            fovea_x_traction = traction_idx
        else:
            fovea_x_traction = horizontal_center
    else:
        fovea_x_traction = horizontal_center
    
    # MÉTODO 4: Zona elipsoide (intensidad máxima)
    intensity_profile = []
    for x in range(width):
        if x < len(top_boundary) and x < len(bottom_boundary):
            top_y = top_boundary[x][1]
            bottom_y = bottom_boundary[x][1]
            mid_y = int(top_y + 0.7 * (bottom_y - top_y))
            
            roi_intensity = np.mean(gray[max(0, mid_y-5):min(height, mid_y+10), 
                                          max(0, x-5):min(width, x+5)])
            intensity_profile.append(roi_intensity)
    
    if len(intensity_profile) > 50:
        smoothed_intensity = signal.savgol_filter(intensity_profile, 31, 3)
        center_start = len(smoothed_intensity) // 3
        center_end = 2 * len(smoothed_intensity) // 3
        center_intensity = smoothed_intensity[center_start:center_end]
        
        max_idx = np.argmax(center_intensity)
        fovea_x_intensity = center_start + max_idx
    else:
        fovea_x_intensity = horizontal_center
    
    # CONSENSO: Votación ponderada
    candidates = [
        (horizontal_center, 1.5),
        (fovea_x_thickness, 2.0),
        (fovea_x_traction, 1.0),
        (fovea_x_intensity, 1.5)
    ]
    
    total_weight = sum(w for _, w in candidates)
    fovea_x = int(sum(x * w for x, w in candidates) / total_weight)
    
    # Calcular fovea_y
    if fovea_x < len(top_boundary) and fovea_x < len(bottom_boundary):
        fovea_y_top = top_boundary[fovea_x][1]
        fovea_y_bottom = bottom_boundary[fovea_x][1]
        fovea_y = int((fovea_y_top + fovea_y_bottom) / 2)
    else:
        fovea_y = height // 2
    
    # Detectar tracción vitreomacular
    has_traction = False
    if len(y_coords) > 0:
        smoothed = signal.savgol_filter(y_coords, 25, 3)
        foveal_region = smoothed[max(0, fovea_x-50):min(len(smoothed), fovea_x+50)]
        peripheral_mean = np.mean(np.concatenate([smoothed[:100], smoothed[-100:]]))
        foveal_mean = np.mean(foveal_region)
        
        if peripheral_mean - foveal_mean > 10:
            has_traction = True
    
    print(f"✓ Fóvea detectada en: X={fovea_x}, Y={fovea_y}")
    if has_traction:
        print(f"⚠ TRACCIÓN VITREOMACULAR DETECTADA")
    
    # ==================== DETECCIÓN DE VÍTREO DESPRENDIDO ====================
    print("\n[3/4] Detectando desprendimiento vítreo posterior (DVP)...")
    
    # Buscar línea del vítreo en la región superior
    vitreous_line = []
    vitreous_region = enhanced[0:height//2, :]
    
    thresh = cv2.adaptiveThreshold(vitreous_region, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Detectar bordes en región del vítreo
    for x in range(width):
        column = edges[:, x]
        for y in range(0, height//2):
            if column[y] > 0 and enhanced[y, x] > 80:
                vitreous_line.append((x, y))
                break
    
    # Análisis de DVP
    has_pvd = False
    pvd_distance = 0
    avg_vitreous_y = 0
    
    if len(vitreous_line) > width * 0.3:
        vitreous_line = np.array(vitreous_line)
        
        if len(vitreous_line) > 20:
            y_vitreous = vitreous_line[:, 1]
            window = 15
            smoothed_vitreous = signal.savgol_filter(y_vitreous, window, 3)
            
            avg_vitreous_y = np.mean(smoothed_vitreous)
            pvd_distance = fovea_y - avg_vitreous_y
            
            if pvd_distance > 50:
                has_pvd = True
                print(f"✓ DVP DETECTADO")
                print(f"  - Posición vítreo: Y={avg_vitreous_y:.1f}")
                print(f"  - Posición retina: Y={fovea_y}")
                print(f"  - Distancia separación: {pvd_distance:.1f} píxeles")
            else:
                print(f"✓ Vítreo adherido o separación mínima ({pvd_distance:.1f} px)")
    else:
        print(f"✓ No se detectó línea vítrea clara - Vítreo adherido")
    
    # ==================== VISUALIZACIÓN INTEGRADA ====================
    print("\n[4/4] Generando visualización...")
    
    result = img.copy()
    
    # ---- FÓVEA ----
    square_size = 100
    x1 = max(0, fovea_x - square_size // 2)
    y1 = max(0, fovea_y - square_size // 2)
    x2 = min(width, fovea_x + square_size // 2)
    y2 = min(height, fovea_y + square_size // 2)
    
    # Color verde normal, rojo si hay tracción
    fovea_color = (0, 0, 255) if has_traction else (0, 255, 0)
    cv2.rectangle(result, (x1, y1), (x2, y2), fovea_color, 2)
    cv2.circle(result, (fovea_x, fovea_y), 5, fovea_color, -1)
    
    # Cruz centrada
    cross_size = 15
    cv2.line(result, (fovea_x - cross_size, fovea_y), 
             (fovea_x + cross_size, fovea_y), fovea_color, 2)
    cv2.line(result, (fovea_x, fovea_y - cross_size), 
             (fovea_x, fovea_y + cross_size), fovea_color, 2)
    
    # Etiqueta fóvea
    label_text = "Fovea (con tracción)" if has_traction else "Fovea"
    cv2.putText(result, label_text, (fovea_x + 20, fovea_y - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, fovea_color, 2)
    
    # ---- VÍTREO SEPARADO ----
    y_offset = height - 60
    y_preoffset = y_offset - 30

    if has_pvd:
        vitreous_line = np.array(vitreous_line)
        
        # Dibujar línea del vítreo
        for i in range(len(vitreous_line)):
            x, y = vitreous_line[i]
            cv2.circle(result, (x, y), 2, (255, 0, 0), -1)
        
        # Línea suavizada
        if len(vitreous_line) > 20:
            y_vit = vitreous_line[:, 1]
            smoothed_vit = signal.savgol_filter(y_vit, 15, 3)
            for i in range(len(smoothed_vit) - 1):
                cv2.line(result, (vitreous_line[i, 0], int(smoothed_vit[i])),
                        (vitreous_line[i+1, 0], int(smoothed_vit[i+1])),
                        (255, 0, 0), 2)
        
        # Etiqueta DVP
        cv2.putText(result, 'DVP (Vitreo Separado)', (20, y_preoffset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Flecha de distancia
        arrow_start = (width // 2, int(avg_vitreous_y))
        arrow_end = (width // 2, fovea_y - 20)
        cv2.arrowedLine(result, arrow_start, arrow_end, (0, 255, 255), 2)
        
        # Texto distancia
        distance_text = f"Separacion: {pvd_distance:.0f}px"
        cv2.putText(result, distance_text, (width // 2 + 10, 
                    (int(avg_vitreous_y) + fovea_y) // 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    
    # ---- INFORMACIÓN DE DIAGNÓSTICO ----
    
    cv2.putText(result, "DIAGNOSTICO:", (20, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    y_offset += 30
    if has_traction:
        cv2.putText(result, "- Traccion Vitreomacular PRESENTE", (30, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        y_offset += 25
    
    if has_pvd:
        cv2.putText(result, "- Desprendimiento Vitreo PRESENTE", (30, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
        y_offset += 25
    
    if not has_traction and not has_pvd:
        cv2.putText(result, "- Hallazgos: Normales/Sin patologia mayor", (30, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 1)
    
    # ==================== GRÁFICOS DE ANÁLISIS ====================
    
    # Gráfico de grosor retiniano
    if len(thickness_profile) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Perfil de grosor
        ax = axes[0, 0]
        norm_thickness = np.array(smoothed_thickness) - np.min(smoothed_thickness)
        norm_thickness = norm_thickness / np.max(norm_thickness) * 100
        ax.plot(norm_thickness, 'b-', linewidth=2)
        ax.axvline(fovea_x, color='g', linestyle='--', linewidth=2, label='Fóvea detectada')
        if has_pvd:
            ax.axhline(10, color='r', linestyle='--', alpha=0.5, label='DVP')
        ax.set_xlabel('Posición horizontal (píxeles)')
        ax.set_ylabel('Grosor retiniano normalizado')
        ax.set_title('Perfil de Grosor Retiniano')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Intensidad elipsoide
        ax = axes[0, 1]
        smoothed_int = signal.savgol_filter(intensity_profile, 31, 3)
        ax.plot(smoothed_int, 'purple', linewidth=2)
        ax.axvline(fovea_x, color='g', linestyle='--', linewidth=2, label='Fóvea detectada')
        ax.set_xlabel('Posición horizontal (píxeles)')
        ax.set_ylabel('Intensidad (u.a.)')
        ax.set_title('Perfil de Intensidad - Zona Elipsoide')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Posición de ILM (borde superior)
        ax = axes[1, 0]
        ax.plot(y_coords, 'b-', linewidth=1, alpha=0.7, label='ILM (crudo)')
        ax.plot(smoothed_top, 'g-', linewidth=2, label='ILM (suavizado)')
        ax.axvline(fovea_x, color='g', linestyle='--', alpha=0.5)
        if has_pvd:
            ax.axhline(avg_vitreous_y, color='r', linestyle='--', linewidth=2, label='Vítreo separado')
        ax.set_xlabel('Posición horizontal (píxeles)')
        ax.set_ylabel('Posición Y (píxeles)')
        ax.set_title('Contorno de ILM (Superficie Retiniana)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
        
        # 4. Segunda derivada (detección tracción)
        ax = axes[1, 1]
        second_deriv = np.gradient(np.gradient(smoothed_top))
        ax.plot(second_deriv, 'orange', linewidth=2)
        ax.axvline(fovea_x, color='g', linestyle='--', linewidth=2, label='Fóvea detectada')
        ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
        if has_traction:
            ax.fill_between(range(len(second_deriv)), second_deriv, alpha=0.3, color='red', 
                           label='Tracción detectada')
        ax.set_xlabel('Posición horizontal (píxeles)')
        ax.set_ylabel('Segunda derivada')
        ax.set_title('Detección de Tracción (Curvatura)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_folder}/analisis_detallado.png', dpi=150, bbox_inches='tight')
        print(f"✓ Gráficos guardados en {output_folder}/analisis_detallado.png")
    
    # Guardar imagen con anotaciones
    cv2.imwrite(f'{output_folder}/resultado_final.jpg', result)
    print(f"✓ Imagen anotada guardada en {output_folder}/resultado_final.jpg")
    
    # ==================== REPORTE FINAL ====================
    print("\n" + "=" * 60)
    print("REPORTE FINAL")
    print("=" * 60)
    print(f"\nFÓVEA:")
    print(f"  Posición: ({fovea_x}, {fovea_y})")
    print(f"  Estado: {'Distorsionada por tracción' if has_traction else 'Normal'}")
    
    print(f"\nVÍTREO:")
    print(f"  Estado: {'Desprendido (DVP)' if has_pvd else 'Adherido'}")
    if has_pvd:
        print(f"  Distancia de separación: {pvd_distance:.1f} píxeles")
    
    print(f"\nDIAGNÓSTICO:")
    diagnostico = []
    if has_traction:
        diagnostico.append("- Tracción Vitreomacular Syndrome (VMT)")
    if has_pvd:
        diagnostico.append("- Desprendimiento Vítreo Posterior (DVP)")
    if not diagnostico:
        diagnostico.append("- Sin hallazgos patológicos significativos")
    
    for d in diagnostico:
        print(d)
    
    print("\n" + "=" * 60)
    
    # Mostrar imagen
    cv2.imshow('Análisis OCT Macular Completo', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return result, (fovea_x, fovea_y), has_traction, has_pvd

# ==================== EJECUCIÓN ====================
if __name__ == "__main__":
    # Cambiar por la ruta de tu imagen
    resultado, fovea_pos, traction, pvd = analyze_oct_macular_scan('OCT_Dataset/NO/no_1250592_2.jpg')
