"""
Script de Python para preprocesar imágenes OCT de retina
Realza el contraste y la nitidez para distinguir mejor las capas de la retina

Autor: Script generado para procesamiento de imágenes OCT
Fecha: Noviembre 2024
"""

from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import os

def ecualizar_histograma(img_array):
    """
    Aplica ecualización de histograma para mejorar el contraste
    
    Args:
        img_array: Array numpy de la imagen en escala de grises
    
    Returns:
        Array numpy con la imagen ecualizada
    """
    hist, bins = np.histogram(img_array.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    img_equalized = cdf[img_array]
    return img_equalized

def aplicar_pseudocolor(img_array):
    """
    Aplica un mapa de colores personalizado similar a las imágenes OCT estándar
    
    Args:
        img_array: Array numpy de la imagen en escala de grises
    
    Returns:
        Array numpy RGB con pseudocolor aplicado
    """
    img_rgb = np.zeros((img_array.shape[0], img_array.shape[1], 3), dtype=np.uint8)
    normalized = img_array.astype(float) / 255.0
    
    # Esquema de colores: verde-amarillo-azul (similar a OCT estándar)
    img_rgb[:, :, 0] = (normalized * 100).astype(np.uint8)   # R - componente rojo bajo
    img_rgb[:, :, 1] = (normalized * 255).astype(np.uint8)   # G - componente verde alto
    img_rgb[:, :, 2] = (normalized * 200).astype(np.uint8)   # B - componente azul medio
    
    return img_rgb

def procesar_imagen_oct(ruta_entrada, directorio_salida='/procesados'):
    """
    Procesa una imagen OCT de retina aplicando múltiples técnicas de mejora
    
    Args:
        ruta_entrada: Ruta de la imagen OCT a procesar
        directorio_salida: Directorio donde se guardarán las imágenes procesadas
    
    Returns:
        None (guarda las imágenes en el directorio especificado)
    """
    
    # Cargar la imagen en escala de grises
    print(f"Cargando imagen: {ruta_entrada}")
    img = Image.open(ruta_entrada).convert('L')
    print(f"Dimensiones: {img.size}")
    
    # Paso 0: Guardar imagen original
    img.save(f'{directorio_salida}/01_original.jpg')
    print("✓ Paso 0: Imagen original guardada")
    
    # Paso 1: Realce de contraste
    print("Procesando: Realce de contraste...")
    enhancer = ImageEnhance.Contrast(img)
    img_contrast = enhancer.enhance(2.0)
    img_contrast.save(f'{directorio_salida}/02_contrast_enhanced.jpg')
    print("✓ Paso 1: Realce de contraste aplicado (factor 2.0)")
    
    # Paso 2: Realce de nitidez
    print("Procesando: Realce de nitidez...")
    enhancer = ImageEnhance.Sharpness(img_contrast)
    img_sharp = enhancer.enhance(2.5)
    img_sharp.save(f'{directorio_salida}/03_sharpness_enhanced.jpg')
    print("✓ Paso 2: Realce de nitidez aplicado (factor 2.5)")
    
    # Paso 3: Ecualización del histograma
    print("Procesando: Ecualización de histograma...")
    img_array = np.array(img_sharp)
    img_equalized = ecualizar_histograma(img_array)
    img_eq = Image.fromarray(img_equalized)
    img_eq.save(f'{directorio_salida}/04_histogram_equalized.jpg')
    print("✓ Paso 3: Ecualización de histograma aplicada")
    
    # Paso 4: Filtro de suavizado para reducir ruido
    print("Procesando: Reducción de ruido...")
    img_smooth = img_eq.filter(ImageFilter.SMOOTH_MORE)
    img_smooth.save(f'{directorio_salida}/05_smoothed.jpg')
    print("✓ Paso 4: Suavizado aplicado para reducir ruido")
    
    # Paso 5: Realce de bordes
    print("Procesando: Realce de bordes...")
    img_edges = img_smooth.filter(ImageFilter.EDGE_ENHANCE_MORE)
    img_edges.save(f'{directorio_salida}/06_edges_enhanced.jpg')
    print("✓ Paso 5: Realce de bordes aplicado")
    
    # Paso 6: Detección de bordes
    print("Procesando: Detección de bordes...")
    img_find_edges = img_smooth.filter(ImageFilter.FIND_EDGES)
    img_find_edges.save(f'{directorio_salida}/07_edges_detected.jpg')
    print("✓ Paso 6: Detección de bordes aplicada")
    
    # Paso 7: Contraste final
    print("Procesando: Ajuste de contraste final...")
    enhancer = ImageEnhance.Contrast(img_edges)
    img_final_bw = enhancer.enhance(1.5)
    img_final_bw.save(f'{directorio_salida}/08_final_bw.jpg')
    print("✓ Paso 7: Contraste final aplicado (factor 1.5)")
    
    # Paso 8: Aplicar pseudocolor (similar a imagen de referencia OCT)
    print("Procesando: Aplicando mapa de color...")
    img_final_array = np.array(img_final_bw)
    img_rgb = aplicar_pseudocolor(img_final_array)
    img_colored = Image.fromarray(img_rgb)
    img_colored.save(f'{directorio_salida}/09_colored_custom.jpg')
    print("✓ Paso 8: Mapa de color personalizado aplicado (verde-amarillo-azul)")
    
    # Paso 9: Versión con brillo mejorado
    print("Procesando: Mejora de brillo...")
    enhancer_brightness = ImageEnhance.Brightness(img_final_bw)
    img_bright = enhancer_brightness.enhance(1.2)
    img_bright.save(f'{directorio_salida}/10_brightness_enhanced.jpg')
    print("✓ Paso 9: Realce de brillo aplicado (factor 1.2)")
    
    # Paso 10: Versión final optimizada (combinación de mejores parámetros)
    print("Procesando: Imagen final optimizada...")
    enhancer_contrast = ImageEnhance.Contrast(img_bright)
    img_final = enhancer_contrast.enhance(1.3)
    img_final.save(f'{directorio_salida}/11_final_optimized.jpg')
    print("✓ Paso 10: Imagen final optimizada guardada")
    
    print(f"\n{'='*60}")
    print(f"✓ PROCESAMIENTO COMPLETADO")
    print(f"{'='*60}")
    print(f"\nTodas las imágenes han sido guardadas en: '{directorio_salida}'")
    print(f"\nImágenes generadas:")
    
    for i, filename in enumerate(sorted(os.listdir(directorio_salida)), 1):
        print(f"  {i:2d}. {filename}")
    
    print(f"\nTotal de imágenes procesadas: {len(os.listdir(directorio_salida))}")


# Función principal
if __name__ == "__main__":
    # Configuración
    RUTA_IMAGEN_ENTRADA = 'OCT Database/VID/vid_vmd_27.jpg'
    DIRECTORIO_SALIDA = '/procesados'
    
    print("="*60)
    print("SCRIPT DE PROCESAMIENTO DE IMÁGENES OCT DE RETINA")
    print("="*60)
    print()
    
    # Verificar si existe la imagen de entrada
    if not os.path.exists(RUTA_IMAGEN_ENTRADA):
        print(f"❌ Error: No se encontró la imagen '{RUTA_IMAGEN_ENTRADA}'")
        print("Por favor, verifica que el archivo existe en el directorio actual.")
    else:
        # Procesar la imagen
        procesar_imagen_oct(RUTA_IMAGEN_ENTRADA, DIRECTORIO_SALIDA)
        
        print("\n" + "="*60)
        print("Proceso finalizado exitosamente")
        print("="*60)