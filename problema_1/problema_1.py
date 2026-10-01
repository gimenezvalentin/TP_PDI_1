import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def ecualizacion_local_histograma(imagen:np.ndarray,MxN: tuple,border = cv2.BORDER_REPLICATE)-> np.ndarray:
    '''
    Recibe una imagen a procesar y un tamaño de ventana de procesamiento.

    El procedimiento consiste en definir 
    una ventana cuadrada o rectangular de tamaño MxN y desplazar su centro de píxel en píxel a lo largo 
    de la imagen. En cada posición, se calcula el histograma correspondiente a los píxeles contenidos 
    dentro de la ventana y, mediante el mismo, se obtiene la transformación local de la ecualización del 
    histograma.  
    '''
    if MxN[0] > imagen.shape[0] or MxN[1] > imagen.shape[1]:
        return print("La ventana de procesamiento es mas grande que la imagen")
    
    M,N = MxN
    
    #impares para centro unico
    if M % 2 == 0 or N % 2 == 0:
        return print("El tamaño de la ventana (MxN) debe estar compuesto por números impares.")
    # Creamos margenes en los costados de la imagen que es la mitad de la ventana de procesamiento MxN
    top = bottom = M // 2
    left = right = N // 2
    
    imagen_borde = cv2.copyMakeBorder(imagen,top,bottom,left,right,borderType=border) 
    imagen_ecualizada = np.zeros(imagen.shape)
    for i in range(imagen.shape[0]):
        for j in range(imagen.shape[1]):
            # Ventana MxN centrada en el píxel (i, j) original
            ventana = imagen_borde[i : i + M, j : j + N]
            ventana_eq = cv2.equalizeHist(ventana)
            imagen_ecualizada[i, j] = ventana_eq[M // 2, N //2]
    
    return imagen_ecualizada

#----------Ejecucion-----------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Determinación robusta de la carpeta del ejercicio (problema_1)
    if "__file__" in globals():
        base_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        base_dir = os.path.abspath("problema_1") if os.path.exists("problema_1") else os.getcwd()

    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "output")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Carga de imagen desde carpeta 'input' y contraste con ecualización global
    ruta_imagen = os.path.join(input_dir, "Imagen_con_detalles_escondidos.tif")
    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"No se encontró la imagen en: {ruta_imagen}.\nVerifica que esté dentro de 'problema_1/input/'.")

    ecua_global = cv2.equalizeHist(img)

    plt.figure(figsize=(10, 8))
    plt.subplot(221), plt.imshow(img, cmap='gray'), plt.title("Imagen Original")
    plt.subplot(222), plt.hist(img.flatten(), 256, [0, 256]), plt.title("Histograma Original")
    plt.subplot(223), plt.imshow(ecua_global, cmap='gray'), plt.title("Ecualización Global")
    plt.subplot(224), plt.hist(ecua_global.flatten(), 256, [0, 256]), plt.title("Histograma Ecualizado Global")
    plt.tight_layout()
    plt.show(block=False)

    # 2. PUNTO B: Detección de objetos ocultos con el Kernel Óptimo (17x17)
    kernel_optimo = (17, 17)
    img_optima = ecualizacion_local_histograma(img, kernel_optimo)

    plt.figure(figsize=(8, 4))
    plt.suptitle("Punto b: Revelado de objetos ocultos (Ventana óptima 17x17)", fontsize=13)
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title("Imagen Original")
    plt.subplot(122), plt.imshow(img_optima, cmap='gray'), plt.title("Ecualización Local 17x17")
    plt.tight_layout()
    plt.show(block=False)

    # "--- DETALLES OCULTOS DETECTADOS (Ventana 17x17) ---"
    # "1. Superior Izquierda : Cuadrado sólido pequeño"
    # "2. Superior Derecha   : Línea diagonal ascendente"
    # "3. Centro             : Letra 'a' minúscula"
    # "4. Inferior Izquierda : Cuatro líneas horizontales paralelas"
    # "5. Inferior Derecha   : Círculo sólido"

    # 3. PUNTO C: Influencia del tamaño de la ventana (Comparativa)
    ventana3x3 = ecualizacion_local_histograma(img, (3, 3))
    ventana7x7 = ecualizacion_local_histograma(img, (7, 7))
    # Reutilizamos img_optima (17x17) para no volver a calcularla
    ventana35x35 = ecualizacion_local_histograma(img, (35, 35))

    plt.figure(figsize=(10, 8))
    plt.suptitle("Punto c: Influencia del tamaño de la ventana (MxN)", fontsize=14)
    plt.subplot(221), plt.imshow(ventana3x3, cmap='gray'), plt.title("Ventana 3x3 (Huecos y alto ruido)")
    plt.subplot(222), plt.imshow(ventana7x7, cmap='gray'), plt.title("Ventana 7x7 (Transición)")
    plt.subplot(223), plt.imshow(img_optima, cmap='gray'), plt.title("Ventana 17x17 (Óptima)")
    plt.subplot(224), plt.imshow(ventana35x35, cmap='gray'), plt.title("Ventana 35x35 (Menor ruido, mayor costo)")
    plt.tight_layout()
    plt.show()

    # Guardado de todas las imágenes resultantes en 'problema_1/output'
    cv2.imwrite(os.path.join(output_dir, "ecualizacion_global.tif"), ecua_global)
    cv2.imwrite(os.path.join(output_dir, "ecualizacion_local_17x17.tif"), img_optima.astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, "ecualizacion_local_3x3.tif"), ventana3x3.astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, "ecualizacion_local_7x7.tif"), ventana7x7.astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, "ecualizacion_local_35x35.tif"), ventana35x35.astype(np.uint8))
