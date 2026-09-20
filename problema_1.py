import cv2
import numpy as np
from random import randint
import matplotlib.pyplot as plt

# Cargar y mostrar imagen
img = cv2.imread("Imagen_con_detalles_escondidos.tif",cv2.IMREAD_GRAYSCALE)
plt.figure(), plt.imshow(img, cmap='gray'), plt.title("Imagen Original"), plt.show(block=False)

# Imagen booleana para ver los resultados ocultos
img_zeros = img < 5
plt.figure(), plt.imshow(img_zeros, cmap='gray'), plt.title("Imagen Booleana"), plt.show(block=False)

# Crear borde alrededor (FALTA)
# // TODO: todavia no se para que sirve
top = int(0.05 * img.shape[0])  # shape[0] = rows
bottom = top
left = int(0.05 * img.shape[1])  # shape[1] = cols
right = left
value = [randint(0, 255), randint(0, 255), randint(0, 255)]
img_borde = cv2.copyMakeBorder(img,top,bottom,left,right,borderType=cv2.BORDER_CONSTANT,value=value)

plt.figure(), plt.imshow(img_borde, cmap='gray'), plt.title("Imagen Borde"), plt.show(block=False)


def ecualizacion_local_histograma(imagen ,MxN: tuple)-> None:
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
    # Creamos margenes en los costados de la imagen que es la mitad de la ventana de procesamiento MxN
    top = M // 2
    bottom = top
    left = N // 2
    right = left
    
    imagen_borde = cv2.copyMakeBorder(img,top,bottom,left,right,borderType=cv2.BORDER_DEFAULT) 
    return imagen_borde

    

prueba = ecualizacion_local_histograma(img, (3,3))
prueba.shape
plt.figure(), plt.imshow(prueba, cmap='gray'), plt.title("Imagen Ecualizada"), plt.show(block=False)

print(img.shape)