import cv2
import numpy as np
from random import randint
import matplotlib.pyplot as plt

# Cargar y mostrar imagen
img = cv2.imread("Imagen_con_detalles_escondidos.tif",cv2.IMREAD_GRAYSCALE)
plt.figure(), plt.imshow(img, cmap='gray'), plt.title("Imagen Original"), plt.show(block=False)

# Imagen booleana para ver los resultados ocultos
img_zeros = img < 2
plt.figure(), plt.imshow(img_zeros, cmap='gray'), plt.title("Imagen Booleana"), plt.show(block=False)

# Crear borde alrededor (FALTA) - BORRAR
# // TODO: todavia no se para que sirve
top = int(0.05 * img.shape[0])  # shape[0] = rows
bottom = top
left = int(0.05 * img.shape[1])  # shape[1] = cols
right = left
value = [randint(0, 255), randint(0, 255), randint(0, 255)]
img_borde = cv2.copyMakeBorder(img,top,bottom,left,right,borderType=cv2.BORDER_CONSTANT,value=value)
plt.figure(), plt.imshow(img_borde, cmap='gray'), plt.title("Imagen Borde"), plt.show(block=False)

# ANALIZANDO POR QUÉ NO FUNCIONA LA ECUALIZACION TOTAL DE LA IMAGEN
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.figure(), plt.hist(img.flatten(), 256, [0, 256]), plt.title("Histograma"), plt.show(block=False)
ecua = cv2.equalizeHist(img)
plt.figure(), plt.imshow(ecua, cmap='gray'), plt.title("Imagen Ecualizada Total"), plt.show(block=False)
plt.figure(), plt.hist(ecua.flatten(), 256, [0, 256]), plt.title("Ecualizacion"), plt.show(block=False)


def ecualizacion_local_histograma(imagen ,MxN: tuple, border=cv2.BORDER_REPLICATE)-> None:
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
    
    imagen_borde = cv2.copyMakeBorder(img,top,bottom,left,right,borderType=border)

    # Empezamos a recorrer la imagen
    imagen_ecualizada = imagen_borde.copy()
    for i in range(imagen.shape[0]):
        for j in range(imagen.shape[1]):
        # Ventana MxN centrada en el píxel (i, j) original
            ventana = imagen_borde[i : i + M, j : j + N]
            ventana_eq = cv2.equalizeHist(ventana)
            imagen_ecualizada[i : i + M, j : j + N] = ventana_eq
    
    return imagen_ecualizada
   
# Probamos con distintos tipos de ventana

ventana3x3 = ecualizacion_local_histograma(img, (3,3))
ventana5x5 = ecualizacion_local_histograma(img, (5,5))
ventana15x15 = ecualizacion_local_histograma(img, (15,15))
ventana30x30 = ecualizacion_local_histograma(img, (30,30))
#prueba.shape
plt.figure()
plt.subplot(221), plt.imshow(ventana3x3, cmap='gray'), plt.title("Imagen Ecualizada 3x3")
plt.subplot(222), plt.imshow(ventana5x5, cmap='gray'), plt.title("Imagen Ecualizada 5x5")
plt.subplot(223), plt.imshow(ventana15x15, cmap='gray'), plt.title("Imagen Ecualizada 15x15")
plt.subplot(224), plt.imshow(ventana30x30, cmap='gray'), plt.title("Imagen Ecualizada 30x30")
plt.show(block=False)

print(img.shape)

'''
1. el borde con el replicate esta bien?
2. la imagen quedo mas grande que la original
3. el kernel toma el valor del borde creado, esta bien?

'''

# Comprobamos tipo de borde
## Borde 15x15

ventana15x15_replicate = ecualizacion_local_histograma(img, (15,15))
ventana15x15_constant = ecualizacion_local_histograma(img, (15,15),border=cv2.BORDER_CONSTANT)
ventana15x15_reflect = ecualizacion_local_histograma(img, (15,15),border=cv2.BORDER_REFLECT)
ventana15x15_default = ecualizacion_local_histograma(img, (15,15),border=cv2.BORDER_DEFAULT)
ventana15x15_isolated = ecualizacion_local_histograma(img, (15,15),border=cv2.BORDER_ISOLATED)
ventana15x15_wrap = ecualizacion_local_histograma(img, (15,15),border=cv2.BORDER_WRAP)

plt.figure()
plt.subplot(231), plt.imshow(ventana15x15_replicate, cmap='gray'), plt.title("REPLICATE")
plt.subplot(232), plt.imshow(ventana15x15_constant, cmap='gray'), plt.title("CONSTANT")
plt.subplot(233), plt.imshow(ventana15x15_reflect, cmap='gray'), plt.title("REFLECT")
plt.subplot(234), plt.imshow(ventana15x15_default, cmap='gray'), plt.title("DEFAULT")
plt.subplot(235), plt.imshow(ventana15x15_isolated, cmap='gray'), plt.title("ISOLATED")
plt.subplot(236), plt.imshow(ventana15x15_wrap, cmap='gray'), plt.title("WRAP")
plt.show(block=False)

## Borde 3x3

ventana3x3_replicate = ecualizacion_local_histograma(img, (3,3))
ventana3x3_constant = ecualizacion_local_histograma(img, (3,3),border=cv2.BORDER_CONSTANT)
ventana3x3_reflect = ecualizacion_local_histograma(img, (3,3),border=cv2.BORDER_REFLECT)
ventana3x3_default = ecualizacion_local_histograma(img, (3,3),border=cv2.BORDER_DEFAULT)
ventana3x3_isolated = ecualizacion_local_histograma(img, (3,3),border=cv2.BORDER_ISOLATED)
ventana3x3_wrap = ecualizacion_local_histograma(img, (3,3),border=cv2.BORDER_WRAP)

plt.figure()
plt.subplot(231), plt.imshow(ventana3x3_replicate, cmap='gray'), plt.title("REPLICATE")
plt.subplot(232), plt.imshow(ventana3x3_constant, cmap='gray'), plt.title("CONSTANT")
plt.subplot(233), plt.imshow(ventana3x3_reflect, cmap='gray'), plt.title("REFLECT")
plt.subplot(234), plt.imshow(ventana3x3_default, cmap='gray'), plt.title("DEFAULT")
plt.subplot(235), plt.imshow(ventana3x3_isolated, cmap='gray'), plt.title("ISOLATED")
plt.subplot(236), plt.imshow(ventana3x3_wrap, cmap='gray'), plt.title("WRAP")
plt.show(block=False)

## Borde 30x30

ventana30x30_replicate = ecualizacion_local_histograma(img, (30,30))
ventana30x30_constant = ecualizacion_local_histograma(img, (30,30),border=cv2.BORDER_CONSTANT)
ventana30x30_reflect = ecualizacion_local_histograma(img, (30,30),border=cv2.BORDER_REFLECT)
ventana30x30_default = ecualizacion_local_histograma(img, (30,30),border=cv2.BORDER_DEFAULT)
ventana30x30_isolated = ecualizacion_local_histograma(img, (30,30),border=cv2.BORDER_ISOLATED)
ventana30x30_wrap = ecualizacion_local_histograma(img, (30,30),border=cv2.BORDER_WRAP)

plt.figure()
plt.subplot(231), plt.imshow(ventana30x30_replicate, cmap='gray'), plt.title("REPLICATE")
plt.subplot(232), plt.imshow(ventana30x30_constant, cmap='gray'), plt.title("CONSTANT")
plt.subplot(233), plt.imshow(ventana30x30_reflect, cmap='gray'), plt.title("REFLECT")
plt.subplot(234), plt.imshow(ventana30x30_default, cmap='gray'), plt.title("DEFAULT")
plt.subplot(235), plt.imshow(ventana30x30_isolated, cmap='gray'), plt.title("ISOLATED")
plt.subplot(236), plt.imshow(ventana30x30_wrap, cmap='gray'), plt.title("WRAP")
plt.show(block=False)

# Conclusion: el tipo de borde afecta principalmente a ventanas de procesamiento grandes a los \n
# contornos, no afecta el procesamiento interno de la imagen.