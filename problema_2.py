import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

# Como primer objetivo tenemos que detectar los renglones en el excel

img_vacia = cv2.imread("grade_sheet_empty.png",cv2.IMREAD_GRAYSCALE)
plt.figure(), plt.imshow(img_vacia, cmap='gray'), plt.title("Excel Vacio"), plt.show(block=False)

img_vacia_zeros = img_vacia < 5
plt.figure(), plt.imshow(img_vacia_zeros, cmap='gray'), plt.title("Excel Vacio Bool"), plt.show(block=False)

cols = np.sum(img_vacia_zeros,0)
cols_ordenado = np.sort(np.unique(cols))
cols_idx = np.argwhere(cols == cols_ordenado[-1])
cols_idx = np.append(cols_idx, np.argwhere(cols == cols_ordenado[-2]))
cols_idx = np.sort(cols_idx)
#r_idxs = np.reshape(cols_idx, (-1,2)) # Re-ordeno de a pares

columnas = []
for idx,c in enumerate(cols_idx):
    if idx == 0:
        columna_anterior = c
        continue

    columnas.append(
    {"idx": idx,
    "img": img_vacia[:,columna_anterior:c],
    "inicio_col": columna_anterior,
    "termina_col": c
    }
    )
    columna_anterior = c

plt.figure()
subplot = 101 + len(columnas)*10
for c in columnas:
    plt.subplot(subplot)
    plt.imshow(c["img"],cmap="gray")
    plt.title(f"Columna {c["idx"]}")
    subplot += 1
plt.show()

# -----------------------

filas = []
for col in columnas:
    columna_zeros = col["img"] < 5
    #plt.figure(), plt.imshow(col["img"][1],cmap='gray'),plt.show()
    rows = np.sum(columna_zeros,1)
    rows_idx = np.argwhere(rows == rows.max())

    for idx,fila in enumerate(rows_idx):
        if idx == 0:
            fila_anterior = fila
            continue
        

        filas.append(
        {"idx": idx,
         "indice_columna": col["idx"],
        "img": img_vacia[fila_anterior[0]:fila[0],col["inicio_col"]:col["termina_col"]]
        }
        )
        #plt.figure(), plt.imshow(img_vacia[fila_anterior[0]:fila[0],col["inicio_col"]:col["termina_col"]],cmap='gray'),plt.show()
        fila_anterior = fila

plt.figure()
subplot = 241
for f in filas:
    plt.subplot(subplot)
    plt.imshow(f["img"],cmap="gray")
    plt.title(f"Fila {f["idx"]}")
    subplot += 1
    if subplot == 249:
        break
plt.show(block=False)

# DETECTAR CARACTES

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filas[0]["img"], connectivity=8, ltype=cv2.CV_32S)  # https://docs.opencv.org/4.5.3/d3/dc0/group__imgproc__shape.html#ga107a78bf7cd25dec05fb4dfc5c9e765f
# num_labels: Cantidad de elementos
# labels: Matriz con etiquetas
# stats: Matriz de estadisticas de los elementos (bounding box + area)
# centroids: Centroides de elementos
th_area = 160
ix_area = stats[:,-1] > th_area
stats = stats[ix_area,:]

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filas[0]["img"],stats=stats, connectivity=8, ltype=cv2.CV_32S)  # https://docs.opencv.org/4.5.3/d3/dc0/group__imgproc__shape.html#ga107a78bf7cd25dec05fb4dfc5c9e765f

num_labels
stats
centroids
labels
np.unique(labels)
plt.figure(), plt.imshow(labels,cmap='gray'),plt.show(block=False)
plt.figure(), plt.imshow(filas[0]["img"],cmap='gray'),plt.show(block=False)

def detectar_lineas(img_th, eje, factor=0.5):
    suma = np.sum(img_th, eje)                  # eje=0 -> suma por columna | eje=1 -> suma por fila (como en ej2.py)
    suma_th = suma > factor * suma.max()  # True donde hay línea (muchos más píxeles que en el resto)
    print(suma_th)
    x = np.diff(suma_th)                        # Igual que en Letras: detecta los cambios F->T y T->F
    lineas_idxs = np.argwhere(x)
    ii = np.arange(0, len(lineas_idxs), 2)      # Corrijo los inicios (+1), como en Letras
    lineas_idxs[ii] += 1
    return lineas_idxs.reshape((-1, 2))         # Cada fila: [inicio, fin] de una línea

img = cv2.imread("grade_sheet_1.png", cv2.IMREAD_GRAYSCALE)
img_th = img < 150                              # Con < 5 las letras quedan cortadas
lineas_h = detectar_lineas(img_th, 1)           # 22 líneas horizontales
lineas_v = detectar_lineas(img_th, 0)           # 8 líneas verticales

plt.figure(), plt.imshow(img_th,cmap='gray'),plt.show(block=False)

nombres_campos = ["Legajo", "Nombre y apellido", "Parcial 1", "Parcial 2", "Parcial 3", "Condición Final"]
registros = []
for ir in range(1, len(lineas_h) - 1):
    y1 = lineas_h[ir][1] + 1
    y2 = lineas_h[ir + 1][0]
    campos = []
    for ic in range(1, len(lineas_v) - 1):
        x1 = lineas_v[ic][1] + 1
        x2 = lineas_v[ic + 1][0]
        campos.append({"nombre": nombres_campos[ic - 1], "cord": [y1, x1, y2, x2], "img": img[y1:y2, x1:x2]})
        registros.append({"ir": ir, "campos": campos})
    
registros
    

    
