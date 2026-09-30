import cv2
import numpy as np
import matplotlib.pyplot as plt

# Como primer objetivo tenemos que detectar los renglones en el excel

img_vacia = cv2.imread("grade_sheet_1.png",cv2.IMREAD_GRAYSCALE)
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

