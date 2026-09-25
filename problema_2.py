import cv2
import numpy as np
import matplotlib.pyplot as plt

# Como primer objetivo tenemos que detectar los renglones en el excel

img_vacia = cv2.imread("grade_sheet_empty.png",cv2.IMREAD_GRAYSCALE)
plt.figure(), plt.imshow(img_vacia, cmap='gray'), plt.title("Excel Vacio"), plt.show(block=False)

img_vacia_zeros = img_vacia < 5
plt.figure(), plt.imshow(img_vacia_zeros, cmap='gray'), plt.title("Excel Vacio Bool"), plt.show(block=False)

cols = np.sum(img_vacia_zeros,0)
cols_idx = np.argwhere(cols == cols.max())
#r_idxs = np.reshape(cols_idx, (-1,2)) # Re-ordeno de a pares

columnas = []
for idx,c in enumerate(cols_idx):
    if idx == 0:
        columna_anterior = c
        continue

    columnas.append(
    {"idx": idx,
    "img": img_vacia[:,columna_anterior[0]:c[0]]
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

plt.figure(), plt.imshow(columnas[4]["img"], cmap='gray'), plt.title(f"Columna {columnas[2]["idx"]}"), plt.show(block=False)


