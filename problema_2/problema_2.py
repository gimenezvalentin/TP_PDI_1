import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# Rutas de entrada y salida de archivos
if "__file__" in globals():
    base_dir = os.path.dirname(os.path.abspath(__file__))
else:
    base_dir = os.path.abspath("problema_2") if os.path.exists("problema_2") else os.getcwd()

input_dir = os.path.join(base_dir, "input")
output_dir = os.path.join(base_dir, "output")
os.makedirs(output_dir, exist_ok=True)

def detectar_lineas(img_th, eje, factor=0.5):
    '''
    Recibe una imagen umbralada y devuelve una lista con el inicio y fin de cada línea.
    '''
    suma = np.sum(img_th, eje)
    suma_th = suma > factor * suma.max()

    x = np.diff(suma_th)
    lineas_idxs = np.argwhere(x)
    ii = np.arange(0, len(lineas_idxs), 2)
    lineas_idxs[ii] += 1

    return lineas_idxs.reshape((-1, 2))


# Exploracion y deteccion de lineas en una planilla

img = cv2.imread(os.path.join(input_dir, "grade_sheet_1.png"), cv2.IMREAD_GRAYSCALE)
img_th = img < 150                  
plt.figure(), plt.imshow(img_th,cmap='gray'),plt.show(block=False)

lineas_h = detectar_lineas(img_th, 1)           # 22 líneas horizontales
lineas_v = detectar_lineas(img_th, 0)           # 8 líneas verticales

lineas_h, lineas_v

# Analizando el diferencial que marca la línea detectada del excel en lineas_h y líneas_v
# podemos deducir que, en este problema en particular, para toda línea del excel el rango de pixels
# es siempre 1. Ya que en líneas_h -> el límite_sup = límite_inf y en líneas_v -> límite_izq = límite_der
# para toda línea en el excel.

# Por heurística del problema con el análisis correspondiente vamos a utilizar esta información para
# borrar las líneas. También armamos la estructura para el posterior análisis.

# Comprobamos para todos los archivos

if __name__ == "__main__":
    for nombre_archivo in sorted(os.listdir(input_dir)):
        if nombre_archivo.startswith("grade_sheet_"):
            ruta_imagen = os.path.join(input_dir, nombre_archivo)
            img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
            img_th = img < 150                              
            
            lineas_h = detectar_lineas(img_th, 1)
            lineas_v = detectar_lineas(img_th, 0)

            print(f"===== {nombre_archivo} =====")
            # Si (límite_izq = límite_der) y (límite_sup = límite_inf) para todas las líneas
            if all(x[1] - x[0] == 0 for x in lineas_v) and all(y[1] - y[0] == 0 for y in lineas_h):
                print("Todas las líneas miden 1")
            else:
                print("Alguna línea tiene un grosor distinto")
            
# Guardamos los registros

img = cv2.imread(os.path.join(input_dir, "grade_sheet_1.png"), cv2.IMREAD_GRAYSCALE)
img_th = img < 150                              
lineas_h = detectar_lineas(img_th, 1)           
lineas_v = detectar_lineas(img_th, 0)

nombres_campos = ["Legajo", "Nombre y apellido", "Parcial 1", "Parcial 2", "Parcial 3", "Condición Final"]
registros = []
for ir in range(1, len(lineas_h) - 1):          # +1 (por heurística) usamos para descartar la línea superior del excel
    y1 = lineas_h[ir][1] + 1                    # límite superior
    y2 = lineas_h[ir + 1][0]                    # límite inferior
    campos = []
    for ic in range(1, len(lineas_v) - 1):      # +1 (por heurística) usamos para descartar la línea izquierda del excel
        x1 = lineas_v[ic][1] + 1                # límite izquierdo  
        x2 = lineas_v[ic + 1][0]                # límite derecho
        campos.append({"nombre": nombres_campos[ic - 1], "cord": [y1, x1, y2, x2], "img": img[y1:y2, x1:x2]})
    registros.append({"ir": ir, "campos": campos})   

# Visualizamos la primera fila de registros
plt.figure()
plt.suptitle("Primer fila de registros")
subplot = 321
for fila in registros[0]["campos"]:
    plt.subplot(subplot)
    plt.title(fila["nombre"])
    plt.imshow(fila["img"],cmap='gray')
    subplot += 1
plt.show(block=False)

def analizar_celda(celda, th_area=2, th_espacio=7):
    '''
    Recibe una celda de registros y devuelve:
    - n_car: Cantidad de caracteres detectados
    - n_pal: Cantidad de palabras detectadas
    - caracteres: Lista de stats sobre los caracteres detectados [x,y,ancho,alto,área]
    - celda_bin: celda binarizada
    '''
    celda_bin = (celda < 150).astype(np.uint8)          # Binarizo: letras = 1, fondo = 0

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(celda_bin, connectivity=8, ltype=cv2.CV_32S)
    alto_celda, ancho_celda = celda_bin.shape

    caracteres = []
    for st in stats[1:]:                                    # stats[0] es el fondo, lo salteamos
        x, y, ancho, alto, area = st
        if area <= th_area:                                 # si el área es muy chica: ruido
            continue
        if ancho >= ancho_celda - 2 or alto >= alto_celda - 2:   # si quedó algún resto de línea, lo salteamos
            continue
        caracteres.append(st)
    caracteres = sorted(caracteres, key=lambda st: st[0])   # ordenamos por x (de izquierda a derecha)
    n_car = len(caracteres)
    
    # Cuento palabras mirando el espacio entre letras vecinas
    if n_car == 0:
        n_pal = 0
    else:
        n_pal = 1                                           # si hay letras, hay al menos 1 palabra
        for i in range(1, n_car):
            letra_anterior = caracteres[i - 1]
            letra_actual = caracteres[i]
            fin_anterior = letra_anterior[0] + letra_anterior[2]    # x + ancho
            inicio_actual = letra_actual[0]                         # x
            espacio = inicio_actual - fin_anterior
            if espacio > th_espacio:                          # espacio grande = nueva palabra
                n_pal += 1
    return n_car, n_pal, np.array(caracteres), celda_bin

# Distintos casos según th_area

## Con th_area = 30, detecta la mayoría de letras.
celda = registros[0]["campos"][0]["img"]
n_car, n_pal, stats, celda_bin = analizar_celda(celda,th_area=30)
celda_color = cv2.cvtColor(celda, cv2.COLOR_GRAY2RGB)
for st in stats:
    cv2.rectangle(celda_color, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]), color=(0,255,0), thickness=1)
plt.figure(), plt.imshow(celda_color), plt.title(f"{n_car} caracteres - {n_pal} palabras"), plt.show(block=False)


## Con th_area = 15, incluye caracteres como "/".
celda = registros[0]["campos"][0]["img"]
n_car, n_pal, stats, celda_bin = analizar_celda(celda,th_area=15)
celda_color = cv2.cvtColor(celda, cv2.COLOR_GRAY2RGB)
for st in stats:
    cv2.rectangle(celda_color, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]), color=(0,255,0), thickness=1)
plt.figure(), plt.imshow(celda_color), plt.title(f"{n_car} caracteres - {n_pal} palabras"), plt.show(block=False)


## Optimo th_area = 2, incluye "-" y puede descartar algun pixel suelto que genere ruido en la imagen.
celda = registros[0]["campos"][0]["img"]
n_car, n_pal, stats, celda_bin = analizar_celda(celda)
celda_color = cv2.cvtColor(celda, cv2.COLOR_GRAY2RGB)
for st in stats:
    cv2.rectangle(celda_color, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]), color=(0,255,0), thickness=1)
plt.figure(), plt.imshow(celda_color), plt.title(f"{n_car} caracteres - {n_pal} palabras"), plt.show(block=False)

def validar_campo(nombre, n_car, n_pal):
    '''
    Aplica las restricciones del enunciado a un campo:
    - Legajo: 8 caracteres en 1 palabra.
    - Nombre y apellido: al menos 2 palabras y hasta 12 caracteres.
    - Parciales: 1 o 2 caracteres juntos.
    - Condición Final: 1 caracter.
    '''
    if nombre == "Legajo":
        return n_car == 8 and n_pal == 1
    if nombre == "Nombre y apellido":
        return n_pal >= 2 and n_car <= 12
    if nombre == "Condición Final":
        return n_car == 1
    return 1 <= n_car <= 2 and n_pal == 1       # Parciales 1, 2 y 3

# Analizamos primera fila de registros.
for reg in registros[0]["campos"]:
    celda = reg["img"]
    n_car, n_pal, stats, celda_bin = analizar_celda(celda)
    estado = "OK" if validar_campo(reg["nombre"],n_car,n_pal) else "MAL"
    print(f"{reg["nombre"]}: {estado}")


def clasificar_condicion(celda_bin, stats):
    '''
    Debe recibir una celda binaria con la condición final y sus stats.
    Devuelve si la condición es:
    -"A": APROBADO
    -"R": RECUPERA
    -"L": LIBRE
    '''
    x, y, w, h, area = stats[0]
    letra = celda_bin[y:y+h, x:x+w] 

    fondo = cv2.copyMakeBorder(1 - letra, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)   # borde para unir el fondo exterior (ayuda con IA)
    num_labels, labels, st, cen = cv2.connectedComponentsWithStats(fondo, connectivity=8, ltype=cv2.CV_32S)
    n_agujeros = num_labels - 2                                                       # resto etiqueta 0 y el fondo exterior
    col_izq_llena = letra[:, 0].sum() == h                                            # true si la columna izquierda es todo 0
    if n_agujeros == 0 and col_izq_llena:
        return "L"      
    if n_agujeros == 1 and col_izq_llena:
        return "R"
    return "A"

# Analizamos un rango de condiciones
for i in range(0,10):
    cond = registros[i]["campos"][5]
    n_car, n_pal, stat, celda_bin = analizar_celda(cond["img"])

    # descartamos condiciones inválidas
    if validar_campo(cond["nombre"],n_car,n_pal):
        letra = clasificar_condicion(celda_bin, stat)
        print(f"Registro {i + 1}:")
        print(f"Condicion final: {letra}")
        print("---------------------")

# Unificamos todas las funciones para analizar una tabla entera
def procesar_planilla(nombre_archivo, input_dir, output_dir):
    '''
    Procesa una planilla completa:
    - imprime OK/MAL por celda de cada registro,
    - arma una imagen con los nombres de los alumnos con registro correcto y condición L o R, con una etiqueta LIBRE/RECUPERA,
    - guarda un CSV con el resultado de cada validación en 'output'.
    '''
    ruta_imagen = os.path.join(input_dir, nombre_archivo)
    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se encontró la imagen en: {ruta_imagen}.\nVerifica que esté dentro de 'problema_2/input/'.")
    img_th = img < 150
    lineas_h = detectar_lineas(img_th, 1)
    lineas_v = detectar_lineas(img_th, 0)
    filas_csv = []
    filas_salida = []
    nombres_campos = ["Legajo", "Nombre y apellido", "Parcial 1", "Parcial 2", "Parcial 3", "Condición Final"]

    for ir in range(1, len(lineas_h) - 1):
        y1 = lineas_h[ir][1] + 1        # +1 (heurística)
        y2 = lineas_h[ir + 1][0]
        print(f"> Registro {ir}:")

        resultados = []
        for ic in range(1, len(lineas_v) - 1):
            x1 = lineas_v[ic][1] + 1
            x2 = lineas_v[ic + 1][0]
            nombre = nombres_campos[ic - 1]

            n_car, n_pal, stats, celda_bin = analizar_celda(img[y1:y2, x1:x2])
            ok = validar_campo(nombre, n_car, n_pal)
            resultados.append("OK" if ok else "MAL")
            print(f"> {nombre}: {'OK' if ok else 'MAL'}")                # (a) mostramos por terminal
        print(">")

        filas_csv.append([ir] + resultados)                              # ejemplo: [ir = 1, Legajo = OK,..., Condicion final = MAL] 
        condicion_final = celda_bin                                      # celda_bin = ultima celda del excel

        if all(r == "OK" for r in resultados):                           # (b) solo registros correctos
            cond = clasificar_condicion(condicion_final, stats)          # celda_bin/stats de la última columna
            if cond in ("L", "R"):
                x1n = lineas_v[2][1] + 1                                 # columna Nombre y Apellido
                x2n = lineas_v[3][0]                                     
                crop = cv2.cvtColor(img[y1:y2, x1n:x2n], cv2.COLOR_GRAY2RGB)
                etiqueta = np.full((y2 - y1, 110, 3), 255, dtype=np.uint8)  # (alto, ancho, canales)
                color = (255, 140, 0) if cond == "R" else (255, 0, 0)       # (255, 140, 0) = naranja
                texto = "RECUPERA" if cond == "R" else "LIBRE"
                cv2.putText(etiqueta, texto, (5, (y2 - y1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)  # (5, (y2 - y1) - 8) = abajo a la izquierda de la etiqueta
                filas_salida.append(np.hstack([crop, etiqueta]))            # np.hstack: pone una matriz al lado de la otra (ayuda con IA)
    
    nombre_base = os.path.splitext(nombre_archivo)[0]   # devuelve ("grade_sheet_...", ".png")                               
    with open(os.path.join(output_dir, f"{nombre_base}_validacion.csv"), "w", newline="", encoding="utf-8") as f:   # (c)
        writer = csv.writer(f)
        writer.writerow(["ID"] + nombres_campos)
        writer.writerows(filas_csv)
    if filas_salida:                                                     # (b) única imagen de salida
        salida = np.vstack(filas_salida)                                 # np.vstack: apila las matrices verticalmente
        plt.figure(), plt.imshow(salida), plt.title(f"No aprobados - {nombre_archivo}"), plt.show(block=False)
        cv2.imwrite(os.path.join(output_dir, f"{nombre_base}_no_aprobados.png"), cv2.cvtColor(salida, cv2.COLOR_RGB2BGR))   # OpenCV guarda en BGR

# ---------- Ejecucion y guardado final ----------

if __name__ == "__main__":
    # (d) Procesamiento cíclico de todas las planillas de 'input' (sin la vacía)
    for nombre_archivo in sorted(os.listdir(input_dir)):
        if nombre_archivo.startswith("grade_sheet_") and nombre_archivo != "grade_sheet_empty.png":
            print(f"===== {nombre_archivo} =====")
            procesar_planilla(nombre_archivo, input_dir, output_dir)
