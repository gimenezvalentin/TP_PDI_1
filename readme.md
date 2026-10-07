# Trabajo Práctico N°1 - Procesamiento de Imágenes I

Repositorio con las resoluciones del primer trabajo práctico de la materia Procesamiento de Imágenes I (IA 4.4) de la Tecnicatura Universitaria en Inteligencia Artificial (FCEIA - UNR).

El trabajo se divide en dos problemas principales:
1. **Ecualización local de histograma:** Algoritmo de ecualización mediante ventana deslizante para revelar detalles ocultos en zonas de bajo contraste local.
2. **Validación de planillas:** Algoritmo de visión artificial que utiliza proyecciones, detección de contornos y componentes conectadas para aislar, validar y clasificar el contenido de planillas de calificaciones escaneadas.

## Requisitos e Instalación

La versión de Python requerida es **>3.15**. Los paquetes necesarios para ejecutar los scripts se encuentran listados en el archivo `requirements.txt`:

*   `matplotlib==3.11.1`
*   `numpy==2.5.2`
*   `opencv-contrib-python==5.0.0.93`

Se recomienda crear un entorno virtual antes de instalar las dependencias:

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno (Windows)
.\venv\Scripts\activate
# Activar entorno (Linux/Mac)
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```
## Uso y Ejecución de Programas
Las resoluciones están organizadas en carpetas separadas para cada problema, conteniendo sus propios scripts (problema_1.py y problema_2.py) junto con sus respectivas subcarpetas de entrada y salida (input/ y output/).

Estructura general de los directorios:

```bash
  problema_*/
    input/
      ...
    output/
      ...
  problema_*.py
```
Para ejecutar las resoluciones, ubicarse en la raíz del problema elegido y correr el script correspondiente:

```bash
# Para ejecutar el Problema 1
python problema_1.py

# Para ejecutar el Problema 2
python problema_2.py
```

> Nota: La ejecución del Problema 1 desplegará ventanas interactivas con las gráficas de Matplotlib. La ejecución del Problema 2 mostrará la validación por terminal y generará archivos automáticos (imágenes recortadas y un .csv) dentro de su carpeta output/.

Los enunciados detallados se encuentran en el archivo TUIA_PDI_TP1_2026_C2.pdf dentro de la raíz del directorio.

### Integrantes:

* Aguilera, Joaquín
* Arias, Federico
* Bousoño, Guillermina
* Gimenez, Valentin
