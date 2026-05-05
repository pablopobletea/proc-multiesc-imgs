# Tarea 2 — Procesamiento Multiescala de Imágenes

![Python](https://img.shields.io/badge/python-3.10+-blue)
![Conda](https://img.shields.io/badge/conda-supported-brightgreen)
![Jupyter](https://img.shields.io/badge/jupyter-notebook-orange)
![Image Processing](https://img.shields.io/badge/domain-image_processing-blueviolet)
![Status](https://img.shields.io/badge/status-completed-success)
![License](https://img.shields.io/badge/license-MIT-green)

## Descripción

Repositorio de la **Tarea 2** de la asignatura: IEE3787 - Procesamiento Multiescala de Imágenes

Se abordan dos problemas principales:

- Evaluación perceptual de imágenes 3D (MRI) utilizando una extensión de HaarPSI (WavePsi)
- Denoising de imágenes 2D mediante Transformada Discreta de Wavelets (DWT)

El proyecto incluye implementación en Python, análisis cuantitativo y cualitativo, generación de figuras e informe final.

---

# 🧩 Tarea 2 — MRI 3D y Wavelets

## Contenido

### Parte 1 — MRI 3D
- Reconstrucciones: streaking, reconNDI*, recontv*
- Métricas: RMSE, SSIM, WavePsi
- Visualizaciones: slices, MIP, mapas de error, mapas HS/W

### Parte 2 — Denoising 2D
- Imágenes: Moon, Deep sky
- Ruido: gaussiano y uniforme
- Métodos: DWT, hard y soft thresholding
- Evaluación: RMSE, SSIM, WavePsi

---

# 🧩 Tarea 3 — Starlet, MMT y QSM

## Descripción

Se estudia el uso de transformadas multiescala para análisis y procesamiento de imágenes, extendiendo el enfoque a volúmenes 3D de fase en QSM.

## Contenido

### Parte 1 — Denoising 2D (Starlet vs MMT)
- Transformada Starlet (à trous) con múltiples funciones de escalamiento
- Transformada MMT (Median Multiscale Transform)
- Evaluación con RMSE, SSIM y HaarPSI
- Análisis de configuraciones óptimas

### Parte 2 — Starlet 3D en QSM
- Descomposición multiescala en volúmenes de fase
- Análisis de energía por escala
- Error multiescala respecto a fase local verdadera
- Energía del Laplaciano como indicador de alta frecuencia

### Parte 3 — Aproximación de SMV
- Construcción de operador pasaaltos \( g_J = b - c_J \)
- Comparación con V-SHARP
- Deconvolución regularizada
- Análisis de efectos de borde
- Comparación Starlet vs MMT

---

## Informe

El informe final de la Tarea 3 se encuentra disponible en:

- `assignments/task3/report/Informe_Tarea3.pdf`

---

## Estructura del repositorio

proc-multiesc-imgs/
├── src/course_utils/
│   ├── io_utils.py
│   ├── plot_utils.py
│   ├── metrics.py
│   ├── wavepsi.py
│   ├── noise_utils.py
│   ├── dwt2d.py
│   ├── starlet2d.py
│   ├── starlet3d.py
│   └── mmt2d.py
├── notebooks/
│   ├── task2/Tarea2.ipynb
│   └── task3/
│       ├── T3_Parte1_Denoising_2D.ipynb
│       ├── T3_Parte2_Starlet3D_QSM.ipynb
│       └── T3_Parte3_SMV_QSM.ipynb
├── data/raw/
│   ├── task2/
│   └── task3/
├── results/
│   ├── task2/
│   └── task3/
├── figures/
└── assignments/
    ├── task2/report/
    └── task3/report/

```

---

## Requisitos

- Python ≥ 3.10
- Conda (Miniforge recomendado)

## Instalación de Conda (Miniforge)

Se recomienda utilizar **Miniforge**, una distribución ligera de Conda basada en conda-forge.

### 1. Descargar Miniforge

Descargar desde:

https://github.com/conda-forge/miniforge

Seleccionar la versión correspondiente a tu sistema operativo:

- macOS (Apple Silicon / Intel)
- Linux
- Windows

---

### 2. Instalar Miniforge

En macOS / Linux:

```bash
bash Miniforge3-*.sh
```

---

## Creación del entorno

```bash
conda env create -f environment.yml
conda activate multiscale
```

---

## Ejecución

Después de activar el entorno `multiscale` (o el nombre definido), ejecutar:

```bash
jupyter lab
```

Desde la raíz del repositorio, abrir los notebooks correspondientes y ejecutar todas las celdas:

- **Tarea 2:** `notebooks/task2/Tarea2.ipynb`
- **Tarea 3:**  
  - `notebooks/task3/T3_Parte1_Denoising_2D.ipynb`  
  - `notebooks/task3/T3_Parte2_Starlet3D_QSM.ipynb`  
  - `notebooks/task3/T3_Parte3_SMV_QSM.ipynb`
---

## Reproducibilidad

Los siguientes archivos deben estar en **data/raw/task1/** para la correcta ejecución del notebook:
- `local_vsharp_m4.nii`
- `unwrapped_seguetotalphase.nii`
- `unwrapped_truelocalphase.nii`
- `mask4.nii`

Los siguientes archivos deben estar en **data/raw/task2/** para la correcta ejecución del notebook:

- `gt.nii`
- `mask.nii`
- `streaking.nii`
- `reconNDI1.nii`, `reconNDI2.nii`
- `recontv1.nii`, `recontv2.nii`, `recontv3.nii`
- `Moon.png`
- `EX3_01.png`

---

## Resultados

Los resultados generados durante la ejecución de los notebooks se almacenan en:

- **results/task2/**  
  Contiene métricas cuantitativas en formato CSV y resultados asociados a MRI y denoising 2D.

- **results/task3/**  
  Contiene resultados multiescala, análisis QSM, aproximaciones SMV, figuras generadas y tablas en formato CSV.

- **assignments/task2/report/figures/**  
  Figuras utilizadas en el informe de la Tarea 2.

- **assignments/task3/report/**  
  Informe final de la Tarea 3 (PDF).

- **figures/task3/**  
  Figuras optimizadas para su inclusión en el informe de la Tarea 3.

> Nota: Algunas figuras en `figures/task3/` corresponden a versiones optimizadas para su integración en el informe final.

---


## Referencias

- Mallat (2009) — *A Wavelet Tour of Signal Processing*
- Reisenhofer et al. (2018) — HaarPSI
- Starck et al. (2010) — *Sparse Image and Signal Processing*
- Schweser et al. (2011, 2017) — QSM
- Sun & Wilman (2014) — SMV / V-SHARP

---

## Autor

Pablo Poblete Arrué

---


## 📌 Nota

> Este proyecto fue desarrollado y probado utilizando Miniforge y conda-forge. Se recomienda no usar Anaconda para evitar conflictos de dependencias.