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

## Contenido del proyecto

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


## Estructura del repositorio


```text
proc-multiesc-imgs/
├── src/course_utils/
│   ├── io_utils.py
│   ├── plot_utils.py
│   ├── metrics.py
│   ├── wavepsi.py
│   ├── noise_utils.py
│   └── dwt2d.py
├── notebooks/task2/Tarea2.ipynb
├── assignments/task2/report/
├── data/raw/task2/
└── results/task2/
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

Desde la raíz del repositorio, abrir: notebooks/task2/Tarea2.ipynb y ejecutar todas las celdas.

---

## Reproducibilidad

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

Los resultados generados durante la ejecución del notebook se almacenan en:

- **results/task2/**  
  Contiene las métricas en formato CSV.

- **assignments/task2/report/figures/**  
  Contiene las figuras utilizadas en el informe.

---


## Referencias

- Reisenhofer et al. (2018) — HaarPSI
- Mallat (2009) — Wavelet Tour of Signal Processing

---

## Autor

Pablo Poblete Arrué

---


## 📌 Nota

> Este proyecto fue desarrollado y probado utilizando Miniforge y conda-forge. Se recomienda no usar Anaconda para evitar conflictos de dependencias.