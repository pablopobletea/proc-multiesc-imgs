import os
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# Utilidad de guardado
# --------------------------------------------------

def save_figure(fig, path: str, dpi: int = 300, close: bool = False):
    """
    Guarda una figura de matplotlib en disco.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figura a guardar.
    path : str
        Ruta de salida.
    dpi : int
        Resolución de exportación.
    close : bool
        Si True, cierra la figura después de guardarla.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}")

    if close:
        plt.close(fig)


# --------------------------------------------------
# Función interna para extraer slices
# --------------------------------------------------

def _get_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    """
    Extrae un slice 2D desde un volumen 3D.
    """
    if axis == 0:
        return volume[index, :, :]
    if axis == 1:
        return volume[:, index, :]
    if axis == 2:
        return volume[:, :, index]
    raise ValueError("axis must be 0, 1, or 2")


# --------------------------------------------------
# 1. Cortes ortogonales
# --------------------------------------------------

def show_orthogonal_slices(
    volume: np.ndarray,
    title: str = "",
    cmap: str = "gray",
    figsize=(12, 4)
):
    """
    Muestra los cortes centrales sagital, coronal y axial.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    x = volume.shape[0] // 2
    y = volume.shape[1] // 2
    z = volume.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(volume[x, :, :], cmap=cmap)
    axes[0].set_title("Sagittal")

    axes[1].imshow(volume[:, y, :], cmap=cmap)
    axes[1].set_title("Coronal")

    axes[2].imshow(volume[:, :, z], cmap=cmap)
    axes[2].set_title("Axial")

    for ax in axes:
        ax.axis("off")

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    return fig


# --------------------------------------------------
# 2. Comparación lado a lado (volúmenes 3D)
# --------------------------------------------------

def compare_slices(
    volumes,
    titles,
    axis: int = 2,
    index: int | None = None,
    cmap: str = "gray",
    figsize=None,
):
    """
    Compara múltiples volúmenes 3D usando un mismo slice.

    Parameters
    ----------
    volumes : list[np.ndarray]
        Lista de volúmenes 3D.
    titles : list[str]
        Títulos para cada panel.
    axis : int
        Eje del corte (0, 1, 2).
    index : int or None
        Índice del corte. Si es None, usa el central.
    cmap : str
        Colormap.
    figsize : tuple or None
        Tamaño de la figura.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if len(volumes) != len(titles):
        raise ValueError("volumes and titles must have the same length")

    if len(volumes) == 0:
        raise ValueError("volumes must not be empty")

    if index is None:
        index = volumes[0].shape[axis] // 2

    n = len(volumes)
    if figsize is None:
        figsize = (4 * n, 4)

    fig, axes = plt.subplots(1, n, figsize=figsize)

    if n == 1:
        axes = [axes]

    for ax, vol, ttl in zip(axes, volumes, titles):
        img = _get_slice(vol, axis=axis, index=index)
        ax.imshow(img, cmap=cmap)
        ax.set_title(ttl)
        ax.axis("off")

    fig.tight_layout()
    return fig


# --------------------------------------------------
# 3. Comparación lado a lado (imágenes 2D)
# --------------------------------------------------

def compare_images(
    images,
    titles,
    cmap: str = "gray",
    figsize=None,
):
    """
    Compara múltiples imágenes 2D.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if len(images) != len(titles):
        raise ValueError("images and titles must have the same length")

    if len(images) == 0:
        raise ValueError("images must not be empty")

    n = len(images)
    if figsize is None:
        figsize = (4 * n, 4)

    fig, axes = plt.subplots(1, n, figsize=figsize)

    if n == 1:
        axes = [axes]

    for ax, img, ttl in zip(axes, images, titles):
        ax.imshow(img, cmap=cmap)
        ax.set_title(ttl)
        ax.axis("off")

    fig.tight_layout()
    return fig


# --------------------------------------------------
# 4. Grid de múltiples slices
# --------------------------------------------------

def show_slice_grid(
    volume: np.ndarray,
    axis: int = 2,
    step: int = 10,
    start: int | None = None,
    stop: int | None = None,
    cmap: str = "gray",
    cols: int = 5,
    figsize=None,
):
    """
    Muestra múltiples slices de un volumen en una grilla.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n_slices = volume.shape[axis]

    if start is None:
        start = 0
    if stop is None:
        stop = n_slices

    indices = list(range(start, stop, step))
    if len(indices) == 0:
        raise ValueError("No slices selected. Check start/stop/step values.")

    n = len(indices)
    rows = int(np.ceil(n / cols))

    if figsize is None:
        figsize = (3 * cols, 3 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    for i, idx in enumerate(indices):
        img = _get_slice(volume, axis=axis, index=idx)
        axes[i].imshow(img, cmap=cmap)
        axes[i].set_title(f"Slice {idx}")
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    return fig


# --------------------------------------------------
# 5. Error map simple
# --------------------------------------------------

def show_error_map(
    gt: np.ndarray,
    recon: np.ndarray,
    axis: int = 2,
    index: int | None = None,
    img_cmap: str = "gray",
    err_cmap: str = "hot",
    figsize=(12, 4),
):
    """
    Muestra GT, reconstrucción y mapa de error absoluto.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if gt.shape != recon.shape:
        raise ValueError("gt and recon must have the same shape")

    if index is None:
        index = gt.shape[axis] // 2

    g = _get_slice(gt, axis=axis, index=index)
    r = _get_slice(recon, axis=axis, index=index)
    e = np.abs(g - r)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(g, cmap=img_cmap)
    axes[0].set_title("GT")
    axes[0].axis("off")

    axes[1].imshow(r, cmap=img_cmap)
    axes[1].set_title("Recon")
    axes[1].axis("off")

    im = axes[2].imshow(e, cmap=err_cmap)
    axes[2].set_title("Error")
    axes[2].axis("off")

    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


# --------------------------------------------------
# 6. Grid de mapas de error
# --------------------------------------------------

def compare_error_maps(
    gt: np.ndarray,
    reconstructions,
    titles,
    axis: int = 2,
    index: int | None = None,
    err_cmap: str = "hot",
    figsize=None,
):
    """
    Compara varios mapas de error absoluto respecto a GT.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if len(reconstructions) != len(titles):
        raise ValueError("reconstructions and titles must have the same length")

    if index is None:
        index = gt.shape[axis] // 2

    n = len(reconstructions)
    if figsize is None:
        figsize = (4 * n, 4)

    g = _get_slice(gt, axis=axis, index=index)

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    last_im = None
    for ax, recon, ttl in zip(axes, reconstructions, titles):
        if gt.shape != recon.shape:
            raise ValueError("All reconstructions must match gt shape")

        r = _get_slice(recon, axis=axis, index=index)
        e = np.abs(g - r)

        last_im = ax.imshow(e, cmap=err_cmap)
        ax.set_title(ttl)
        ax.axis("off")

    if last_im is not None:
        fig.colorbar(last_im, ax=axes, fraction=0.02, pad=0.02)

    #fig.tight_layout()
    #plt.tight_layout(rect=[0, 0, 0.95, 1])
    fig.subplots_adjust(wspace=0.1)
    return fig


# --------------------------------------------------
# 7. Perfil de intensidad
# --------------------------------------------------

def plot_intensity_profile(
    gt: np.ndarray,
    recon: np.ndarray,
    axis: int = 0,
    index: int | None = None,
    slice_idx: int | None = None,
    figsize=(8, 4),
):
    """
    Grafica un perfil 1D de intensidad para comparar GT y reconstrucción.

    Para volúmenes 3D, se toma un slice y luego una línea dentro de ese slice.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if gt.shape != recon.shape:
        raise ValueError("gt and recon must have the same shape")

    if gt.ndim != 3:
        raise ValueError("plot_intensity_profile expects 3D volumes")

    if slice_idx is None:
        slice_idx = gt.shape[2] // 2

    if index is None:
        index = gt.shape[axis] // 2

    if axis == 0:
        gt_profile = gt[index, :, slice_idx]
        recon_profile = recon[index, :, slice_idx]
    elif axis == 1:
        gt_profile = gt[:, index, slice_idx]
        recon_profile = recon[:, index, slice_idx]
    elif axis == 2:
        gt_profile = gt[:, :, index].ravel()
        recon_profile = recon[:, :, index].ravel()
    else:
        raise ValueError("axis must be 0, 1, or 2")

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(gt_profile, label="GT")
    ax.plot(recon_profile, label="Recon", linestyle="--")
    ax.set_title("Intensity Profile")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    return fig


# --------------------------------------------------
# 8. MIP
# --------------------------------------------------

def show_mip(
    volume: np.ndarray,
    axis: int = 2,
    cmap: str = "gray",
    figsize=(5, 5),
):
    """
    Muestra Maximum Intensity Projection.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    mip = np.max(volume, axis=axis)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(mip, cmap=cmap)
    ax.set_title("MIP")
    ax.axis("off")

    fig.tight_layout()
    return fig