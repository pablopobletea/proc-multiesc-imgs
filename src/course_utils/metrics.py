import numpy as np
from skimage.metrics import structural_similarity as ssim


# --------------------------------------------------
# Utilidades internas
# --------------------------------------------------

def _validate_same_shape(a: np.ndarray, b: np.ndarray) -> None:
    """
    Verifica que dos arreglos tengan la misma forma.
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")


def _to_float_array(x: np.ndarray) -> np.ndarray:
    """
    Convierte a float64 para evitar problemas numéricos.
    """
    return np.asarray(x, dtype=np.float64)


def _prepare_mask(mask: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Convierte una máscara a booleana y valida forma.
    """
    mask_bool = np.asarray(mask) > 0
    if mask_bool.shape != shape:
        raise ValueError(f"Mask shape mismatch: {mask_bool.shape} vs {shape}")
    return mask_bool


def _safe_data_range(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calcula un rango de datos robusto para SSIM.
    """
    data_min = min(np.min(a), np.min(b))
    data_max = max(np.max(a), np.max(b))
    data_range = float(data_max - data_min)

    if data_range == 0:
        data_range = 1.0

    return data_range


# --------------------------------------------------
# Métricas básicas
# --------------------------------------------------

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """
    Root Mean Squared Error global.
    """
    _validate_same_shape(a, b)
    a = _to_float_array(a)
    b = _to_float_array(b)

    return float(np.sqrt(np.mean((a - b) ** 2)))


def masked_rmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """
    RMSE calculado solo dentro de la máscara.
    """
    _validate_same_shape(a, b)
    a = _to_float_array(a)
    b = _to_float_array(b)
    mask = _prepare_mask(mask, a.shape)

    if np.count_nonzero(mask) == 0:
        raise ValueError("The mask contains no valid voxels/pixels.")

    diff = a[mask] - b[mask]
    return float(np.sqrt(np.mean(diff ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    """
    Mean Absolute Error global.
    """
    _validate_same_shape(a, b)
    a = _to_float_array(a)
    b = _to_float_array(b)

    return float(np.mean(np.abs(a - b)))


def masked_mae(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """
    MAE calculado solo dentro de la máscara.
    """
    _validate_same_shape(a, b)
    a = _to_float_array(a)
    b = _to_float_array(b)
    mask = _prepare_mask(mask, a.shape)

    if np.count_nonzero(mask) == 0:
        raise ValueError("The mask contains no valid voxels/pixels.")

    return float(np.mean(np.abs(a[mask] - b[mask])))


# --------------------------------------------------
# SSIM 2D
# --------------------------------------------------

def ssim_2d(a: np.ndarray, b: np.ndarray) -> float:
    """
    SSIM entre dos imágenes 2D.
    """
    _validate_same_shape(a, b)

    if a.ndim != 2:
        raise ValueError(f"ssim_2d expects 2D arrays, got ndim={a.ndim}")

    a = _to_float_array(a)
    b = _to_float_array(b)

    data_range = _safe_data_range(a, b)
    return float(ssim(a, b, data_range=data_range))


def masked_ssim_2d(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """
    SSIM 2D aproximado usando bounding box de la máscara.
    Esto evita evaluar fondo irrelevante cuando hay mucho cero.
    """
    _validate_same_shape(a, b)

    if a.ndim != 2:
        raise ValueError(f"masked_ssim_2d expects 2D arrays, got ndim={a.ndim}")

    a = _to_float_array(a)
    b = _to_float_array(b)
    mask = _prepare_mask(mask, a.shape)

    if np.count_nonzero(mask) == 0:
        raise ValueError("The mask contains no valid pixels.")

    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1

    a_crop = a[r0:r1, c0:c1]
    b_crop = b[r0:r1, c0:c1]

    data_range = _safe_data_range(a_crop, b_crop)
    return float(ssim(a_crop, b_crop, data_range=data_range))


# --------------------------------------------------
# SSIM 3D por slices
# --------------------------------------------------

def volume_ssim_slicewise(
    a: np.ndarray,
    b: np.ndarray,
    axis: int = 2
) -> float:
    """
    Calcula SSIM promedio slice-wise sobre un volumen 3D.

    axis=2 -> slices axiales
    axis=1 -> coronales
    axis=0 -> sagitales
    """
    _validate_same_shape(a, b)

    if a.ndim != 3:
        raise ValueError(f"volume_ssim_slicewise expects 3D arrays, got ndim={a.ndim}")

    a = _to_float_array(a)
    b = _to_float_array(b)

    n_slices = a.shape[axis]
    scores = []

    for i in range(n_slices):
        if axis == 0:
            sa = a[i, :, :]
            sb = b[i, :, :]
        elif axis == 1:
            sa = a[:, i, :]
            sb = b[:, i, :]
        elif axis == 2:
            sa = a[:, :, i]
            sb = b[:, :, i]
        else:
            raise ValueError("axis must be 0, 1, or 2")

        # Evita slices completamente constantes en ambos
        if np.allclose(sa, sa.flat[0]) and np.allclose(sb, sb.flat[0]):
            continue

        scores.append(ssim_2d(sa, sb))

    if len(scores) == 0:
        raise ValueError("No valid slices found for SSIM computation.")

    return float(np.mean(scores))


def volume_ssim_slicewise_masked(
    a: np.ndarray,
    b: np.ndarray,
    mask: np.ndarray,
    axis: int = 2,
    min_mask_pixels: int = 16
) -> float:
    """
    Calcula SSIM promedio slice-wise usando una máscara 3D.
    Solo considera slices con suficiente contenido en la máscara.
    """
    _validate_same_shape(a, b)

    if a.ndim != 3:
        raise ValueError(f"volume_ssim_slicewise_masked expects 3D arrays, got ndim={a.ndim}")

    a = _to_float_array(a)
    b = _to_float_array(b)
    mask = _prepare_mask(mask, a.shape)

    n_slices = a.shape[axis]
    scores = []

    for i in range(n_slices):
        if axis == 0:
            sa = a[i, :, :]
            sb = b[i, :, :]
            sm = mask[i, :, :]
        elif axis == 1:
            sa = a[:, i, :]
            sb = b[:, i, :]
            sm = mask[:, i, :]
        elif axis == 2:
            sa = a[:, :, i]
            sb = b[:, :, i]
            sm = mask[:, :, i]
        else:
            raise ValueError("axis must be 0, 1, or 2")

        if np.count_nonzero(sm) < min_mask_pixels:
            continue

        scores.append(masked_ssim_2d(sa, sb, sm))

    if len(scores) == 0:
        raise ValueError("No valid masked slices found for SSIM computation.")

    return float(np.mean(scores))


# --------------------------------------------------
# Evaluación compacta
# --------------------------------------------------

def evaluate_pair_2d(
    reference: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None
) -> dict:
    """
    Devuelve métricas básicas para una pareja de imágenes 2D.
    """
    metrics = {
        "rmse": rmse(reference, target),
        "mae": mae(reference, target),
        "ssim": ssim_2d(reference, target),
    }

    if mask is not None:
        metrics["rmse_masked"] = masked_rmse(reference, target, mask)
        metrics["mae_masked"] = masked_mae(reference, target, mask)
        metrics["ssim_masked"] = masked_ssim_2d(reference, target, mask)

    return metrics


def evaluate_pair_3d(
    reference: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
    axis: int = 2
) -> dict:
    """
    Devuelve métricas básicas para una pareja de volúmenes 3D.
    """
    metrics = {
        "rmse": rmse(reference, target),
        "mae": mae(reference, target),
        "ssim_slicewise": volume_ssim_slicewise(reference, target, axis=axis),
    }

    if mask is not None:
        metrics["rmse_masked"] = masked_rmse(reference, target, mask)
        metrics["mae_masked"] = masked_mae(reference, target, mask)
        metrics["ssim_slicewise_masked"] = volume_ssim_slicewise_masked(
            reference, target, mask, axis=axis
        )

    return metrics