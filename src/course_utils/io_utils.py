import numpy as np
import nibabel as nib
from PIL import Image


# --------------------------------------------------
# NIfTI (MRI 3D)
# --------------------------------------------------

def load_nifti(path: str):
    """
    Carga un archivo NIfTI (.nii o .nii.gz)

    Returns:
        volume (np.ndarray): volumen 3D
        affine (np.ndarray): matriz affine
        header: header NIfTI
    """
    nii = nib.load(path)
    volume = nii.get_fdata().astype(np.float64)
    return volume, nii.affine, nii.header


def save_nifti(path: str, volume: np.ndarray, affine=None):
    """
    Guarda volumen como NIfTI
    """
    if affine is None:
        affine = np.eye(4)

    nii = nib.Nifti1Image(volume, affine)
    nib.save(nii, path)


# --------------------------------------------------
# Imágenes 2D
# --------------------------------------------------

def load_grayscale_image(path: str, normalize: bool = True):
    """
    Carga imagen 2D en escala de grises

    Returns:
        image (np.ndarray): float64
    """
    img = Image.open(path).convert("L")
    img = np.asarray(img, dtype=np.float64)

    if normalize:
        img = normalize_image(img)

    return img


def save_image(path: str, image: np.ndarray):
    """
    Guarda imagen 2D (auto-normaliza a 0-255)
    """
    img = normalize_image(image)
    img_uint8 = (img * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(path)


# --------------------------------------------------
# Normalización
# --------------------------------------------------

def normalize_image(img: np.ndarray, eps: float = 1e-8):
    """
    Normaliza a rango [0, 1]
    """
    img = np.asarray(img, dtype=np.float64)

    min_val = np.min(img)
    max_val = np.max(img)

    return (img - min_val) / (max_val - min_val + eps)


def normalize_volume(vol: np.ndarray, eps: float = 1e-8):
    """
    Normaliza volumen 3D a [0, 1]
    """
    return normalize_image(vol, eps)


# --------------------------------------------------
# Máscaras
# --------------------------------------------------

def load_mask(path: str, threshold: float = 0.0):
    """
    Carga máscara NIfTI y la convierte a booleana
    """
    mask, _, _ = load_nifti(path)
    return mask > threshold


# --------------------------------------------------
# Utilidades útiles
# --------------------------------------------------

def get_center_slice(volume: np.ndarray, axis: int = 2):
    """
    Devuelve slice central de un volumen
    """
    idx = volume.shape[axis] // 2

    if axis == 0:
        return volume[idx, :, :]
    elif axis == 1:
        return volume[:, idx, :]
    elif axis == 2:
        return volume[:, :, idx]
    else:
        raise ValueError("axis must be 0, 1, or 2")


def apply_mask(volume: np.ndarray, mask: np.ndarray, fill_value=0):
    """
    Aplica máscara a volumen
    """
    return np.where(mask, volume, fill_value)