import numpy as np
import pywt


# --------------------------------------------------
# Función de similitud tipo SSIM
# --------------------------------------------------

def similarity_map(a, b, c=1e-6):
    """
    Similaridad tipo SSIM simplificada
    """
    return (2 * a * b + c) / (a**2 + b**2 + c)


# --------------------------------------------------
# WavePsi 2D
# --------------------------------------------------

def wavepsi_2d(img1, img2, wavelet="haar", level=3):
    """
    Calcula una métrica tipo HaarPSI usando wavelets generales
    """
    coeffs1 = pywt.wavedec2(img1, wavelet=wavelet, level=level)
    coeffs2 = pywt.wavedec2(img2, wavelet=wavelet, level=level)

    sim_maps = []
    weights = []

    # Ignoramos cA (aproximación) → usamos detalles
    for (cH1, cV1, cD1), (cH2, cV2, cD2) in zip(coeffs1[1:], coeffs2[1:]):

        # similitud por dirección
        sim_H = similarity_map(cH1, cH2)
        sim_V = similarity_map(cV1, cV2)
        sim_D = similarity_map(cD1, cD2)

        sim = (sim_H + sim_V + sim_D) / 3.0

        # peso basado en energía (importante)
        weight = np.abs(cH1) + np.abs(cV1) + np.abs(cD1)

        sim_maps.append(sim)
        weights.append(weight)

    # Agregación global
    numerator = sum(np.sum(w * s) for w, s in zip(weights, sim_maps))
    denominator = sum(np.sum(w) for w in weights) + 1e-8

    return float(numerator / denominator)


# --------------------------------------------------
# WavePsi 3D (slice-wise)
# --------------------------------------------------

def wavepsi_3d_slicewise(vol1, vol2, mask=None, wavelet="haar", level=3, axis=2):
    """
    Aplica WavePsi 2D slice-wise sobre un volumen 3D
    """
    if vol1.shape != vol2.shape:
        raise ValueError("Volumes must have same shape")

    n_slices = vol1.shape[axis]
    scores = []

    for i in range(n_slices):

        if axis == 0:
            s1 = vol1[i, :, :]
            s2 = vol2[i, :, :]
            sm = mask[i, :, :] if mask is not None else None

        elif axis == 1:
            s1 = vol1[:, i, :]
            s2 = vol2[:, i, :]
            sm = mask[:, i, :] if mask is not None else None

        else:
            s1 = vol1[:, :, i]
            s2 = vol2[:, :, i]
            sm = mask[:, :, i] if mask is not None else None

        # ignorar slices vacíos
        if sm is not None and np.count_nonzero(sm) < 10:
            continue

        scores.append(wavepsi_2d(s1, s2, wavelet=wavelet, level=level))

    if len(scores) == 0:
        raise ValueError("No valid slices for WavePsi")

    return float(np.mean(scores))


# --------------------------------------------------
# Evaluación con múltiples wavelets
# --------------------------------------------------

def evaluate_wavepsi_family(vol1, vol2, mask=None, wavelets=None):
    """
    Calcula WavePsi para múltiples wavelets
    """
    if wavelets is None:
        wavelets = ["haar", "db2", "sym4", "coif1"]

    results = {}

    for w in wavelets:
        results[f"wavepsi_{w}"] = wavepsi_3d_slicewise(
            vol1, vol2, mask=mask, wavelet=w
        )

    return results