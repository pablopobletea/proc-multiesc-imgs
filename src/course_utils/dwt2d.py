import numpy as np
import pywt


# --------------------------------------------------
# Thresholding
# --------------------------------------------------

def hard_threshold(coeffs, threshold):
    return np.where(np.abs(coeffs) > threshold, coeffs, 0)


def soft_threshold(coeffs, threshold):
    return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0)


# --------------------------------------------------
# Aplicar threshold a coeficientes wavelet
# --------------------------------------------------

def threshold_wavelet_coeffs(coeffs, threshold, mode="soft"):
    """
    Aplica threshold a coeficientes DWT 2D
    """
    cA, detail_coeffs = coeffs[0], coeffs[1:]

    new_coeffs = [cA]

    for (cH, cV, cD) in detail_coeffs:

        if mode == "soft":
            cH = soft_threshold(cH, threshold)
            cV = soft_threshold(cV, threshold)
            cD = soft_threshold(cD, threshold)

        elif mode == "hard":
            cH = hard_threshold(cH, threshold)
            cV = hard_threshold(cV, threshold)
            cD = hard_threshold(cD, threshold)

        else:
            raise ValueError("mode must be 'soft' or 'hard'")

        new_coeffs.append((cH, cV, cD))

    return new_coeffs


# --------------------------------------------------
# Denoising completo
# --------------------------------------------------

def dwt_denoise_2d(
    image: np.ndarray,
    wavelet: str = "haar",
    level: int = 3,
    threshold: float = 0.05,
    mode: str = "soft"
):
    """
    Pipeline completo de denoising con DWT
    """

    # Descomposición
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)

    # Threshold
    coeffs_thresh = threshold_wavelet_coeffs(coeffs, threshold, mode)

    # Reconstrucción
    recon = pywt.waverec2(coeffs_thresh, wavelet=wavelet)

    # Ajustar tamaño (puede cambiar ligeramente)
    recon = recon[:image.shape[0], :image.shape[1]]

    # Clip final
    return np.clip(recon, 0.0, 1.0)


# --------------------------------------------------
# Evaluación múltiple
# --------------------------------------------------

def evaluate_dwt_configurations(
    image,
    noisy,
    wavelets=("haar", "db2"),
    levels=(3,),
    thresholds=(0.02, 0.05, 0.1),
    modes=("soft", "hard"),
):
    """
    Genera múltiples combinaciones para análisis
    """
    results = []

    for w in wavelets:
        for lvl in levels:
            for t in thresholds:
                for m in modes:

                    recon = dwt_denoise_2d(
                        noisy,
                        wavelet=w,
                        level=lvl,
                        threshold=t,
                        mode=m,
                    )

                    results.append({
                        "wavelet": w,
                        "level": lvl,
                        "threshold": t,
                        "mode": m,
                        "reconstruction": recon
                    })

    return results