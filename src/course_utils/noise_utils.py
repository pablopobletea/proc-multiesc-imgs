import numpy as np


# --------------------------------------------------
# Ruido Gaussiano
# --------------------------------------------------

def add_gaussian_noise(image: np.ndarray, sigma: float = 0.05):
    """
    Agrega ruido gaussiano (media 0)

    sigma: desviación estándar relativa (asumiendo imagen en [0,1])
    """
    noise = np.random.normal(loc=0.0, scale=sigma, size=image.shape)
    noisy = image + noise
    return np.clip(noisy, 0.0, 1.0)


# --------------------------------------------------
# Ruido Uniforme
# --------------------------------------------------

def add_uniform_noise(image: np.ndarray, low: float = -0.1, high: float = 0.1):
    """
    Agrega ruido uniforme U(low, high)
    """
    noise = np.random.uniform(low, high, size=image.shape)
    noisy = image + noise
    return np.clip(noisy, 0.0, 1.0)


# --------------------------------------------------
# Ruido sal y pimienta (opcional)
# --------------------------------------------------

def add_salt_pepper_noise(image: np.ndarray, prob: float = 0.02):
    """
    Ruido sal y pimienta (no requerido pero útil)
    """
    noisy = image.copy()
    rnd = np.random.rand(*image.shape)

    noisy[rnd < prob / 2] = 0.0
    noisy[rnd > 1 - prob / 2] = 1.0

    return noisy