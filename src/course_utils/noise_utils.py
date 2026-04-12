import numpy as np


# --------------------------------------------------
# Gaussian Noise
# --------------------------------------------------

def add_gaussian_noise(image: np.ndarray, sigma: float = 0.05, rng=None):
    """
    Add zero-mean Gaussian noise to an image.

    Parameters
    ----------
    image : np.ndarray
        Input image in range [0, 1].
    sigma : float
        Standard deviation of the noise.
    rng : np.random.Generator or None
        Random number generator for reproducibility.

    Returns
    -------
    noisy : np.ndarray
        Noisy image clipped to [0, 1].
    """
    if rng is None:
        rng = np.random

    noise = rng.normal(loc=0.0, scale=sigma, size=image.shape)
    noisy = image + noise
    return np.clip(noisy, 0.0, 1.0)

# --------------------------------------------------
# Uniform Niose
# --------------------------------------------------

def add_uniform_noise(image: np.ndarray, low: float = -0.1, high: float = 0.1, rng=None):
    """
    Add uniform noise U(low, high) to an image.
    """
    if rng is None:
        rng = np.random

    noise = rng.uniform(low, high, size=image.shape)
    noisy = image + noise
    return np.clip(noisy, 0.0, 1.0)


# --------------------------------------------------
# Salt & Pepper Noise
# --------------------------------------------------

def add_salt_pepper_noise(image: np.ndarray, prob: float = 0.02, rng=None):
    """
    Add salt and pepper noise to an image.
    """
    if rng is None:
        rng = np.random

    noisy = image.copy()
    rnd = rng.random(image.shape)

    noisy[rnd < prob / 2] = 0.0
    noisy[rnd > 1 - prob / 2] = 1.0

    return noisy