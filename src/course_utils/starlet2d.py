import numpy as np
from scipy.ndimage import convolve


# -----------------------------
# B3-spline 1D kernel
# -----------------------------
def b3_spline_kernel_1d():
    return np.array([1, 4, 6, 4, 1], dtype=float) / 16.0


# -----------------------------
# Construcción del kernel 2D
# -----------------------------
def b3_spline_kernel_2d():
    h = b3_spline_kernel_1d()
    return np.outer(h, h)


# -----------------------------
# À trous: inserta ceros
# -----------------------------
def atrous_kernel(kernel, level):
    if level == 0:
        return kernel

    step = 2 ** level
    size = kernel.shape[0]
    new_size = size + (size - 1) * (step - 1)

    new_kernel = np.zeros((new_size, new_size))
    for i in range(size):
        for j in range(size):
            new_kernel[i * step, j * step] = kernel[i, j]

    return new_kernel


# -----------------------------
# Starlet decomposition
# -----------------------------
def starlet_transform(image, n_scales=5):
    """
    Returns:
        coeffs: list of detail coefficients [w1, w2, ..., wJ]
        cJ: coarse residual
    """
    image = image.astype(float)
    h = b3_spline_kernel_2d()

    c = image.copy()
    coeffs = []

    for j in range(n_scales):
        hj = atrous_kernel(h, j)
        c_smooth = convolve(c, hj, mode='mirror')

        w = c - c_smooth
        coeffs.append(w)

        c = c_smooth

    return coeffs, c


# -----------------------------
# Reconstruction
# -----------------------------
def starlet_reconstruction(coeffs, cJ):
    rec = cJ.copy()
    for w in coeffs:
        rec += w
    return rec