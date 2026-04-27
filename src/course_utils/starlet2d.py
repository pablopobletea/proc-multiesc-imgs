import numpy as np
from scipy.ndimage import convolve


def scaling_kernel_1d(kind="b3spline"):
    kind = kind.lower()

    if kind in {"b3", "b3spline", "b3-spline"}:
        return np.array([1, 4, 6, 4, 1], dtype=float) / 16.0

    if kind in {"linear", "triangle", "binomial3"}:
        return np.array([1, 2, 1], dtype=float) / 4.0

    raise ValueError("kind must be one of: b3spline, linear")


def scaling_kernel_2d(kind="b3spline"):
    h = scaling_kernel_1d(kind)
    return np.outer(h, h)


def atrous_kernel(kernel, level):
    if level == 0:
        return kernel

    step = 2 ** level
    size = kernel.shape[0]
    new_size = size + (size - 1) * (step - 1)

    new_kernel = np.zeros((new_size, new_size), dtype=float)

    for i in range(size):
        for j in range(size):
            new_kernel[i * step, j * step] = kernel[i, j]

    return new_kernel


def starlet_transform(image, n_scales=5, scaling="b3spline", mode="mirror"):
    image = np.asarray(image, dtype=float)

    h = scaling_kernel_2d(scaling)

    c = image.copy()
    details = []
    approximations = [c.copy()]

    for j in range(n_scales):
        hj = atrous_kernel(h, j)
        c_smooth = convolve(c, hj, mode=mode)

        w = c - c_smooth
        details.append(w)

        c = c_smooth
        approximations.append(c.copy())

    return details, c, approximations


def starlet_reconstruction(details, cJ):
    rec = np.asarray(cJ, dtype=float).copy()

    for w in details:
        rec += w

    return rec


def hard_threshold(x, threshold):
    return x * (np.abs(x) >= threshold)


def soft_threshold(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def threshold_details(details, threshold, mode="soft"):
    if mode == "soft":
        fn = soft_threshold
    elif mode == "hard":
        fn = hard_threshold
    else:
        raise ValueError("mode must be 'soft' or 'hard'")

    return [fn(w, threshold) for w in details]


def sigma_mad(x):
    return np.median(np.abs(x - np.median(x))) / 0.6745


def denoise_starlet(
    image,
    n_scales=5,
    scaling="b3spline",
    threshold_mode="soft",
    threshold_factor=3.0,
):
    details, cJ, approximations = starlet_transform(
        image,
        n_scales=n_scales,
        scaling=scaling,
    )

    sigma = sigma_mad(details[0])
    denoised_details = [
        soft_threshold(w, threshold_factor * sigma)
        if threshold_mode == "soft"
        else hard_threshold(w, threshold_factor * sigma)
        for w in details
    ]

    rec = starlet_reconstruction(denoised_details, cJ)

    return rec, denoised_details, cJ, approximations