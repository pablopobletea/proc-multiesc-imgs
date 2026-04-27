import numpy as np
from scipy.ndimage import convolve


def kernel_1d(kind="b3spline"):
    kind = kind.lower()

    if kind in {"b3", "b3spline", "b3-spline"}:
        return np.array([1, 4, 6, 4, 1], dtype=float) / 16.0

    if kind in {"binomial3", "binom3"}:
        return np.array([1, 2, 1], dtype=float) / 4.0

    if kind in {"binomial5", "binom5"}:
        return np.array([1, 4, 6, 4, 1], dtype=float) / 16.0

    raise ValueError("kind must be one of: b3spline, binomial3, binomial5")


def atrous_kernel_3d(kind="b3spline", level=0):
    h = kernel_1d(kind)

    k = (
        h[:, None, None]
        * h[None, :, None]
        * h[None, None, :]
    )

    if level == 0:
        return k

    step = 2 ** level
    size = k.shape[0]
    new_size = size + (size - 1) * (step - 1)

    out = np.zeros((new_size, new_size, new_size), dtype=float)

    for i in range(size):
        for j in range(size):
            for z in range(size):
                out[i * step, j * step, z * step] = k[i, j, z]

    return out


def starlet_transform_3d(volume, n_scales=5, kernel="b3spline", mode="mirror"):
    """
    3D Starlet decomposition.

    Parameters
    ----------
    volume : np.ndarray
        Input 3D volume.
    n_scales : int
        Number of decomposition scales.
    kernel : str
        Smoothing kernel: {"b3spline", "binomial3", "binomial5"}.
    mode : str
        Boundary mode used by scipy.ndimage.convolve.

    Returns
    -------
    details : list[np.ndarray]
        Detail coefficients [w1, ..., wJ].
    cJ : np.ndarray
        Coarse residual.
    approximations : list[np.ndarray]
        Approximations [c0, c1, ..., cJ].
    """
    if n_scales < 1:
        raise ValueError("n_scales must be >= 1")

    volume = np.asarray(volume, dtype=float)

    if volume.ndim != 3:
        raise ValueError("volume must be a 3D array")

    c = volume.copy()
    details = []
    approximations = [c.copy()]

    for level in range(n_scales):
        h = atrous_kernel_3d(kind=kernel, level=level)
        c_next = convolve(c, h, mode=mode)

        w = c - c_next
        details.append(w)

        c = c_next
        approximations.append(c.copy())

    return details, c, approximations


def starlet_reconstruct_3d(details, cJ):
    """
    Reconstruct a 3D volume from Starlet detail coefficients and residual.
    """
    rec = np.asarray(cJ, dtype=float).copy()

    for w in details:
        rec += np.asarray(w, dtype=float)

    return rec


def partial_reconstruction(details, cJ=None, scales=None, include_residual=False):
    """
    Reconstruct selected scales.

    Parameters
    ----------
    details : list[np.ndarray]
        Detail coefficients.
    cJ : np.ndarray or None
        Coarse residual.
    scales : list[int] or None
        1-based scales to include. Example: [1, 2, 3].
        If None, all detail scales are included.
    include_residual : bool
        Whether to include cJ.

    Returns
    -------
    rec : np.ndarray
        Partial reconstruction.
    """
    if scales is None:
        selected = range(len(details))
    else:
        selected = [s - 1 for s in scales]

    rec = np.zeros_like(details[0], dtype=float)

    for idx in selected:
        rec += details[idx]

    if include_residual:
        if cJ is None:
            raise ValueError("cJ must be provided when include_residual=True")
        rec += cJ

    return rec


def scale_energy(details, mask=None, relative=True):
    """
    Compute energy per Starlet scale.

    Parameters
    ----------
    details : list[np.ndarray]
        Starlet detail coefficients.
    mask : np.ndarray or None
        Optional binary mask.
    relative : bool
        If True, normalize energies by total energy.

    Returns
    -------
    energies : np.ndarray
        Energy per scale.
    """
    energies = []

    for w in details:
        if mask is not None:
            values = w[mask > 0]
        else:
            values = w.ravel()

        energies.append(np.sum(values**2))

    energies = np.asarray(energies, dtype=float)

    if relative:
        total = energies.sum()
        if total > 0:
            energies = energies / total

    return energies