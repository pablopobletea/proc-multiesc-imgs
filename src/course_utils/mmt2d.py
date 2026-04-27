import numpy as np
from scipy.ndimage import median_filter
from skimage.morphology import diamond, disk


def get_kernel_footprint(kernel_type="square", radius=1):
    """
    Create a neighborhood footprint for the Multiscale Median Transform.

    Parameters
    ----------
    kernel_type : str
        One of {"square", "diamond", "circle"}.
    radius : int
        Neighborhood radius.

    Returns
    -------
    footprint : np.ndarray
        Boolean footprint used by scipy.ndimage.median_filter.
    """
    if radius < 1:
        raise ValueError("radius must be >= 1")

    kernel_type = kernel_type.lower()

    if kernel_type == "square":
        size = 2 * radius + 1
        return np.ones((size, size), dtype=bool)

    if kernel_type == "diamond":
        return diamond(radius).astype(bool)

    if kernel_type in {"circle", "disk"}:
        return disk(radius).astype(bool)

    raise ValueError("kernel_type must be one of: square, diamond, circle")


def mmt_transform(image, n_scales=5, kernel_type="square", base_radius=1):
    """
    Multiscale Median Transform decomposition.

    The approximation at each scale is obtained by median filtering
    the previous approximation with an increasing neighborhood size.

    Parameters
    ----------
    image : np.ndarray
        Input 2D image.
    n_scales : int
        Number of decomposition levels.
    kernel_type : str
        Neighborhood type: {"square", "diamond", "circle"}.
    base_radius : int
        Initial radius. At scale j, radius = base_radius * 2**j.

    Returns
    -------
    details : list[np.ndarray]
        Detail images [w1, w2, ..., wJ].
    cJ : np.ndarray
        Coarse residual.
    approximations : list[np.ndarray]
        Approximations [c0, c1, ..., cJ].
    """
    if n_scales < 1:
        raise ValueError("n_scales must be >= 1")

    image = np.asarray(image, dtype=float)

    c = image.copy()
    details = []
    approximations = [c.copy()]

    for j in range(n_scales):
        radius = base_radius * (2 ** j)
        footprint = get_kernel_footprint(kernel_type=kernel_type, radius=radius)

        c_smooth = median_filter(c, footprint=footprint, mode="reflect")

        w = c - c_smooth
        details.append(w)

        c = c_smooth
        approximations.append(c.copy())

    return details, c, approximations


def mmt_reconstruction(details, cJ):
    """
    Reconstruct image from MMT detail coefficients and coarse residual.
    """
    rec = np.asarray(cJ, dtype=float).copy()

    for w in details:
        rec += w

    return rec


def average_mmt_decomposition(image, n_scales=5, kernel_types=("square", "diamond"), base_radius=1):
    """
    Bonus helper: average MMT decompositions obtained with different kernels.

    This is useful for the optional task asking to average square and
    diamond-kernel decompositions.

    Parameters
    ----------
    image : np.ndarray
        Input 2D image.
    n_scales : int
        Number of scales.
    kernel_types : tuple[str]
        Kernels to average.
    base_radius : int
        Initial radius.

    Returns
    -------
    avg_details : list[np.ndarray]
        Averaged detail coefficients.
    avg_cJ : np.ndarray
        Averaged coarse residual.
    """
    all_details = []
    all_cJ = []

    for kernel_type in kernel_types:
        details, cJ, _ = mmt_transform(
            image,
            n_scales=n_scales,
            kernel_type=kernel_type,
            base_radius=base_radius,
        )
        all_details.append(details)
        all_cJ.append(cJ)

    avg_details = []
    for j in range(n_scales):
        avg_details.append(np.mean([details[j] for details in all_details], axis=0))

    avg_cJ = np.mean(all_cJ, axis=0)

    return avg_details, avg_cJ