import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Mostrar cortes ortogonales
# --------------------------------------------------

def show_orthogonal_slices(volume, title="", cmap="gray"):
    x = volume.shape[0] // 2
    y = volume.shape[1] // 2
    z = volume.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(volume[x, :, :], cmap=cmap)
    axes[0].set_title("Sagittal")

    axes[1].imshow(volume[:, y, :], cmap=cmap)
    axes[1].set_title("Coronal")

    axes[2].imshow(volume[:, :, z], cmap=cmap)
    axes[2].set_title("Axial")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# 2. Comparación lado a lado (slice)
# --------------------------------------------------

def compare_slices(volumes, titles, z=None, cmap="gray"):
    n = len(volumes)

    if z is None:
        z = volumes[0].shape[2] // 2

    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))

    for i in range(n):
        axes[i].imshow(volumes[i][:, :, z], cmap=cmap)
        axes[i].set_title(titles[i])
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# 3. Grid de múltiples slices
# --------------------------------------------------

def show_slice_grid(volume, axis=2, step=10, cmap="gray"):
    indices = range(0, volume.shape[axis], step)
    n = len(indices)

    cols = 5
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))

    axes = axes.flatten()

    for i, idx in enumerate(indices):
        if axis == 0:
            img = volume[idx, :, :]
        elif axis == 1:
            img = volume[:, idx, :]
        else:
            img = volume[:, :, idx]

        axes[i].imshow(img, cmap=cmap)
        axes[i].set_title(f"Slice {idx}")
        axes[i].axis("off")

    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# 4. Error map
# --------------------------------------------------

def show_error_map(gt, recon, z=None, cmap="hot"):
    if z is None:
        z = gt.shape[2] // 2

    error = np.abs(gt - recon)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(gt[:, :, z], cmap="gray")
    axes[0].set_title("GT")

    axes[1].imshow(recon[:, :, z], cmap="gray")
    axes[1].set_title("Recon")

    im = axes[2].imshow(error[:, :, z], cmap=cmap)
    axes[2].set_title("Error")

    for ax in axes:
        ax.axis("off")

    fig.colorbar(im, ax=axes[2])
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# 5. Perfil de intensidad
# --------------------------------------------------

def plot_intensity_profile(gt, recon, axis=0, index=None, slice_idx=None):
    if slice_idx is None:
        slice_idx = gt.shape[2] // 2

    if index is None:
        index = gt.shape[axis] // 2

    if axis == 0:
        gt_profile = gt[index, :, slice_idx]
        recon_profile = recon[index, :, slice_idx]
    elif axis == 1:
        gt_profile = gt[:, index, slice_idx]
        recon_profile = recon[:, index, slice_idx]
    else:
        gt_profile = gt[:, :, index]
        recon_profile = recon[:, :, index]

    plt.figure(figsize=(8, 4))
    plt.plot(gt_profile, label="GT")
    plt.plot(recon_profile, label="Recon", linestyle="--")
    plt.title("Intensity Profile")
    plt.legend()
    plt.grid()
    plt.show()


# --------------------------------------------------
# 6. MIP (Maximum Intensity Projection)
# --------------------------------------------------

def show_mip(volume, axis=2, cmap="gray"):
    mip = np.max(volume, axis=axis)

    plt.figure(figsize=(5, 5))
    plt.imshow(mip, cmap=cmap)
    plt.title("MIP")
    plt.axis("off")
    plt.show()