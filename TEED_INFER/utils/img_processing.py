import numpy as np

def image_normalization(img, img_min=0, img_max=255,
                        epsilon=1e-12):
    """Normalize intensities to the requested range."""

    img = np.float32(img)

    img = (img - np.min(img)) * (img_max - img_min) /\
        ((np.max(img) - np.min(img)) + epsilon) + img_min
    return img
