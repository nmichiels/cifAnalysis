import numpy as np
from skimage import measure
from skimage import filters


def hasOneBlob(mask):
    

    blobs = mask > 0.7 * mask.mean()

    all_labels = measure.label(blobs)
    blobs_labels = measure.label(blobs, background=0)

    if (np.amax(blobs_labels) == 1):
        return True
    else:
        return False
