import numpy as np
from LucasKanadeAffine import LucasKanadeAffine
from scipy.ndimage import binary_dilation
from scipy.ndimage import binary_erosion
from scipy.ndimage import affine_transform
from InverseCompositionAffine import InverseCompositionAffine
import ipdb

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """
    
    # put your implementation here
    mask = np.ones(image1.shape, dtype=bool)
    # M = LucasKanadeAffine(image1, image2, threshold, num_iters)
    M = InverseCompositionAffine(image1, image2, threshold, num_iters)
    print("M", M)
    # M = np.linalg.inv(M)
    # r1,c1 = image1.shape
    # r2, c2 = image2.shape
    w1 = affine_transform(image1, -M[:2], output_shape = image1.shape)
    print("w1", w1)
    # err = abs(w1 - image2)
    # mask[err > tolerance] = 1
    # mask[err < tolerance] = 0
    # mask = binary_erosion(mask)
    # mask = binary_dilation(mask)

    ed_2 = binary_erosion(w1)
    dil_2 = binary_dilation(ed_2)

    err = np.abs(dil_2 - image2)
    mask = (err > tolerance)
    print("Mask", mask)

    return mask
