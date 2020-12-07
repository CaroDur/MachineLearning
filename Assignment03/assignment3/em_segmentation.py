import numpy as np
from sklearn.mixture import GaussianMixture

import time


def em_segmentation(img, k, max_iter=20):
    """
    Learns a MoG model using the EM-algorithm for image-segmentation.

    Args:
        img: The input color image of shape [h, w, 3]
        k: The number of gaussians to be used

    Returns:
        label_img: A matrix of labels indicating the gaussian of size [h, w]

    """

    label_img = None

    #######################################################################
    # TODO:                                                               #
    # 1st: Augment the pixel features with their 2D coordinates to get    #
    #      features of the form RGBXY (see np.meshgrid)                   #
    # 2nd: Fit the MoG to the resulting data using                        #
    #      sklearn.mixture.GaussianMixture                                #
    # 3rd: Predict the assignment of the pixels to the gaussian and       #
    #      generate the label-image                                       #
    #######################################################################
    
    # prepare all data
    ny, nx, nz = img.shape
    x = np.linspace(0, nx, nx)
    y = np.linspace(0, ny, ny)
    xv, yv = np.meshgrid(x, y)
    xv = np.concatenate(xv)
    yv = np.concatenate(yv)
    img_conc = np.concatenate(img, axis=0)
    # stack xv and yv to img to get RGBXY shape
    img_large = np.stack([img_conc[:,0],img_conc[:,1],img_conc[:,2], xv, yv], axis=-1)
    # apply Gaussian Mixture to RGBXY image and then reshape to return wanted image
    img_gaussian = GaussianMixture(n_components=k, max_iter=max_iter).fit_predict(img_large)
    label_img = np.reshape(img_gaussian, img.shape[:2])

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    return label_img
