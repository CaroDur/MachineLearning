from sklearn.cluster import KMeans
import numpy as np
from time import time
from sklearn.utils import shuffle
from sklearn.metrics import pairwise_distances_argmin

def kmeans_colors(img, k, max_iter=100):
    """
    Performs k-means clusering on the pixel values of an image.
    Used for color-quantization/compression.

    Args:
        img: The input color image of shape [h, w, 3]
        k: The number of color clusters to be computed

    Returns:
        img_cl:  The color quantized image of shape [h, w, 3]

    """

    img_cl = None

    #######################################################################
    # TODO:                                                               #
    # Perfom k-means clustering of the pixel values of the image img.     #
    #######################################################################
    img_conc = np.concatenate(img, axis=0)
    # Generate k clusters and get their labels and values
    img_kmeans = KMeans(n_clusters=k, max_iter=max_iter).fit(img_conc)
    img_labels = img_kmeans.labels_
    img_clusters = img_kmeans.cluster_centers_

    img_cl = img_clusters[img_labels, :].astype(int)
    img_cl = np.reshape(img_cl, img.shape)

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    return img_cl
