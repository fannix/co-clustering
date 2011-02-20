
import numpy as np
from numpy.linalg import svd
from numpy import dot
from time import time
from scipy.cluster.hierarchy import dendrogram, linkage

#fast version
from scikits.learn.utils.extmath import fast_svd
from scikits.learn.cluster import k_means, affinity_propagation




if __name__ == "__main__":
    A = np.loadtxt('town.csv', delimiter=",")

    _, labels, _ = k_means(A, 3)
    print labels
