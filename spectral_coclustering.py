import numpy as np
from numpy.linalg import svd
from numpy import dot
from time import time

#fast version
from sklearn.utils.extmath import fast_svd
from sklearn.cluster import k_means

def spectral_coclustering(A, k=3):

    D1 = np.sum(A, 1)
    D2 = np.sum(A, 0)


    D1_root = np.diag(np.abs((D1)**(-0.5)))
    D2_root = np.diag(np.abs((D2)**(-0.5)))

    An = dot(D1_root, dot(A, D2_root))

    #print An
    t1 = time()
    U, S, V = svd(An)
    V = np.transpose(V)
    #print U, S, V
    print "%s" % (time() - t1)

    n_eigen = np.ceil(np.log2(k))
    z = np.vstack((dot(D1_root, U[:, 1:n_eigen+1]),
            dot(D2_root, V[:,1:n_eigen+1])))

    _, labels, _ = k_means(z, k)


    r_labels = labels[:A.shape[0]]
    c_labels = labels[A.shape[0]:]
    return r_labels, c_labels

if __name__ == "__main__":
    A = np.loadtxt('town.csv', delimiter=",")
    r_labels, c_labels = spectral_coclustering(A)
