import numpy as np
from numpy.linalg import svd
from numpy import dot
from time import time
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

#fast version
from scikits.learn.utils.extmath import fast_svd
from scikits.learn.cluster import k_means, affinity_propagation

def ap(z):
    sim_matrix = -squareform(pdist(z, 'euclidean'))

    p = np.median(sim_matrix) * 1
    center, labels = affinity_propagation(sim_matrix, p=p, verbose=True, max_iter=2000)

    return labels

def show_cluster(A, name):
    print A
    #pl.matshow(A, aspect='auto', origin = 'lower', cmap=pl.cm.YlGnBu)
    #pl.savefig(name)

def spectral_coclustering(A, use_k_means=False, k=3):

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

    if use_k_means:
        _, labels, _ = k_means(z, k)

    else:
        t1 = time()
        labels = ap(z)
        print "%s" % (time() - t1)

    r_labels = labels[:A.shape[0]]
    c_labels = labels[A.shape[0]:]
    return r_labels, c_labels

def reorder_axis(labels):
    uniq_labels = np.unique(labels)
    new_idx = []
    
    for i in uniq_labels:
        i_idx = np.where(np.array(labels) == i)
        new_idx.extend(i_idx[0].tolist())

    return new_idx

def reorder_matrix(mat, r_labels, c_labels):
    """
    Given a dense cluster, sort the points inside the cluster 
    so that they are less tangled and look nice.
    """
    show_cluster(mat, "before.png")

    new_r_idx = reorder_axis(r_labels)
    new_c_idx = reorder_axis(c_labels)

    print new_r_idx
    print new_c_idx

    new_r_idx = np.array(new_r_idx)[:, np.newaxis]

    new_mat = mat[new_r_idx, new_c_idx]

    show_cluster(new_mat, "after.png")
    

if __name__ == "__main__":
    A = np.loadtxt('town.csv', delimiter=",")
    r_labels, c_labels = spectral_coclustering(A)
    reorder_matrix(A, r_labels, c_labels)
