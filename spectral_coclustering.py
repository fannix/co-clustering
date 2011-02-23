import numpy as np
from numpy.linalg import svd
from numpy import dot
from time import time
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

#fast version
from scikits.learn.utils.extmath import fast_svd
from scikits.learn.cluster import k_means, affinity_propagation
import pylab as pl

def ap(z):
    sim_matrix = -squareform(pdist(z, 'euclidean'))

    p = np.median(sim_matrix) * 1
    center, labels = affinity_propagation(sim_matrix, p=p, verbose=True, max_iter=2000)

    return labels

def show_cluster(A, name):
    print A
    cdict = {'red': ((0.0, 1.0, 1.0),
                     (1.0, 0.5, 0.5)),
             'green': ((0.0, 1.0, 1.0),
                       (1.0, 0.5, 0.5)),
             'blue': ((0.0, 1.0, 1.0),
                      (1.0, 0.5, 0.5))}

    my_cmap = pl.matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    fig = pl.figure()
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    ax.matshow(A, aspect='auto', origin = 'lower', cmap=my_cmap)
    n_row, n_col = A.shape
    for y in np.arange(n_row) + 0.5:
        ax.axhline(y,color="black")

    for y in np.arange(n_col) + 0.5:
        ax.axvline(y,color="black")

    #ax.set_xticks(np.arange(n_col))
    #ax.set_yticks(np.arange(n_row))
    labels = ax.set_yticklabels(["product"] * n_col)
    labels = ax.set_xticklabels(["features"] * n_row, rotation=30)

    fig.savefig(name)

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
        i_idx = np.where(labels == i)
        new_idx.extend(i_idx[0].tolist())

    return new_idx

def reorder_matrix(mat, r_labels, c_labels):
    """
    Given a dense cluster, sort the points inside the cluster 
    so that they are less tangled and look nice.
    """
    show_cluster(mat, "before.pdf")

    new_r_idx = reorder_axis(r_labels)
    new_c_idx = reorder_axis(c_labels)

    new_r_idx = np.array(new_r_idx)[:, np.newaxis]

    new_mat = mat[new_r_idx, new_c_idx]

    # adjust the order between clusters
    new_mat = adjust_col(new_mat, r_labels, c_labels, new_r_idx, new_c_idx)
    
    show_cluster(new_mat, "after.pdf")

    return new_mat, new_r_idx, new_c_idx

def adjust_col(new_mat, r_labels, c_labels, new_r_idx, new_c_idx):
    uniq_r_labels = np.unique(r_labels)

    c_li = [0]

    n_row, n_col = new_mat.shape

    left = range(n_col)
    left.remove(c_li[-1])
    while len(c_li) != n_col:
        last = c_li[-1]
        #print c_li, left
        argmax = left[0]
        max_sim = sim(new_mat, last, argmax)
        for j in left:
            new_simi = sim(new_mat, last, j)
            if new_simi > max_sim:
                argmax = j
                max_sim = new_simi
        c_li.append(argmax)
        left.remove(argmax)

    return new_mat[:, c_li]

def sim(mat, i, j):
    """
    compute the similarity between column i and j
    """
    return (np.sum(mat[:, i] == mat[:, j]))

if __name__ == "__main__":
    A = np.loadtxt('town.csv', delimiter=",")
    r_labels, c_labels = spectral_coclustering(A)
    new_A, new_r_idx, new_c_idx = reorder_matrix(A, r_labels, c_labels)
