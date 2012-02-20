import numpy as np
import scipy.sparse as sparse
import networkx as nx

def save_sparse_matrix(filename, x):
    x_coo = x.tocoo()
    row = x_coo.row
    col = x_coo.col
    data = x_coo.data
    shape = x_coo.shape
    np.savez(filename, row=row, col=col, data=data, shape=shape)

def load_sparse_matrix(filename):
    y = np.load(filename)
    z = sparse.coo_matrix((y['data'], (y['row'], y['col'])), shape=y['shape'])
    return z

def sparse_biparty_matrix_to_graph(S):
    """
    Convert a sparse biparty matrix to a graph
    """
    S_coo = S.tocoo()
    nrow, ncol = S_coo.shape
    n = nrow + ncol
    S_new = sparse.dok_matrix((n, n))
    for i in range(S.nnz):
        row_idx = S_coo.row[i]
        col_idx = S_coo.col[i] + nrow
        data = S_coo.data[i]
        S_new[row_idx, col_idx] = data
        S_new[col_idx, row_idx] = data
    G = nx.Graph(S_new.tolil())
    nx.write_gexf(G, "matrix/graph.gexf")
