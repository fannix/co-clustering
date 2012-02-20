"""
Create an En-Ch pairwise relation matrix
"""
import sys
import numpy as np
import scipy.sparse as sparse
from util import sparse_biparty_matrix_to_graph

def create_matrix(en_vocab, ch_vocab, en_lines, ch_lines, alignment):
    """
    Create an En-Ch pairwise relation matrix

    The relation is alignment occrurences

    Parameters
    ----------
    en_vocab: dict
    ch_vocab: dict
    en_lines: list of English lines
    ch_lines: list of Chinese lines
    alignment: list of En-Ch alignment

    Returns
    -------
    A sparse relation matrix
    """
    assert len(en_lines) == len(alignment) and len(ch_lines) == len(alignment)

    nline = len(alignment)
    S = sparse.dok_matrix((len(ch_vocab), len(en_vocab)))
    for i in range(nline):
        align_words = alignment[i].split()
        en_words = en_lines[i].split()
        ch_words = ch_lines[i].split()

        for pair in align_words:
            ch_order, en_order = pair.split('-')
            ch_idx = ch_vocab[ch_words[int(ch_order)]]
            en_idx = en_vocab[en_words[int(en_order)]]

            #print "%s-%s" % (ch_idx, en_idx)
            S[ch_idx-1, en_idx-1] += 1

    return S

if __name__ == "__main__":

    alignment = []

    for line in sys.stdin:
        alignment.append(line.strip())

    en_file = "raw/training.en"
    ch_file = "raw/training.ch"
    en_lines = open(en_file).readlines()[:len(alignment)]
    ch_lines = open(ch_file).readlines()[:len(alignment)]

    en_vocab_file = "matrix/en.span"
    ch_vocab_file = "matrix/ch.span"
    en_vocab_li = [e.strip() for e in open(en_vocab_file).readlines()]
    ch_vocab_li = [e.strip() for e in open(ch_vocab_file).readlines()]
    en_vocab = dict(zip(en_vocab_li, range(1, len(en_vocab_li)+1)))
    ch_vocab = dict(zip(ch_vocab_li, range(1, len(ch_vocab_li)+1)))
    S = create_matrix(en_vocab, ch_vocab, en_lines, ch_lines, alignment)
    sparse_biparty_matrix_to_graph(S)
