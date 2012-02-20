"""
Convert from text to vector space
"""
import sys
from optparse import OptionParser
from collections import defaultdict
import os.path

def span(span_space, with_label = None):
    """
    Span the string read from stdin into the vector space

    Parameter
    ---------
    span_space : This file save the spanned space

    with_label : If not None, indicate the input have labels 
    """
    word2idx = {}
    idx2word = {}
    i = 1
    if os.path.exists(span_space):
        with open(span_space) as f:
            for line in f:
                word = line.strip()
                word2idx[word] = i
                idx2word[i] = word
                i += 1
    old_vocab_size = len(word2idx)
    print >> sys.stderr, "Oirginal dictionary contain %d words" % old_vocab_size
    for line in sys.stdin:
        li = line.strip().split()
        if with_label:
            if with_label == "0":
                li = li[1:]
            else:
                li = li[:-1]
        for word in li:
            if word not in word2idx:
                word2idx[word] = i
                idx2word[i] = word
                i += 1
    new_vocab_size = len(word2idx)
    if new_vocab_size > old_vocab_size:
        with open(span_space, 'a') as f:
            for j in range(old_vocab_size+1, new_vocab_size+1):
                f.write(idx2word[j]+"\n")

    print >> sys.stderr, "new dictionary contain %d words" % new_vocab_size

def project(project_space, with_label):
    """
    Project the string from stdin into the vector space

    Parameter
    ---------
    project_space : This file indicate the vector space to project into

    with_label : If not None, indicate the input have labels 
    """
    word2idx = {}
    i = 1
    with open(project_space) as f:
        for line in f:
            word = line.strip()
            word2idx[word] = i
            i += 1

    for line in sys.stdin:
        li = line.strip().split()
        if with_label:
            if with_label == "0":
                label = li[0]
                li = li[1:]
            else:
                label = li[1]
                li = li[:-1]
            print label,
        else:
            print "0",
        di = defaultdict(lambda: 0)
        for word in li:
            if word in word2idx:
                di[word] += 1

        li = []
        for word in di:
            li.append((word2idx[word], di[word]))
        li.sort()
        for idx, count in li:
            print "%d:%d" % (idx, count),
        print

if __name__ == "__main__":
    usage = "Usage: %prog -s/-p dict_file < documents"
    parser = OptionParser(usage=usage)
    parser.add_option("-s", "--span", dest = "span_space", 
            help="This file saves the spanned space.")
    parser.add_option("-p", "--project", dest = "project_space",
            help="This file indicates the space to project to.")
    parser.add_option("-t", "--label", dest = "with_label", 
            help="The input file have accompany label. \
            Use O indicate the label is the first token and 1 \
            indicate the label is the last token")
    (options, args) = parser.parse_args()
    if options.span_space:
        span(options.span_space, options.with_label)
    elif options.project_space:
        project(options.project_space, options.with_label)
