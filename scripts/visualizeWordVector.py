import sys
import codecs
import numpy as np
import matplotlib.pyplot as plt

import tsne as tsneSource


def main():
    embeddings_file = sys.argv[1]
    labels, vectors = load_embeddings(embeddings_file)

    Y = tsneSource.tsne(vectors, 2, 50, 20.0)
    plt.scatter(Y[:,0], Y[:,1], 20, labels)
    for label, x, y in zip(labels, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()
    plt.savefig('result.png')


def load_embeddings(file_name):
    print('Reading file: ',file_name)
    with codecs.open(file_name, 'r', 'utf-8') as f_in:
        labels, vectors = zip(*[line.strip().split(' ', 1) for line in
f_in])
    vectors = np.loadtxt(vectors)
    return labels, vectors

if __name__ == '__main__':
    main()
