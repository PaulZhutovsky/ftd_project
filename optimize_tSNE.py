import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import os.path as osp
import pandas as pd
from itertools import product

FOLDER_DATA = '/data/shared/bvFTD/Machine_Learning/data'


def load_data():
    X = np.load(osp.join(FOLDER_DATA, 'data_set.npy'))
    group_labels = pd.read_csv(osp.join(FOLDER_DATA, 'class_labels.csv'))
    id_ftd = group_labels.ftd.as_matrix()
    id_neurol = group_labels.neurol.as_matrix()
    id_psych = group_labels.psych.as_matrix()
    return X, id_ftd, id_neurol, id_psych


def mask_zeros(X):
    id_zeros = np.all(X == 0, axis=0)
    X_masked = X[:, ~id_zeros]
    return X_masked


def make_scatter(data, id_ftd, id_neurol, id_psych):
    colors = ['r', 'g', 'b']
    plt.scatter(data[id_ftd, 0], data[id_ftd, 1], c=colors[0], s=30)
    if data.shape[1] == 2:
        plt.scatter(data[id_neurol, 0], data[id_neurol, 1], c=colors[1], s=30)
        plt.scatter(data[id_psych, 0], data[id_psych, 1], c=colors[1], s=30)
        plt.legend(['FTD', 'Rest'])
    else:
        plt.scatter(data[id_neurol, 0], data[id_neurol, 1], c=colors[1], s=30)
        plt.scatter(data[id_psych, 0], data[id_psych, 1], c=colors[2], s=30)
        plt.legend(['FTD', 'Neurological', 'Psychiatry'])
    plt.show()


def inner_loop_iteration(data, rate, perp, n, i, i_total):
    print '#Iteration: {} / {}'.format(i, i_total)
    print '#Learning rate: ({}) #Perplexity: ({}) #Components: ({})'.format(rate, perp, n)
    print
    tsne = TSNE(n_components=n, learning_rate=rate, perplexity=perp, init='pca', method='exact')
    tsne_fit = tsne.fit(data)
    return tsne_fit.kl_divergence_, tsne_fit.embedding_


def run():
    X, id_ftd, id_neurol, id_psych = load_data()
    X = mask_zeros(X)
    pca = PCA(n_components=500)
    X_pca = pca.fit_transform(X)
    scaler = StandardScaler()
    X_pca_scl = scaler.fit_transform(X_pca)

    learn_rate = np.linspace(100, 1000, 200)
    perplexity = np.arange(1, 51)
    n_components = np.arange(2, 4)
    i_total = len(learn_rate) * len(perplexity) * len(n_components)
    n_jobs = 5

    tsne_params = list(product(learn_rate, perplexity, n_components))
    tsne_metrics = Parallel(n_jobs=n_jobs, verbose=1)(delayed(inner_loop_iteration)(X_pca_scl, rate, perp, n, i, i_total)
                                                      for i, (rate, perp, n) in enumerate(tsne_params))

    kl_div, embeddings = zip(*tsne_metrics)
    kl_div_min, kl_div_idx = np.array(kl_div).min(), np.array(kl_div).argmin()
    print 'Minumum KL-divergence found: {}, using the following parameters:'.format(kl_div_min)
    print 'Learning rate: {}, Perplexity: {}, Number of components: {}'.format(tsne_params[kl_div_idx][0],
                                                                               tsne_params[kl_div_idx][1],
                                                                               tsne_params[kl_div_idx][2])

    make_scatter(embeddings[kl_div_idx], id_ftd, id_neurol, id_psych)


if __name__ == '__main__':
    run()
