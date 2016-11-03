import os.path as osp
from itertools import product

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from data_handling import apply_masking as mask_zeros


FOLDER_DATA = '/data/shared/bvFTD/Machine_Learning/data'


def load_data():
    data = np.load(osp.join(FOLDER_DATA, 'data_set.npy'))
    group_labels = pd.read_csv(osp.join(FOLDER_DATA, 'class_labels.csv'))
    id_ftd = group_labels.ftd.as_matrix()
    id_neurol = group_labels.neurol.as_matrix()
    id_psych = group_labels.psych.as_matrix()
    return data, id_ftd, id_neurol, id_psych


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
    print 'Learning rate: ({}) Perplexity: ({}) #Components: ({})'.format(rate, perp, n)
    tsne = TSNE(n_components=n, learning_rate=rate, perplexity=perp, init='pca', method='exact')
    tsne.fit(data)
    print tsne.kl_divergence_
    print
    return tsne.kl_divergence_, tsne.embedding_


def run():
    print 'Load Data'
    data, id_ftd, id_neurol, id_psych = load_data()
    data = mask_zeros(data)
    print 'Run PCA'
    pca = PCA(n_components=50)
    data_pca = pca.fit_transform(data)
    scaler = StandardScaler()
    data_pca_scl = scaler.fit_transform(data_pca)

    learn_rate = np.linspace(100, 1000, 200)
    perplexity = np.arange(1, 51)
    n_components = [2, 3]

    i_total = len(learn_rate) * len(perplexity) * len(n_components)
    n_jobs = 15

    tsne_params = list(product(learn_rate, perplexity, n_components))
    print 'Start Grid Search'
    tsne_metrics = Parallel(n_jobs=n_jobs, verbose=1)(delayed(inner_loop_iteration)(data_pca_scl, rate, perp, n, i, i_total)
                                                      for i, (rate, perp, n) in enumerate(tsne_params))

    kl_div, embeddings = zip(*tsne_metrics)
    kl_div = np.array(kl_div)
    # to make the embeddings better saveable we add a column of nans to the 2 dimensional embeddings (as a dummy)
    # the final embeddings array has the shape: len(learn_rate) * len(perplexity) * len(n_components) x 67 x 3
    embeddings = np.array([np.column_stack((e, np.ones(e.shape[0]) * np.nan)) if e.shape[1] == 2 else e
                           for e in embeddings])
    kl_div_min, kl_div_idx = kl_div.min(), kl_div.argmin()
    print 'Minumum KL-divergence found: {}, using the following parameters:'.format(kl_div_min)
    print 'Learning rate: {}, Perplexity: {}, Number of components: {}'.format(tsne_params[kl_div_idx][0],
                                                                               tsne_params[kl_div_idx][1],
                                                                               tsne_params[kl_div_idx][2])

    np.savez_compressed('tsne_results.npz', kl_div=kl_div, embeddings=embeddings, tsne_params=np.array(tsne_params))


if __name__ == '__main__':
    run()
