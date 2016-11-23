""""
Visualizes the FTD results

Usage:
    visualize_results.py FOLDER PATTERN SAVE_FOLDER

Arguments:
    FOLDER          Folder where the results are stored
    PATTERN         Which folders will be selected (under FOLDER)
    SAVE_FOLDER     Where to store the created figures
"""


import os.path as osp
from cPickle import load
from glob import glob

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from docopt import docopt

from data_handling import ensure_folder


def plot_histograms(data, title, figtitle, range_x=(0, 1), save_name='results.png'):
    fig, ax = plt.subplots(2, 2, sharey=True, figsize=(20, 20))
    ax = ax.ravel()

    n_points, n_plots = data.shape
    for id_plot in xrange(n_plots):
        data_id = data[:, id_plot]
        data_m = data_id.mean()
        data_sd = data_id.std()
        data_se = data_sd/np.sqrt(n_points)

        n, _, _ = ax[id_plot].hist(data_id, bins=30)
        ax[id_plot].vlines(data_m, 0, n.max() + 10, colors='r', lw=2)
        ax[id_plot].vlines(data_m - data_sd, 0, n.max() + 10, colors='g', lw=2)
        ax[id_plot].vlines(data_m + data_sd, 0, n.max() + 10, colors='g', lw=2)
        ax[id_plot].set_xlim(range_x)
        ax[id_plot].tick_params(axis='both', labelsize=18)
        ax[id_plot].set_title('{}: M: {:.4f} SE: {:.4f} SD: {:.4f}'.format(title[id_plot], data_m, data_se, data_sd),
                              fontsize=26)
    if n_plots < ax.size:
        ax[-1].axis('off')
    fig.suptitle(figtitle, fontsize=28)
    fig.savefig(save_name)


def plot_best_models(best_models, title_plots, save_name='model.png'):
    fig, ax = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(15, 15))
    ax = ax.ravel()

    n_plots = len(best_models)

    for id_plot in xrange(n_plots):
        best_model = best_models[id_plot]
        labels, counts = np.unique(best_model, return_counts=True)
        counts = counts/counts.sum(dtype=np.float) * 100
        id_labels = np.arange(1, labels.size + 1)

        ax[id_plot].bar(id_labels, counts, tick_label=labels, align='center')
        ax[id_plot].tick_params(axis='both', labelsize=18)
        ax[id_plot].set_title(title_plots[id_plot], fontsize=20)
        ax[id_plot].set_ylim((0, 100))

        if (id_plot == 0) or (id_plot == 2):
            ax[id_plot].set_ylabel('chosen model (%)', fontsize=20)
    if n_plots < ax.size:
        ax[-1].axis('off')
    fig.savefig(save_name)


def plot_predictions(predictions_all, title_plots, n_ftd=18, threshold_correct=50, save_file='predictions.png'):
    n_plots = len(predictions_all)

    fig, ax = plt.subplots(n_plots, 1, sharey=True, figsize=(15, 15))
    fig.subplots_adjust(top=0.95, bottom=0.05, hspace=0.30)
    ax = ax.ravel()

    for id_plot in xrange(n_plots):
        pred = predictions_all[id_plot].astype(np.int)

        y_true = np.concatenate((np.ones(n_ftd), np.zeros(pred.shape[0] - n_ftd))).astype(np.int)
        correct_pred = (pred == y_true[:, np.newaxis]).sum(axis=1)
        num_predictions_made = (pred.shape[1] - (pred == -1).sum(axis=1, dtype=np.float))
        correct_pred_perc = correct_pred / num_predictions_made * 100

        id_ftd = np.arange(1, n_ftd + 1)
        id_other = np.arange(n_ftd + 1, pred.shape[0] + 1)

        ax[id_plot].axhline(y=threshold_correct, color='r', linewidth=2)
        ax[id_plot].bar(id_ftd, correct_pred_perc[:n_ftd], align='center', color='m')
        ax[id_plot].bar(id_other, correct_pred_perc[n_ftd:], align='center', color='c')
        ax[id_plot].set_title('{} adjusted for undersampling'.format(title_plots[id_plot]), fontsize=20)
        ax[id_plot].set_ylabel('% correct', fontsize=20)
        ax[id_plot].tick_params(axis='both', labelsize=18)
        ax[id_plot].set_xlim((0, id_other.max() + 1))
    if n_plots < ax.size:
        ax[-1].axis('off')
    fig.savefig(save_file)


def plot_roc(roc_data, title_plots, save_file='roc.png'):
    n_plots = len(roc_data)

    fig, ax = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(15, 15))
    ax = ax.ravel()

    for id_plot in xrange(n_plots):
        roc = roc_data[id_plot]
        ax[id_plot].plot([0, 1], [0, 1], '-r', lw=3)
        [ax[id_plot].plot(roc[i][0], roc[i][1], '-o', color='gray', lw=2, alpha=0.6) for i in xrange(len(roc))]
        ax[id_plot].set_title('{}'.format(title_plots[id_plot]), fontsize=20)
        ax[id_plot].set_ylabel('TPR (Sensitivity)', fontsize=20)
        ax[id_plot].set_xlabel('FPR (1-Specificity)', fontsize=20)
        ax[id_plot].tick_params(axis='both', labelsize=18)
        ax[id_plot].set_xlim((-0.05, 1.05))
        ax[id_plot].set_ylim((-0.05, 1.05))
    if n_plots < ax.size:
        ax[-1].axis('off')
    fig.savefig(save_file)


def load_predictions(results_folder):
    predictions = [np.load(osp.join(folder, 'predictions.npy')) for folder in results_folder]
    return predictions


def load_model_chosen(results_folder):
    best_models = []
    for folder in results_folder:
        with open(osp.join(folder, 'best_model_labels.pkl'), 'rb') as f:
            best_models.append(load(f))
    return best_models


def load_performance(results_folder):
    metrics = []
    metric_label = ''

    for folder in results_folder:
        tmp = np.load(osp.join(folder, 'performance_metrics.npz'))
        metric = tmp['metrics']
        metric_label = tmp['metrics_labels']
        metrics.append(metric.mean(axis=0))

    metrics = np.array(metrics).transpose((1, 2, 0))
    return metrics, metric_label


def load_roc(results_folder):
    roc_curves = []

    for folder in results_folder:
        with open(osp.join(folder, 'roc_curves.pkl'), 'rb') as f:
            roc_curves.append(load(f))
    return roc_curves


def compute_average_roc(roc_curves, n_folds=5):
    """
    Function is currently NOT USED due to the fact that our ROC curves don't have enough samples to meaningfully
    interpolate values at. Will be kept here just as a general reference on how to do this

    :param roc_curves:
    :param n_folds:
    :return:
    """

    n_results = len(roc_curves)
    mean_fpr = np.linspace(0, 1, 100)
    roc_avg = []

    for id_result in xrange(n_results):
        roc_result = roc_curves[id_result]
        n_curves = len(roc_result)
        tpr_all = np.zeros((mean_fpr.size, n_curves/n_folds))
        for i, id_curves in enumerate(xrange(0, n_curves, n_folds)):
            roc_fold = roc_result[id_curves: id_curves + n_folds]

            mean_tpr = np.array([np.interp(mean_fpr, roc_fold[id_cv][0], roc_fold[id_cv][0])
                                 for id_cv in xrange(n_folds)]).mean(axis=0)
            mean_tpr[0] = 0.
            mean_tpr[-1] = 1.
            tpr_all[:, i] = mean_tpr
        roc_avg.append((mean_fpr, tpr_all, tpr_all.mean(axis=1)))

    return roc_avg


def histogram_of_performance(data, data_labels, title_results, save_folder):
    for i in xrange(data.shape[1]):
        data_metric = data[:, i, :]
        metric_label = data_labels[i]
        save_file = osp.join(save_folder, metric_label + '.png')
        plot_histograms(data_metric, title_results, metric_label, save_name=save_file)


def strip_results_folder_parts(results_folder):
    results_folder = np.array([osp.basename(results) for results in results_folder])
    results_folder = np.char.split(results_folder, '_')
    results_folder = np.array([folder[1] + ' vs. ' + folder[2] for folder in results_folder])
    return results_folder


def plotter_results(data_pattern, save_folder):
    results_folder = sorted(glob(data_pattern))
    title_results = strip_results_folder_parts(results_folder)

    ensure_folder(save_folder)

    print 'Plot performance metrics'
    metrics, metric_label = load_performance(results_folder)
    histogram_of_performance(metrics, metric_label, title_results, save_folder)

    print 'Plot chosen models'
    best_models = load_model_chosen(results_folder)
    plot_best_models(best_models, title_results, osp.join(save_folder, 'best_models.png'))

    print 'Plot subject predictions'
    predictions_subjects = load_predictions(results_folder)
    plot_predictions(predictions_subjects, title_results, save_file=osp.join(save_folder, 'subject_predictions.png'))

    print 'Plot ROC curves'
    roc_curves = load_roc(results_folder)
    plot_roc(roc_curves, title_results, osp.join(save_folder, 'roc_curves.png'))


def run(args):
    data_pattern = args['PATTERN']
    data_folder = args['FOLDER']
    save_folder = args['SAVE_FOLDER']
    plotter_results(osp.join(data_folder, data_pattern), save_folder)


if __name__ == '__main__':
    arguments = docopt(__doc__)
    run(arguments)

