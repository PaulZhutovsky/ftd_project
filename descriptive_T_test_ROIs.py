from glob import glob
from os import path as osp
from data_handling import create_file_name, ensure_folder
from scipy import stats
import pandas as pd
import numpy as np

REG_PATTERN = 'results_*_atlas_*no_covariates'
ATLAS_RESULTS_FOLDERS = sorted(glob(osp.join('/data/shared/bvFTD/Machine_Learning', REG_PATTERN)))
DATA_FOLDER = '/data/shared/bvFTD/Machine_Learning/data'
PARCELLATED_DATA_FOLDER = '/data/shared/bvFTD/Machine_Learning/data/parcellated_GM_data/'
P_THRESHOLD = 0.05


def load_data(result_folder):
    """
    Class names and smoothing status are extracted from the result folder name with the following structure:
    'results_(CLASS1)vs(CLASS2)_atlas_(SMOOTHING STATUS)_(COVARIATES STATUS)'
    """
    classifiers = result_folder.split('results_')[1]
    cl1, cl2 = str.lower(classifiers.split('vs')[0]), str.lower(classifiers.split('vs')[-1].split('_')[0])
    smoothing = True if classifiers.split('_')[2] == 'smoothed' else False
    data_filename = create_file_name(parcellation=True, smoothing=smoothing)
    parcellated_data = np.load(osp.join(DATA_FOLDER, data_filename))
    predictions = np.load(osp.join(result_folder, 'predictions.npy')).astype(np.int)
    return cl1, cl2, parcellated_data, predictions


def validate_minimum_subjects(correct_idx, incorrect_idx, n_class1):
    # Validate whether a minimum of 2 subjects is included per group for its corresponding t-test
    suff_subj_cl1 = suff_subj_cl2 = True
    _, counts_corr_cl1 = np.unique(correct_idx[:n_class1], return_counts=True)
    _, counts_incorr_cl1 = np.unique(incorrect_idx[:n_class1], return_counts=True)
    _, counts_corr_cl2 = np.unique(correct_idx[n_class1:], return_counts=True)
    _, counts_incorr_cl2 = np.unique(incorrect_idx[n_class1:], return_counts=True)
    if (len(counts_corr_cl1) < 2 or counts_corr_cl1[1] < 2) or \
            (len(counts_incorr_cl1) < 2 or counts_incorr_cl1[1] < 2):
        suff_subj_cl1 = False
    if (len(counts_corr_cl2) < 2 or counts_corr_cl2[1] < 2) or \
            (len(counts_incorr_cl2) < 2 or counts_incorr_cl2[1] < 2):
        suff_subj_cl2 = False
    return suff_subj_cl1, suff_subj_cl2


def roi_t_test(ROI, correct_clf_data, incorrect_clf_data):
    # Test for equal variance between compared groups
    _, p = stats.levene(correct_clf_data[:, ROI], incorrect_clf_data[:, ROI])
    equal_var = True if p > 0.05 else False
    # For every ROI run T-tests between correctly and incorrectly classified subjects
    t_value, p_value = stats.ttest_ind(correct_clf_data[:, ROI], incorrect_clf_data[:, ROI], equal_var=equal_var)
    return t_value, p_value


def save_results(class_name, results, save_folder, suff_subj):
    if suff_subj:
        # Sort results on significance and ROI code prior to saving
        results = results.sort_values(by=['significant', 'code_roi'], ascending=[0, 1])
        filename_results = class_name + '_correct_vs_incorrect_ROI_t_tests.csv'
        results.to_csv(osp.join(save_folder, filename_results))


def descriptive_t_tests():
    class_labels_df = pd.read_csv(osp.join(DATA_FOLDER, 'class_labels.csv'))
    roi_labels_df = pd.read_csv(osp.join(PARCELLATED_DATA_FOLDER, 'cort_subcort_full_labels.csv'))

    for i in xrange(len(ATLAS_RESULTS_FOLDERS)):
        save_folder = osp.join(ATLAS_RESULTS_FOLDERS[i], 'ROI_t_tests')
        ensure_folder(save_folder)

        class1, class2, data, pred = load_data(ATLAS_RESULTS_FOLDERS[i])
        class1_idx, class2_idx = class_labels_df[class1].values, class_labels_df[class2].values
        class1_data, class2_data = data[class1_idx], data[class2_idx]
        n_ROI, n_class1, n_class2 = data.shape[1], len(class1_data), len(class2_data)

        y_true = np.concatenate((np.ones(n_class1), np.zeros(pred.shape[0] - n_class1))).astype(np.int)
        correct_pred = (pred == y_true[:, np.newaxis]).sum(axis=1)
        threshold = (pred.shape[1] - (pred == -1).sum(axis=1, dtype=np.float))
        correct_pred_perc = correct_pred / threshold * 100
        correct_idx, incorrect_idx = correct_pred_perc > 75, correct_pred_perc < 25

        suff_subj_cl1, suff_subj_cl2 = validate_minimum_subjects(correct_idx, incorrect_idx, n_class1)

        cl1_corr_data, cl1_incorr_data = class1_data[correct_idx[:n_class1]], class1_data[incorrect_idx[:n_class1]]
        cl2_corr_data, cl2_incorr_data = class2_data[correct_idx[n_class1:]], class2_data[incorrect_idx[n_class1:]]

        results_cl1, results_cl2 = roi_labels_df.copy(), roi_labels_df.copy()
        p_corr_threshold = P_THRESHOLD / n_ROI

        for ROI in xrange(n_ROI):

            t_cl1_label, t_cl2_label = 'T_value_' + class1, 'T_value_' + class2
            p_cl1_label, p_cl2_label = 'p_value_' + class1, 'p_value_' + class2
            p_corr_cl1_label = p_corr_cl2_label = 'p_corr_threshold'
            sig_cl1_label = sig_cl2_label = 'significant'

            t_cl1, p_cl1 = roi_t_test(ROI, cl1_corr_data, cl1_incorr_data) if suff_subj_cl1 else (np.nan, np.nan)
            t_cl2, p_cl2 = roi_t_test(ROI, cl2_corr_data, cl2_incorr_data) if suff_subj_cl2 else (np.nan, np.nan)

            p_sig_cl1, p_sig_cl2 = True if p_cl1 <= p_corr_threshold else False, \
                                   True if p_cl2 <= p_corr_threshold else False

            results_cl1.loc[ROI, t_cl1_label], results_cl1.loc[ROI, p_cl1_label] = t_cl1, p_cl1
            results_cl1.loc[ROI, p_corr_cl1_label], results_cl1.loc[ROI, sig_cl1_label] = p_corr_threshold, p_sig_cl1
            results_cl2.loc[ROI, t_cl2_label], results_cl2.loc[ROI, p_cl2_label] = t_cl2, p_cl2
            results_cl2.loc[ROI, p_corr_cl2_label], results_cl2.loc[ROI, sig_cl2_label] = p_corr_threshold, p_sig_cl2

        save_results(class1, results_cl1, save_folder, suff_subj_cl1)
        save_results(class2, results_cl2, save_folder, suff_subj_cl2)


if __name__ == '__main__':
    descriptive_t_tests()

