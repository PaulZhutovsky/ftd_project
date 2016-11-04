import os.path as osp
from cPickle import dump
from datetime import datetime
from itertools import product
from time import time

import numpy as np
from feature_selector import FeatureSelector
from imblearn.under_sampling import RandomUnderSampler
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from data_handling import ensure_folder, create_data_matrices, apply_masking
from evaluation_classifier import Evaluater

SAVE_DIR = '/data/shared/bvFTD/Machine_Learning/results_ftr_sel'
SAVE_DATA = '/data/shared/bvFTD/Machine_Learning/data'
LOAD_DATA = SAVE_DATA

NUM_SAMPLING_ITER = 2000

# CLASSIFICATION = 'FTDvsPsych'
CLASSIFICATION = 'FTDvsNeurol'
# CLASSIFICATION = 'NeurolvsPsych'
# CLASSIFICATION = 'FTDvsRest'

COVARIATES = False
NUM_NORMALIZED_FEATURES = 3


def get_sampling_method(X, y):
    under_sampler = RandomUnderSampler(return_indices=True, replacement=False)
    under_sampler.fit(X, y)
    return under_sampler


def get_cross_validator(n_folds):
    return StratifiedKFold(n_splits=n_folds, shuffle=True)


def sample(X, y, sampler):
    return sampler.sample(X, y)


def inner_loop_iteration(clf, id_train, id_test, X, y):
    X_train, y_train = X[id_train], y[id_train]
    X_test, y_test = X[id_test], y[id_test]

    if COVARIATES:
        inner_featureNormalizer = MinMaxScaler()
        inner_featureNormalizer.fit(X_train[:, -NUM_NORMALIZED_FEATURES:])
        # Parallel processing sets matrix flags to read-only, change to writeable to allow assignment
        X_train.flags.writeable = X_test.flags.writeable = True
        X_train[:, -NUM_NORMALIZED_FEATURES:] = inner_featureNormalizer.transform(X_train[:, -NUM_NORMALIZED_FEATURES:])
        X_test[:, -NUM_NORMALIZED_FEATURES:] = inner_featureNormalizer.transform(X_test[:, -NUM_NORMALIZED_FEATURES:])

    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)


def get_models_to_check():
    pca = PCA(n_components=0.9)
    pca_svm = Pipeline([('pca', pca), ('svm', SVC(kernel='linear'))])

    feat_sel = FeatureSelector()
    feat_sel_svm = Pipeline([('feat_sel', feat_sel), ('svm', SVC(kernel='linear'))])

    clfs = [SVC(kernel='linear'), pca_svm, feat_sel_svm]
    clfs_labels = ['svm', 'pca_svm', 'z-thresh_svm']
    return clfs, clfs_labels


def check_diff_models(X_inner_cv, y_inner_cv, X_test_outer_cv, n_folds=5):
    clfs, clfs_labels = get_models_to_check()
    cv = get_cross_validator(n_folds)

    n_jobs = len(clfs_labels) * n_folds
    print 'Choose best model'
    accuracy = Parallel(n_jobs=n_jobs, verbose=1)(delayed(inner_loop_iteration)(clf, id_train, id_test, X_inner_cv,
                                                                                y_inner_cv)
                                                  for clf, (id_train, id_test) in product(clfs, cv.split(X_inner_cv,
                                                                                                         y_inner_cv)))
    accuracy = np.array(accuracy)
    id_best_clf = np.argmax([accuracy[i*n_folds:n_folds*(i+1)].mean() for i in xrange(len(clfs_labels))])
    best_clf = clfs[id_best_clf]
    best_clf_label = clfs_labels[id_best_clf]

    if COVARIATES:
        outer_featureNormalizer = MinMaxScaler()
        outer_featureNormalizer.fit(X_inner_cv[:, -NUM_NORMALIZED_FEATURES:])
        X_inner_cv[:, -NUM_NORMALIZED_FEATURES:] = \
            outer_featureNormalizer.transform(X_inner_cv[:, -NUM_NORMALIZED_FEATURES:])
        X_test_outer_cv[:, -NUM_NORMALIZED_FEATURES:] = \
            outer_featureNormalizer.transform(X_test_outer_cv[:, -NUM_NORMALIZED_FEATURES:])

    best_clf.fit(X_inner_cv, y_inner_cv)
    y_pred = best_clf.predict(X_test_outer_cv)
    y_score = best_clf.decision_function(X_test_outer_cv)
    return y_pred, y_score, best_clf_label


def run_ml(X, y, save_folder, num_resample_rounds=NUM_SAMPLING_ITER, n_folds=5):
    ensure_folder(save_folder)
    evaluator = Evaluater()

    metrics_labels = evaluator.evaluate_labels()
    metrics = np.zeros((n_folds, num_resample_rounds, len(metrics_labels)))
    predictions = np.ones((y.size, num_resample_rounds)) * -1

    roc_curves = []
    best_clf_labels = []

    sampling_method = get_sampling_method(X, y)

    for id_sampling in xrange(num_resample_rounds):
        print 'Sampling Run: {}/{}'.format(id_sampling + 1, num_resample_rounds)
        t1 = time()
        X_sample, y_sample, id_full_sample = sample(X, y, sampling_method)
        X_sample = apply_masking(X_sample)

        cv = get_cross_validator(n_folds)

        for id_iter_cv, (train_id, test_id) in enumerate(cv.split(X_sample, y_sample)):
            print '{}/{}'.format(id_iter_cv + 1, n_folds)
            print '#Train: {} ({}) #Test: {} ({})'.format(train_id.size, y[train_id].sum(),
                                                          test_id.size, y[test_id].sum())

            X_train, y_train = X_sample[train_id], y_sample[train_id]
            X_test, y_test = X_sample[test_id], y_sample[test_id]

            y_pred, y_score, best_model_label = check_diff_models(X_train, y_train, X_test)
            print 'Best model: {}'.format(best_model_label)
            best_clf_labels.append(best_model_label)

            fpr, tpr, threshold = roc_curve(y_true=y_test, y_score=y_score)
            roc_curves.append([fpr, tpr, threshold])
            metrics[id_iter_cv, id_sampling, :] = evaluator.evaluate_prediction(y_true=y_test, y_pred=y_pred,
                                                                                y_score=y_score)
            predictions[id_full_sample[test_id], id_sampling] = y_pred

            evaluator.print_evaluation()

        t2 = time()
        print 'Run took: {:.2f}min'.format((t2 - t1) / 60.)
    np.savez_compressed(osp.join(save_folder, 'performance_metrics.npz'), metrics=metrics,
                        metrics_labels=metrics_labels)
    np.save(osp.join(save_folder, 'predictions.npy'), predictions)

    with open(osp.join(save_folder, 'roc_curves.pkl'), 'wb') as f:
        dump(roc_curves, f)

    with open(osp.join(save_folder, 'best_model_labels.pkl'), 'wb') as f:
        dump(best_clf_labels, f)


def run_classification(X, y, save_folder, label=''):
    print
    print
    print label
    print 'Covariates enabled: {}'.format(COVARIATES)
    print 'Started:', datetime.now()
    t_start = time()
    run_ml(X, y, save_folder)
    t_end = time()
    print 'Time taken: {:.2f}h'.format((t_end - t_start) / 3600.)


def run():
    ensure_folder(SAVE_DATA)
    X_ftd_neurol, y_ftd_neurol, X_ftd_psych, y_ftd_psych, X_neurol_psych, y_neurol_psych, X_ftd_rest, y_ftd_rest = \
        create_data_matrices(SAVE_DATA, load_path=LOAD_DATA, covariates=COVARIATES)

    if COVARIATES:
        save_dir_suffix = '_with_Cov'
    else:
        save_dir_suffix = '_no_Cov'

    if CLASSIFICATION == 'FTDvsPsych':
        run_classification(X_ftd_psych, y_ftd_psych, SAVE_DIR + '_ftd_psych'
                           + save_dir_suffix, 'Ftd vs. Psych')
    elif CLASSIFICATION == 'FTDvsNeurol':
        run_classification(X_ftd_neurol, y_ftd_neurol, SAVE_DIR + '_ftd_neurol'
                           + save_dir_suffix, 'Ftd vs. Neurological')
    elif CLASSIFICATION == 'NeurolvsPsych':
        run_classification(X_neurol_psych, y_neurol_psych, SAVE_DIR + '_neurol_psych'
                           + save_dir_suffix, 'Neurological vs. Psych')
    else:
        run_classification(X_ftd_rest, y_ftd_rest, SAVE_DIR + '_ftd_rest' + save_dir_suffix, 'Ftd vs. Rest')


if __name__ == '__main__':
    run()
