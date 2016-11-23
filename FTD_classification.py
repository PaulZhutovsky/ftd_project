import os.path as osp
from cPickle import dump
from datetime import datetime
from itertools import product
from time import time

import numpy as np
from feature_selector import FeatureSelector
from imblearn.under_sampling import RandomUnderSampler
from sklearn.externals.joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import data_handling as data_funs
from evaluation_classifier import Evaluater
from structural_covariance import StructuralCovariance
from covariate_selector import CovariateSelector

SAVE_DIR = '/data/shared/bvFTD/Machine_Learning/results'
SAVE_DATA = '/data/shared/bvFTD/Machine_Learning/data'
LOAD_DATA = SAVE_DATA

NUM_SAMPLING_ITER = 1000

CLASSIFICATION = 'FTDvsPsych'
# CLASSIFICATION = 'FTDvsNeurol'
# CLASSIFICATION = 'NeurolvsPsych'
# CLASSIFICATION = 'FTDvsRest'

COVARIATES = True
PARCELLATION = False
SMOOTHING = True
# structural covariance can/will be only used in the case of parcellation the syntax below ensures that even if it is
# switched on it will only be used if PARCELLATION is True as well
STRUCTURAL_COVARIANCE = False & PARCELLATION

# In the ATLAS case (PARCELLATION=True) the z-threshold is way to strong so we will have to adjust it
z_THRESHOLD = {True: 1.5, False: 3.5}


def get_sampling_method(X, y):
    under_sampler = RandomUnderSampler(return_indices=True, replacement=False)
    under_sampler.fit(X, y)
    return under_sampler


def get_cross_validator(n_folds, **kwargs):
    return StratifiedKFold(n_splits=n_folds, shuffle=True, **kwargs)


def sample(X, y, sampler):
    return sampler.sample(X, y)


def inner_loop_iteration(clf, id_train, id_test, X, y, use_covariates=COVARIATES):
    X_train, y_train = X[id_train], y[id_train]
    X_test, y_test = X[id_test], y[id_test]

    # TODO: Covariates
    if use_covariates:
        clf.named_steps['cov'].set_ids(id_train, id_test)

    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)


def check_diff_models(train_id, test_id, X_inner_cv, y_inner_cv, X_test_outer_cv, n_folds=5):
    clfs, clfs_labels = get_models_to_check()
    # the random state is added to ensure the usage of the same CV/fold order for the models in question to get the
    # 'fairest' comparison possible. To make it explicit we seed the cross-validator with the same random_state
    cv = get_cross_validator(n_folds, random_state=int(time))

    n_jobs = len(clfs_labels) * n_folds
    print 'Choose best model'
    accuracy = Parallel(n_jobs=n_jobs, verbose=1)(delayed(inner_loop_iteration)(clf, id_train, id_test, X_inner_cv,
                                                                                y_inner_cv, use_covariates=COVARIATES)
                                                  for clf, (id_train, id_test) in product(clfs, cv.split(X_inner_cv,
                                                                                                         y_inner_cv)))
    accuracy = np.array(accuracy)
    id_best_clf = np.argmax([accuracy[i * n_folds:n_folds * (i + 1)].mean() for i in xrange(len(clfs_labels))])
    best_clf = clfs[id_best_clf]
    best_clf_label = clfs_labels[id_best_clf]

    # TODO: Covariates
    if COVARIATES:
        best_clf.named_steps['cov'].set_ids(train_id, test_id)

    best_clf.fit(X_inner_cv, y_inner_cv)
    y_pred = best_clf.predict(X_test_outer_cv)
    y_score = best_clf.predict_proba(X_test_outer_cv)[:, best_clf.n_classes_ == 1]
    return y_pred, y_score, best_clf_label


def get_models_to_check():
    if COVARIATES:
        return use_covariates()

    svm = SVC(kernel='linear', probability=True)

    pca = PCA(n_components=0.9)
    pca_svm = Pipeline([('pca', pca), ('svm', SVC(kernel='linear', probability=True))])

    feat_sel = FeatureSelector(z_thresh=z_THRESHOLD[PARCELLATION])
    feat_sel_svm = Pipeline([('feat_sel', feat_sel), ('svm', SVC(kernel='linear', probability=True))])

    struct_cov_svm, struct_cov_label = [], []
    if STRUCTURAL_COVARIANCE:
        struc_cov = StructuralCovariance()
        svm = SVC(kernel='linear', probability=True)
        struct_cov_svm = [Pipeline([('struc_cov', struc_cov), ('svm', svm)])]
        struct_cov_label = ['struct_cov']

    clfs = [svm, pca_svm, feat_sel_svm] + struct_cov_svm
    clfs_labels = ['svm', 'pca_svm', 'z-thresh_svm'] + struct_cov_label

    return clfs, clfs_labels


def use_covariates():
    """
    TODO: NOT WORKING?FINISHED YET!
    """
    svm = SVC(kernel='linear', probability=True)
    cov = CovariateSelector()
    svm = Pipeline([('cov', cov), ('svm', svm)])
    pca = PCA(n_components=0.9)
    cov = CovariateSelector()
    pca_svm = Pipeline([('pca', pca), ('cov', cov), ('svm', SVC(kernel='linear', probability=True))])
    feat_sel = FeatureSelector(z_thresh=z_THRESHOLD[PARCELLATION])
    cov = CovariateSelector()
    feat_sel_svm = Pipeline([('feat_sel', feat_sel), ('cov', cov), ('svm', SVC(kernel='linear', probability=True))])
    struct_cov_svm = []
    struct_cov_label = []
    if STRUCTURAL_COVARIANCE:
        struc_cov = StructuralCovariance()
        svm = SVC(kernel='linear', probability=True)
        struct_cov_svm = [Pipeline([('struc_cov', struc_cov), ('svm', svm)])]
        struct_cov_label = ['struct_cov']
    clfs = [svm, pca_svm, feat_sel_svm] + struct_cov_svm
    clfs_labels = ['svm', 'pca_svm', 'z-thresh_svm'] + struct_cov_label
    return clfs, clfs_labels


def run_ml(X, y, save_folder, num_resample_rounds=NUM_SAMPLING_ITER, n_folds=5):
    data_funs.ensure_folder(save_folder)
    evaluator = Evaluater()

    metrics_labels = evaluator.evaluate_labels()
    metrics = np.zeros((n_folds, num_resample_rounds, len(metrics_labels)))
    # initialized to -1: if a subject wasn't chosen in the undersampling for the iteration it will remain -1
    predictions = np.ones((y.size, num_resample_rounds)) * -1

    roc_curves = []
    best_clf_labels = []
    sampling_method = get_sampling_method(X, y)

    for id_sampling in xrange(num_resample_rounds):
        print 'Sampling Run: {}/{}'.format(id_sampling + 1, num_resample_rounds)

        t1 = time()
        X_sample, y_sample, id_full_sample = sample(X, y, sampling_method)
        X_sample = data_funs.apply_masking(X_sample)
        cv = get_cross_validator(n_folds)

        for id_iter_cv, (train_id, test_id) in enumerate(cv.split(X_sample, y_sample)):
            print '{}/{}'.format(id_iter_cv + 1, n_folds)
            print '#Train: {} (class1: {}) #Test: {} (class1: {})'.format(train_id.size, y[train_id].sum(),
                                                                          test_id.size, y[test_id].sum())

            X_train, y_train = X_sample[train_id], y_sample[train_id]
            X_test, y_test = X_sample[test_id], y_sample[test_id]

            y_pred, y_score, best_model_label = check_diff_models(train_id, test_id, X_train, y_train, X_test)
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

    print 'Saving data'
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
    print 'Parcellation enabled: {}'.format(PARCELLATION)
    print 'Started:', datetime.now()
    t_start = time()
    run_ml(X, y, save_folder)
    t_end = time()
    print 'Time taken: {:.2f}h'.format((t_end - t_start) / 3600.)


def run():
    data_funs.ensure_folder(SAVE_DATA)
    X, y = data_funs.create_data_matrices(SAVE_DATA, load_path=LOAD_DATA, parcellation=PARCELLATION,
                                          smoothing=SMOOTHING, classification_type=CLASSIFICATION)

    cov_suffix = '_covariates' if COVARIATES else '_no_covariates'
    save_dir_path = data_funs.create_file_name(PARCELLATION, SMOOTHING,
                                               initial_identifier=SAVE_DIR + '_' + CLASSIFICATION,
                                               additional_identifier=cov_suffix,
                                               file_extension='')
    run_classification(X, y, save_dir_path, CLASSIFICATION)

if __name__ == '__main__':
    run()
