import os.path as osp
from cPickle import dump
from datetime import datetime
from itertools import product
from time import time

import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.externals.joblib import Parallel, delayed
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import data_handling as data_funs
from covariate_selector import CovariateSelector, CovariateScaler
from evaluation_classifier import Evaluater
from feature_selector import FeatureSelector
from structural_covariance import StructuralCovariance


SAVE_DIR = '/data/shared/bvFTD/Machine_Learning/results'
SAVE_DATA = '/data/shared/bvFTD/Machine_Learning/data'
LOAD_DATA = SAVE_DATA

NUM_SAMPLING_ITER = 1000

# CLASSIFICATION = 'FTDvsPsych'
# CLASSIFICATION = 'FTDvsNeurol'
# CLASSIFICATION = 'NeurolvsPsych'
CLASSIFICATION = 'FTDvsRest'

COVARIATES = False
PARCELLATION = False
SMOOTHING = False
# if you only want to do classification with one specific model, select it here
# possible values are: 'RF', 'SVM', or ''
# for random forest grid search will be performed
SINGLE_CLASSIFICATION = 'RF'
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


def inner_loop_iteration(clf, id_train, id_test, X, y, org_inner_train_id, org_inner_test_id, covariates=COVARIATES):
    X_train, y_train = X[id_train], y[id_train]
    X_test, y_test = X[id_test], y[id_test]

    if covariates:
        cov_select = CovariateSelector()
        train_cov, test_cov = cov_select.cov_from_id(org_inner_train_id), cov_select.cov_from_id(org_inner_test_id)
        clf.named_steps['cov'].set_covariates(train_cov, test_cov)

    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)


def check_diff_models(X_inner_cv, y_inner_cv, X_test_outer_cv, org_train_id, org_test_id, n_folds=5,
                      covariates=COVARIATES, structural_covariance=STRUCTURAL_COVARIANCE, parcellation=PARCELLATION):
    clfs, clfs_labels = get_models_to_check(covariates=covariates, structural_covariance=structural_covariance,
                                            parcellation=parcellation)
    # the random state is added to ensure the usage of the same CV/fold order for the models in question to get the
    # 'fairest' comparison possible. To make it explicit we seed the cross-validator with the same random_state
    cv = get_cross_validator(n_folds, random_state=int(time()))

    n_jobs = len(clfs_labels) * n_folds
    print 'Choose best model'
    accuracy = Parallel(n_jobs=n_jobs, verbose=1)(delayed(inner_loop_iteration)(clf, id_train, id_test, X_inner_cv,
                                                                                y_inner_cv, org_train_id[id_train],
                                                                                org_train_id[id_test],
                                                                                covariates=covariates)
                                                  for clf, (id_train, id_test) in product(clfs, cv.split(X_inner_cv,
                                                                                                         y_inner_cv)))
    accuracy = np.array(accuracy)
    id_best_clf = np.argmax([accuracy[i * n_folds:n_folds * (i + 1)].mean() for i in xrange(len(clfs_labels))])
    best_clf = clfs[id_best_clf]
    best_clf_label = clfs_labels[id_best_clf]

    if covariates:
        cov_select = CovariateSelector()
        train_cov, test_cov = cov_select.cov_from_id(org_train_id), cov_select.cov_from_id(org_test_id)
        best_clf.named_steps['cov'].set_covariates(train_cov, test_cov)

    best_clf.fit(X_inner_cv, y_inner_cv)
    y_pred = best_clf.predict(X_test_outer_cv)
    y_score = best_clf.predict_proba(X_test_outer_cv)[:, 1]
    return y_pred, y_score, best_clf_label


def get_models_to_check(covariates=COVARIATES, structural_covariance=STRUCTURAL_COVARIANCE, z_threshold=z_THRESHOLD,
                        parcellation=PARCELLATION):
    if covariates:
        return use_covariates(structural_covariance=structural_covariance, parcellation=parcellation)

    svm = SVC(kernel='linear', probability=True)

    pca = PCA(n_components=0.9)
    pca_svm = Pipeline([('pca', pca), ('svm', SVC(kernel='linear', probability=True))])

    feat_sel = FeatureSelector(z_thresh=z_threshold[parcellation])
    feat_sel_svm = Pipeline([('feat_sel', feat_sel), ('svm', SVC(kernel='linear', probability=True))])

    struct_cov_svm, struct_cov_label = [], []
    if structural_covariance:
        struc_cov = StructuralCovariance()
        svm = SVC(kernel='linear', probability=True)
        struct_cov_svm = [Pipeline([('struc_cov', struc_cov), ('svm', svm)])]
        struct_cov_label = ['struct_cov']

    clfs = [svm, pca_svm, feat_sel_svm] + struct_cov_svm
    clfs_labels = ['svm', 'pca_svm', 'z-thresh_svm'] + struct_cov_label

    return clfs, clfs_labels


def use_covariates(structural_covariance=STRUCTURAL_COVARIANCE, parcellation=PARCELLATION, z_threshold=z_THRESHOLD):
    svm = SVC(kernel='linear', probability=True)
    cov = CovariateScaler()
    svm = Pipeline([('cov', cov), ('svm', svm)])

    pca = PCA(n_components=0.9)
    cov = CovariateScaler()
    pca_svm = Pipeline([('pca', pca), ('cov', cov), ('svm', SVC(kernel='linear', probability=True))])

    feat_sel = FeatureSelector(z_thresh=z_threshold[parcellation])
    cov = CovariateScaler()
    feat_sel_svm = Pipeline([('feat_sel', feat_sel), ('cov', cov), ('svm', SVC(kernel='linear', probability=True))])

    struct_cov_svm, struct_cov_label = [], []
    if structural_covariance:
        struc_cov = StructuralCovariance()
        cov = CovariateScaler()
        svm = SVC(kernel='linear', probability=True)
        struct_cov_svm = [Pipeline([('struc_cov', struc_cov), ('cov', cov), ('svm', svm)])]
        struct_cov_label = ['struct_cov']

    clfs = [svm, pca_svm, feat_sel_svm] + struct_cov_svm
    clfs_labels = ['svm', 'pca_svm', 'z-thresh_svm'] + struct_cov_label
    return clfs, clfs_labels


def single_clf_run(X_train, y_train, X_test, clf_str):
    if clf_str == 'SVM':
        svm = SVC(kernel='linear', probability=True)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        y_score = svm.predict_proba(X_test)[:, svm.classes_ == 1]
        meta_data = svm.coef_
    elif clf_str == 'RF':
        rf_clf = RandomForestClassifier()
        params_rf = {'n_estimators': np.arange(10, 200, 20), 'max_features': ['sqrt', 'log2', 0.25, 0.5, 0.75]}
        cv = get_cross_validator(n_folds=5)
        grid_search = GridSearchCV(rf_clf, params_rf, scoring='accuracy', cv=cv, refit=True, verbose=1, n_jobs=15)
        grid_search.fit(X_train, y_train)
        rf_clf = grid_search.best_estimator_
        y_pred = rf_clf.predict(X_test)
        y_score = rf_clf.predict_proba(X_test)[:, 1]
        meta_data = rf_clf.feature_importances_
    else:
        raise RuntimeError('Not recognized type {}. Currently only "RF" and "SVM" are implemented'.format(clf_str))

    return y_pred, y_score, meta_data


def run_ml(X, y, save_folder, num_resample_rounds=NUM_SAMPLING_ITER, n_folds=5, covariates=COVARIATES,
           structural_covariance=STRUCTURAL_COVARIANCE, parcellation=PARCELLATION,
           single_classification=SINGLE_CLASSIFICATION):
    data_funs.ensure_folder(save_folder)
    evaluator = Evaluater()

    metrics_labels = evaluator.evaluate_labels()
    metrics = np.zeros((n_folds, num_resample_rounds, len(metrics_labels)))
    # initialized to -1: if a subject wasn't chosen in the undersampling for the iteration it will remain -1
    predictions = np.ones((y.size, num_resample_rounds)) * -1

    roc_curves = []
    meta_info_classification = []
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
            org_train_id, org_test_id = id_full_sample[train_id], id_full_sample[test_id]

            if single_classification:
                y_pred, y_score, meta_info = single_clf_run(X_train, y_train, X_test, single_classification)
            else:
                y_pred, y_score, meta_info = different_models_run(X_test, X_train, covariates, org_test_id,
                                                                  org_train_id, parcellation, structural_covariance,
                                                                  y_train)
            meta_info_classification.append(meta_info)
            fpr, tpr, threshold = roc_curve(y_true=y_test, y_score=y_score)
            roc_curves.append([fpr, tpr, threshold])
            metrics[id_iter_cv, id_sampling, :] = evaluator.evaluate_prediction(y_true=y_test, y_pred=y_pred,
                                                                                y_score=y_score)
            predictions[id_full_sample[test_id], id_sampling] = y_pred

            evaluator.print_evaluation()

        t2 = time()
        print 'Run took: {:.2f}min'.format((t2 - t1) / 60.)

    saving_results(metrics, metrics_labels, predictions, roc_curves, meta_info_classification, save_folder,
                   single_classification)


def saving_results(metrics, metrics_labels, predictions, roc_curves, meta_info_classification, save_folder,
                   single_classification):
    print 'Saving data'
    np.savez_compressed(osp.join(save_folder, 'performance_metrics.npz'), metrics=metrics,
                        metrics_labels=metrics_labels)
    np.save(osp.join(save_folder, 'predictions.npy'), predictions)
    with open(osp.join(save_folder, 'roc_curves.pkl'), 'wb') as f:
        dump(roc_curves, f)

    meta_data_name = '{}_meta.pkl'.format(single_classification) if single_classification else 'best_model_labels.pkl'
    with open(osp.join(save_folder, meta_data_name), 'wb') as f:
        dump(meta_info_classification, f)


def different_models_run(X_test, X_train, covariates, org_test_id, org_train_id, parcellation, structural_covariance,
                         y_train):
    y_pred, y_score, best_model_label = check_diff_models(X_train, y_train, X_test, org_train_id, org_test_id,
                                                          covariates=covariates,
                                                          structural_covariance=structural_covariance,
                                                          parcellation=parcellation)
    print 'Best model: {}'.format(best_model_label)
    return y_pred, y_score, best_model_label


def run_classification(X, y, save_folder, label='', smoothing=SMOOTHING, covariates=COVARIATES,
                       parcellation=PARCELLATION, num_sampling_rounds=NUM_SAMPLING_ITER,
                       structural_covariance=STRUCTURAL_COVARIANCE, single_classification=SINGLE_CLASSIFICATION):
    print
    print
    print label
    print 'Smoothing enabled: {}'.format(smoothing)
    print 'Covariates enabled: {}'.format(covariates)
    print 'Parcellation enabled: {}'.format(parcellation)
    print 'Structural covariance enabled: {}'.format(structural_covariance)
    print 'Single classification: {}'.format(single_classification)
    print 'Started:', datetime.now()
    t_start = time()
    run_ml(X, y, save_folder, num_resample_rounds=num_sampling_rounds, structural_covariance=structural_covariance,
           parcellation=parcellation, single_classification=single_classification)
    t_end = time()
    print 'Time taken: {:.2f}h'.format((t_end - t_start) / 3600.)


def run(save_data=SAVE_DATA, load_data=LOAD_DATA, parcellation=PARCELLATION, smoothing=SMOOTHING,
        classification=CLASSIFICATION, covariates=COVARIATES, save_dir=SAVE_DIR, num_resample_rounds=NUM_SAMPLING_ITER,
        structural_covariance=STRUCTURAL_COVARIANCE, single_classification=SINGLE_CLASSIFICATION):
    data_funs.ensure_folder(save_data)
    X, y = data_funs.create_data_matrices(save_data, load_path=load_data, parcellation=parcellation,
                                          smoothing=smoothing, classification_type=classification)

    cov_suffix = '_covariates' if covariates else '_no_covariates'
    single_clf = '_{}'.format(single_classification) if single_classification else ''
    save_dir_path = data_funs.create_file_name(parcellation, smoothing,
                                               initial_identifier=save_dir + '_' + classification + single_clf,
                                               additional_identifier=cov_suffix,
                                               file_extension='')
    run_classification(X, y, save_dir_path, classification, smoothing=smoothing, covariates=covariates,
                       parcellation=parcellation, num_sampling_rounds=num_resample_rounds,
                       structural_covariance=structural_covariance, single_classification=single_classification)

if __name__ == '__main__':
    run()
