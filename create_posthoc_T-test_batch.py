from glob import glob
from os import path as osp, mkdir, chdir
import matlab.engine
import numpy as np
import ipdb


REG_PATTERN = 'results_*_no_Cov_no_Parc_Smoothed'
SMOOTHING = True

CLASSES_TRANSFORM = {'ftd': 'bvFTD', 'neurol': 'neurological', 'psych': 'psychiatric'}
PARENT_DIR_SMOOTHED = '/data/shared/bvFTD/VBM/default_LOF5/data'
PARENT_DIR_UNSMOOTHED = '/data/shared/bvFTD/VBM/default_non_modulated_LOF5/data'
PARENT_DIR_PARC = ''

RESULTS_FOLDER = sorted(glob(osp.join('/data/shared/bvFTD/Machine_Learning', REG_PATTERN)))


def load_all_data():
    if SMOOTHING:
        ftd = sorted(glob(osp.join(PARENT_DIR_SMOOTHED, CLASSES_TRANSFORM['ftd'], '*', 'structural', 'mri', 'smw*')))
        neurol = sorted(glob(osp.join(PARENT_DIR_SMOOTHED, CLASSES_TRANSFORM['neurol'], '*', 'structural', 'mri', 'smw*')))
        psych = sorted(glob(osp.join(PARENT_DIR_SMOOTHED, CLASSES_TRANSFORM['psych'], '*', 'structural', 'mri', 'smw*')))
    else:
        ftd = sorted(glob(osp.join(PARENT_DIR_UNSMOOTHED, CLASSES_TRANSFORM['ftd'], '*', 'structural', 'wc1*')))
        neurol = sorted(glob(osp.join(PARENT_DIR_UNSMOOTHED, CLASSES_TRANSFORM['neurol'], '*', 'structural', 'wc1*')))
        psych = sorted(glob(osp.join(PARENT_DIR_UNSMOOTHED, CLASSES_TRANSFORM['psych'], '*', 'structural', 'wc1*')))

    all_paths, n_paths = ftd + neurol + psych, len(ftd + neurol + psych)
    data_idx_map = dict()
    data_idx_map['ftd'] = range(0, len(ftd))
    data_idx_map['neurol'] = range(len(ftd), len(ftd) + len(neurol))
    data_idx_map['psych'] = range(len(ftd) + len(neurol), n_paths)
    data_idx_map['rest'] = range(len(ftd), n_paths)
    return np.array(all_paths, dtype=basestring), data_idx_map


def run():
    all_paths, data_idx_map = load_all_data()
    eng = matlab.engine.start_matlab()
    chdir('/data/shared/bvFTD/matlab/spm_batch/')

    for i in xrange(len(RESULTS_FOLDER)):
        save_folder = osp.join(RESULTS_FOLDER[i], 'descriptive_T-tests')
        if not osp.exists(save_folder):
            mkdir(save_folder)

        classifiers = RESULTS_FOLDER[i].split('results_')[1]
        class1, class2 = classifiers.split('_')[:2]
        class1_idx, class2_idx = data_idx_map[class1], data_idx_map[class2]
        class1_paths, class2_paths = all_paths[class1_idx], all_paths[class2_idx]
        n_class1, n_class2 = len(class1_idx), len(class2_idx)

        pred = np.load(osp.join(RESULTS_FOLDER[i], 'predictions.npy')).astype(np.int)
        y_true = np.concatenate((np.ones(n_class1), np.zeros(pred.shape[0] - n_class1))).astype(np.int)
        correct_pred = (pred == y_true[:, np.newaxis]).sum(axis=1)
        threshold = (pred.shape[1] - (pred == -1).sum(axis=1, dtype=np.float))
        correct_pred_perc = correct_pred / threshold * 100
        correct_idx, incorrect_idx = correct_pred_perc > 75, correct_pred_perc < 25

        class1_correctly_classified_paths = set(class1_paths[correct_idx[:n_class1]])
        class1_incorrectly_classified_paths = set(class1_paths[incorrect_idx[:n_class1]])
        class2_correctly_classified_paths = set(class2_paths[correct_idx[n_class1:]])
        class2_incorrectly_classified_paths = set(class2_paths[incorrect_idx[n_class1:]])

        eng.create_Ttest_batch(save_folder,
                               class1_correctly_classified_paths,
                               class1_incorrectly_classified_paths,
                               class2_correctly_classified_paths,
                               class2_incorrectly_classified_paths,
                               nargout=0)


if __name__ == '__main__':
    run()
