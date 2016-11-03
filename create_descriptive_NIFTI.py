from glob import glob
from os import path as osp, mkdir

import nibabel as nib
import numpy as np
from feature_selector import FeatureSelector

CLASSES_TRANSFORM = {'ftd': 'bvFTD', 'neurol': 'neurological', 'psych': 'psychiatric'}
PARENT_DIR = '/data/shared/bvFTD/VBM/default/data'
RESULTS_FOLDER = sorted(glob(osp.join('/data/shared/bvFTD/Machine_Learning', 'results_ftr_sel_ftd*')))
AFFINE_MATRIX = nib.load('/data/shared/bvFTD/VBM/default/data/'
                         'bvFTD/4908/structural/mri/smwp14908_T1_reoriented_time01.nii').affine

Z_TRESHOLDS = [2., 2.5, 3., 3.5, 4.]
SIZE_VOXELS = [121, 145, 121]
TOTAL_VOXELS = 121 * 145 * 121
threshold_correct = 0.5


def load_all_data():
    ftd = sorted(glob(osp.join(PARENT_DIR, CLASSES_TRANSFORM['ftd'], '*', 'structural', 'mri', 'smw*')))
    neurol = sorted(glob(osp.join(PARENT_DIR, CLASSES_TRANSFORM['neurol'], '*', 'structural', 'mri', 'smw*')))
    psych = sorted(glob(osp.join(PARENT_DIR, CLASSES_TRANSFORM['psych'], '*', 'structural', 'mri', 'smw*')))
    all_paths, n_paths = ftd + neurol + psych, len(ftd + neurol + psych)

    data_idx_map = dict()
    data_idx_map['ftd'] = range(0, len(ftd))
    data_idx_map['neurol'] = range(len(ftd), len(ftd) + len(neurol))
    data_idx_map['psych'] = range(len(ftd) + len(neurol), n_paths)
    data_idx_map['rest'] = range(len(ftd), n_paths)

    data = np.zeros(np.append(SIZE_VOXELS, n_paths))
    for i in xrange(n_paths):
        data[..., i] = nib.load(all_paths[i]).get_data().astype(np.float64)
    return data, data_idx_map


def save_NIFTI(image_data, filepath):
    image = nib.Nifti1Image(image_data, affine=AFFINE_MATRIX)
    nib.save(image, filepath)


def create_descriptive_NIFTIs(correctly_classified, incorrectly_classified, save_folder, filename_prefix):
    correct_mu, incorrect_mu = np.mean(correctly_classified, axis=3), np.mean(incorrectly_classified, axis=3)
    correct_std, incorrect_std = np.std(correctly_classified, axis=3), np.std(incorrectly_classified, axis=3)
    abs_diff = abs(correct_mu - incorrect_mu)
    save_NIFTI(correct_mu, osp.join(save_folder, filename_prefix + 'correctly_classified__mean'))
    save_NIFTI(incorrect_mu, osp.join(save_folder, filename_prefix + 'incorrectly_classified__mean'))
    save_NIFTI(correct_std, osp.join(save_folder, filename_prefix + 'correctly_classified__std'))
    save_NIFTI(incorrect_std, osp.join(save_folder, filename_prefix + 'incorrectly_classified__std'))
    save_NIFTI(abs_diff, osp.join(save_folder, filename_prefix + 'abs_diff'))


def create_z_tresh_NIFTI(X, y_true, z_tresholds, save_folder):
    shape = np.shape(X)
    X_temp = np.zeros((shape[-1], TOTAL_VOXELS,))
    for i in xrange(shape[-1]):
        X_temp[i] = X[..., i].ravel()
    for z in z_tresholds:
        feat_sel = FeatureSelector(z_thresh=z)
        feat_sel.fit(X_temp, y_true)
        features = feat_sel.chosen_ftrs.astype(float)
        z_tresh_data = features.reshape(shape[:-1])
        filename = str(z).replace('.', ',') + '_Z_tresholded'
        save_NIFTI(z_tresh_data, osp.join(save_folder, filename))


def run():
    data, data_idx_map = load_all_data()

    for i in xrange(len(RESULTS_FOLDER)):
        save_folder = osp.join(RESULTS_FOLDER[i], 'descriptive_NIFTIs')

        if not osp.exists(save_folder):
            mkdir(save_folder)

        classifiers = RESULTS_FOLDER[i].split('results_ftr_sel_')[1]
        class1, class2 = classifiers.split('_')[:2]
        class1_idx, class2_idx = data_idx_map[class1], data_idx_map[class2]
        class1_data, class2_data = data[..., class1_idx], data[..., class2_idx]
        n_class1, n_class2 = len(class1_idx), len(class2_idx)

        pred = np.load(osp.join(RESULTS_FOLDER[i], 'predictions.npy')).astype(np.int)
        y_true = np.concatenate((np.ones(n_class1), np.zeros(pred.shape[0] - n_class1))).astype(np.int)
        correct_pred = (pred == y_true[:, np.newaxis]).sum(axis=1)
        threshold = (pred.shape[1] - (pred == -1).sum(axis=1, dtype=np.float))
        correct_pred_perc = correct_pred / threshold * 100
        correct_idx, incorrect_idx = correct_pred_perc > 75, correct_pred_perc < 25

        class1_correctly_classified_data = class1_data[..., correct_idx[:n_class1]]
        class1_incorrectly_classified_data = class1_data[..., incorrect_idx[:n_class1]]
        class2_correctly_classified_data = class2_data[..., correct_idx[n_class1:]]
        class2_incorrectly_classified_data = class2_data[..., incorrect_idx[n_class1:]]

        X = np.concatenate((class1_data, class2_data), axis=3)

        create_z_tresh_NIFTI(X, y_true, Z_TRESHOLDS, save_folder)

        create_descriptive_NIFTIs(class1_correctly_classified_data, class1_incorrectly_classified_data, save_folder,
                                  'class1_')
        create_descriptive_NIFTIs(class2_correctly_classified_data, class2_incorrectly_classified_data, save_folder,
                                  'class2_')


if __name__ == '__main__':
    run()
