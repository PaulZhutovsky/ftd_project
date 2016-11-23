import os
from itertools import product
from glob import glob
from os import path as osp

import nibabel as nib
import numpy as np
import pandas as pd

# avoid cyclic imports
import parcellate_NIFTI
import FTD_classification as clf_script


PARENT_DIR_SMOOTHED = '/data/shared/bvFTD/VBM/default_LOF5/data'
PARENT_DIR_UNSMOOTHED = '/data/shared/bvFTD/VBM/default_non_modulated_LOF5/data'

CLASSES_TRANSFORM = {'ftd': 'bvFTD', 'neurol': 'neurological', 'psych': 'psychiatric'}
SIZE_VOXELS = 121 * 145 * 121


def create_file_name(parcellation, smoothing, initial_identifier='data_set', additional_identifier='',
                     file_extension='.npy'):
    atlas_suffix = '_atlas' if parcellation else '_voxel_wise'
    smoothing_suffix = '_smoothed' if smoothing else '_unsmoothed'
    return initial_identifier + atlas_suffix + smoothing_suffix + additional_identifier + file_extension


def ensure_folder(folder_dir):
    if not osp.exists(folder_dir):
        os.makedirs(folder_dir)


def get_file_path(class_folder, smoothing=False):
    if smoothing:
        return sorted(glob(osp.join(PARENT_DIR_SMOOTHED, class_folder, '*', 'structural', 'mri', 'smw*')))
    return sorted(glob(osp.join(PARENT_DIR_UNSMOOTHED, class_folder, '*', 'structural', 'wc1*')))


def load_data(data_path, keepdims=False):
    if keepdims:
        return nib.load(data_path).get_data().astype(np.float64)
    return nib.load(data_path).get_data().astype(np.float64).ravel()


def load_all_data(files_to_load, parcellation=False):
    if parcellation:
        return parcellate_NIFTI.run_parcellation(files_to_load)

    data = np.zeros((len(files_to_load), SIZE_VOXELS))
    for i, file_path in enumerate(files_to_load):
        data[i, :] = load_data(file_path)
    return data


def create_labels(class1_num, class2_num, class3_num):
    """
    General idea: create a boolean matrix of class1_num + class2_num + class3_num x 4 elements.
     - First column will be True for the first class1_num elements
     - Second column will be True for the class1_num until class1_num + class2_num elements
     - Third column will be True for the class1_num + class2_num until class1_num + class2_num + class3_num (end)
       elements
     - Forth column will be True for all elements starting at class1_num

    :param class1_num:      e.g. amount of FTD patients
    :param class2_num:      e.g. amount of neurological patients
    :param class3_num:      e.g. amount of psychiatric patients
    :return:                boolean array
    """
    y = np.zeros((class1_num + class2_num + class3_num, 4), dtype=np.bool)
    y[:class1_num, 0] = True
    y[class1_num:class1_num + class2_num, 1] = True
    y[class1_num + class2_num:class1_num + class2_num + class3_num, 2] = True
    y[class1_num:, 3] = True
    return y


def create_classification_data(data, class_labels_df, label1, label2):
    class1_labels = class_labels_df[label1].values.astype(np.bool)
    class2_labels = class_labels_df[label2].values.astype(np.bool)
    size_class1 = class1_labels.sum()
    size_class2 = class2_labels.sum()

    X = data[np.any((class1_labels, class2_labels), axis=0)]
    y = np.concatenate((np.ones(size_class1, dtype=np.int), np.zeros(size_class2, dtype=np.int)))
    return X, y


def extract_subject_ids(path_files):
    path_files = np.array(path_files)
    path_files = np.char.asarray(np.char.split(path_files, os.sep))
    return path_files[np.char.isdigit(path_files)]


def create_data_matrices(save_path, load_path='', parcellation=False, smoothing=False,
                         classification_type='FTDvsPsych'):
    data_filename = create_file_name(parcellation, smoothing)

    if load_path:
        data = np.load(osp.join(load_path, data_filename))
        class_labels_df = pd.read_csv(osp.join(load_path, 'class_labels.csv'))
    else:
        class_labels_df, data = initialize_and_load_data(data_filename, parcellation, save_path, smoothing)

    if classification_type == 'FTDvsPsych':
        X, y = create_classification_data(data, class_labels_df, 'ftd', 'psych')
    elif classification_type == 'FTDvsNeurol':
        X, y = create_classification_data(data, class_labels_df, 'ftd', 'neurol')
    elif classification_type == 'NeurolvsPsych':
        X, y = create_classification_data(data, class_labels_df, 'neurol', 'psych')
    elif classification_type == 'FTDvsRest':
        X, y = create_classification_data(data, class_labels_df, 'ftd', 'rest')
    else:
        raise RuntimeError('Unrecognized classification: {}. '.format(classification_type) +
                           'Possible values are: "FTDvsPsych", "FTDvsNeurol", "NeurolvsPsych", "FTDvsRest"')
    return X, y


def initialize_and_load_data(data_filename, parcellation, save_path, smoothing):
    ftd_files = get_file_path(CLASSES_TRANSFORM['ftd'], smoothing=smoothing)
    neurological_files = get_file_path(CLASSES_TRANSFORM['neurol'], smoothing=smoothing)
    psychiatry_files = get_file_path(CLASSES_TRANSFORM['psych'], smoothing=smoothing)

    all_files_to_load = ftd_files + neurological_files + psychiatry_files
    size_classes = [len(ftd_files), len(neurological_files), len(psychiatry_files)]

    data = load_all_data(all_files_to_load, parcellation=parcellation)
    subj_ids = extract_subject_ids(all_files_to_load)
    class_labels = create_labels(*size_classes)
    subj_info = np.column_stack((subj_ids, class_labels))

    class_labels_df = pd.DataFrame(data=subj_info, columns=['subj_id', 'ftd', 'neurol', 'psych', 'rest'])

    np.save(osp.join(save_path, data_filename), data)
    class_labels_df.to_csv(osp.join(save_path, 'class_labels.csv'), index=False)
    return class_labels_df, data


def apply_masking(X):
    # since we don't have a mask (yet) we are basically comparing our data across subjects for each voxel and remove all
    # voxels which are 0 across ALL subjects
    id_keep = ~np.all(X == 0, axis=0)
    return X[:, id_keep]


def run():
    """
    Creates ALL data sets which we are currently using
    """
    smoothing_or_not = [True, False]
    parcellation_or_not = [True, False]
    # we just need one since the matrix which will be created contains all of them
    clf_type = 'FTDvsPsych'

    for smoothing, parcellation in product(smoothing_or_not, parcellation_or_not):
        print 'Smoothing: {} Parcellation: {}'.format(smoothing, parcellation)
        _, _ = create_data_matrices(clf_script.SAVE_DATA, load_path='', parcellation=parcellation, smoothing=smoothing,
                                    classification_type=clf_type)


if __name__ == '__main__':
    run()