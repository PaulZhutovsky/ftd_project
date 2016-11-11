import os
from glob import glob
from os import path as osp

import nibabel as nib
import numpy as np
import pandas as pd

from parcellate_NIFTI import parcellate_nifti

PARENT_DIR = '/data/shared/bvFTD/VBM/default_LOF5/data'
CLASSES_TRANSFORM = {'ftd': 'bvFTD', 'neurol': 'neurological', 'psych': 'psychiatric'}
SIZE_VOXELS = 121 * 145 * 121

ftd_csv_file = '/data/shared/bvFTD/Machine_Learning/data/AMC_VUMC_bvFTD.csv'
TIV_csv_file = '/data/shared/bvFTD/Machine_Learning/data/TIV.csv'


def ensure_folder(folder_dir):
    if not osp.exists(folder_dir):
        os.makedirs(folder_dir)


def get_file_path(class_folder):
    return sorted(glob(osp.join(PARENT_DIR, class_folder, '*', 'structural', 'mri', 'smw*')))


def load_data(data_path):
    return nib.load(data_path).get_data().astype(np.float64).ravel()


def extract_IDs(ftd_csv_df, files_to_load_df):
    # separate all file path strings with the slash character and extract element corresponding to subject ID
    select_IDs = np.array(files_to_load_df['raw'].str.split('/').str.get(8), dtype=int)
    ftd_csv_df = ftd_csv_df[ftd_csv_df.ID_D.isin(list(select_IDs))]
    return ftd_csv_df


def load_covariates(files_to_load):
    # semicolon used as separator because of SPSS formatting
    ftd_csv_df = pd.read_csv(ftd_csv_file, sep=';')
    files_to_load_df = pd.DataFrame(files_to_load, columns=['raw'])
    ftd_csv_df = extract_IDs(ftd_csv_df, files_to_load_df)

    ftd_csv_df.Sex.loc[ftd_csv_df.Sex == 'm'], ftd_csv_df.Sex.loc[ftd_csv_df.Sex == 'f'] = 0., 1.
    ftd_csv_df.Sex = ftd_csv_df.Sex.astype(np.float)
    ftd_csv_df.age_Dx_T0 = ftd_csv_df.age_Dx_T0.str.replace(',', '.').astype(np.float)

    X_sex_age = ftd_csv_df.loc[:, ['age_Dx_T0', 'Sex']].as_matrix()
    X_TIV = pd.read_csv(TIV_csv_file, header=None)

    # verify whether same subjects are loaded
    X = np.concatenate((X_TIV, X_sex_age), axis=1)
    return X


def load_all_data(files_to_load, create_covariates, parcellation):
    if parcellation:
        data = parcellate_nifti(files_to_load)
    else:
        data = np.zeros((len(files_to_load), SIZE_VOXELS))
        for i, file_path in enumerate(files_to_load):
            data[i, :] = load_data(file_path)

    if create_covariates:
        X_covariates = load_covariates(files_to_load)
        data = np.concatenate((data, X_covariates), axis=1)
    return data


def create_labels(class1_num, class2_num, class3_num):
    y = np.zeros((class1_num + class2_num + class3_num, 4), dtype=np.bool)
    y[:class1_num, 0] = True
    y[class1_num:class1_num + class2_num, 1] = True
    y[class1_num + class2_num:class1_num + class2_num + class3_num, 2] = True
    y[class1_num:, 3] = True
    return y


def create_classification_data(data, class_labels_df, label1, label2):
    class1_labels = class_labels_df[label1].as_matrix().astype(np.bool)
    class2_labels = class_labels_df[label2].as_matrix().astype(np.bool)
    size_class1 = class1_labels.sum()
    size_class2 = class2_labels.sum()

    X = data[np.any((class1_labels, class2_labels), axis=0)]
    y = np.concatenate((np.ones(size_class1, dtype=np.int), np.zeros(size_class2, dtype=np.int)))
    return X, y


def create_data_matrices(save_path, load_path='', covariates=False, parcellation=False):
    if covariates:
        data_filename = 'data_set_with_cov.npy'
        if parcellation:
            data_filename = 'data_set_with_cov_with_parc.npy'
    else:
        data_filename = 'data_set.npy'
        if parcellation:
            data_filename = 'data_set_with_parc.npy'

    if load_path:
        data = np.load(osp.join(load_path, data_filename))
        class_labels_df = pd.read_csv(osp.join(load_path, 'class_labels.csv'))
    else:
        ftd_files = get_file_path(CLASSES_TRANSFORM['ftd'])
        neurological_files = get_file_path(CLASSES_TRANSFORM['neurol'])
        psychiatry_files = get_file_path(CLASSES_TRANSFORM['psych'])

        size_classes = {'ftd': len(ftd_files), 'neurol': len(neurological_files), 'psych': len(psychiatry_files)}

        data = load_all_data(ftd_files + neurological_files + psychiatry_files,
                             create_covariates=covariates, parcellation=parcellation)
        class_labels = create_labels(size_classes['ftd'], size_classes['neurol'], size_classes['psych'])

        class_labels_df = pd.DataFrame(data=class_labels, columns=['ftd', 'neurol', 'psych', 'rest'])

        np.save(osp.join(save_path, data_filename), data)
        class_labels_df.to_csv(osp.join(save_path, 'class_labels.csv'), index=False)

    X_ftd_neurol, y_ftd_neurol = create_classification_data(data, class_labels_df, 'ftd', 'neurol')
    X_ftd_psych, y_ftd_psych = create_classification_data(data, class_labels_df, 'ftd', 'psych')
    X_neurol_psych, y_neurol_psych = create_classification_data(data, class_labels_df, 'neurol', 'psych')
    X_ftd_rest, y_ftd_rest = create_classification_data(data, class_labels_df, 'ftd', 'rest')

    return X_ftd_neurol, y_ftd_neurol, X_ftd_psych, y_ftd_psych, X_neurol_psych, y_neurol_psych, X_ftd_rest, y_ftd_rest


def apply_masking(X):
    # since we don't have a mask (yet) we are basically comparing our data across subjects for each voxel and remove all
    # voxels which are 0 across ALL subjects
    id_keep = ~np.all(X == 0, axis=0)
    return X[:, id_keep]
