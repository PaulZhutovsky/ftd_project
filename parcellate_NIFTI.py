import os.path as osp

import nibabel as nib
import numpy as np
import pandas as pd

scans_dir = '/data/shared/bvFTD/VBM/default_LOF5/data'
atlas_dir = '/data/shared/bvFTD/Machine_Learning/data/parcellated_GM_data'


def load_data(data_path):
    return nib.load(data_path).get_data().astype(np.float64)


def parcellate_nifti(files_to_load):

    # Create dictionaries for atlas labels
    cort_labels = pd.read_csv(osp.join(atlas_dir, 'cortical_labels.csv')).code_roi.values
    subcort_labels = pd.read_csv(osp.join(atlas_dir, 'subcortical_labels.csv')).code_roi.values

    n_cort_labels, n_subcort_labels = cort_labels.size, subcort_labels.size

    atlas_cort = load_data(osp.join(atlas_dir, 'HarvardOxford-cort-resampled.nii.gz'))
    atlas_subcort = load_data(osp.join(atlas_dir, 'HarvardOxford-subcort-resampled.nii.gz'))

    n_files = len(files_to_load)

    return create_parcellated_data(atlas_cort, atlas_subcort, cort_labels, files_to_load, n_cort_labels, n_files,
                                   n_subcort_labels, subcort_labels)


def create_parcellated_data(atlas_cort, atlas_subcort, cort_labels, files_to_load, n_cort_labels, n_files,
                            n_subcort_labels, subcort_labels):
    data = np.zeros((len(files_to_load), n_cort_labels + n_subcort_labels))
    print 'Starting NIFTI parcellation... \n'
    for i, file_path in enumerate(files_to_load):
        print 'Running subject {} out of {} ... '.format(i, n_files)
        subj_data = load_data(file_path)

        cort_parcellation = [subj_data[atlas_cort == roi_cort].mean() for roi_cort in cort_labels]
        subcort_parcellation = [subj_data[atlas_subcort == roi_subcort].mean() for roi_subcort in subcort_labels]

        data[i, :] = cort_parcellation + subcort_parcellation
    print 'Finished parcellating NIFTIs \n'
    return data
