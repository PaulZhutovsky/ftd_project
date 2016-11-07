import re, os
import resample_atlas as res
from glob import glob
import nibabel as nib
import numpy as np

CLASSES_TRANSFORM = {'ftd': 'bvFTD', 'neurol': 'neurological', 'psych': 'psychiatric'}
scans_dir = '/data/shared/bvFTD/VBM/default/data'
parent_dir = '/data/shared/bvFTD/Machine_Learning/data/parcellated_GM_data'


def load_data(data_path):
    return nib.load(data_path).get_data().astype(np.float64)


def parcellate_NIFTI():
    # Resample atlases
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    content = os.listdir(parent_dir)
    select_atlases = filter(lambda x: re.search(r'.nii.gz$', x), content)
    if not select_atlases:
        res.resample_atlases()

    # Create dictionaries for atlas labels
    cort_labels, subcort_labels = res.load_atlas_labels()
    n_cort_labels, n_subcort_labels = len(cort_labels), len(subcort_labels)
    cort_atlas = load_data(os.path.join(parent_dir, select_atlases[0]))
    subcort_atlas = load_data(os.path.join(parent_dir, select_atlases[1]))

    files_to_load = sorted(glob(os.path.join(scans_dir, '*', '*', 'structural', 'mri', 'smw*')))
    n_files = len(files_to_load)
    data = np.zeros((len(files_to_load), n_cort_labels + n_subcort_labels))

    print 'Starting NIFTI parcellation... \n'
    for i, file_path in enumerate(files_to_load):
        print 'Running subject {} out of {} ... '.format(i, n_files)
        tmp = load_data(file_path)
        for cort_idx in sorted(np.array(cort_labels.keys(), dtype=int)):
            select_cort_area = (cort_atlas == cort_idx)
            data[i, cort_idx] = tmp[select_cort_area].mean()
        for sb_cort_idx in sorted(np.array(subcort_labels.keys(), dtype=int)):
            select_subcort_area = (subcort_atlas == sb_cort_idx)
            data[i, n_cort_labels + sb_cort_idx] = tmp[select_subcort_area].mean()

    print 'Finished parcellating NIFTIs \n'
    return data, cort_atlas, subcort_atlas


if __name__ == '__main__':
    parcellate_NIFTI()
