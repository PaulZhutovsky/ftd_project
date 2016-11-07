import os
import subprocess
from FTD_classification import ensure_folder
import xml.etree.ElementTree as ET

FLIRT = '/usr/local/fsl_5.0.9/bin/flirt'
ATLAS_PARENT = '/usr/local/fsl_5.0.9/data/atlases'
CORT_ATLAS = os.path.join(ATLAS_PARENT, 'HarvardOxford/HarvardOxford-cort-maxprob-thr0-1mm.nii.gz')
SUBCORT_ATLAS = os.path.join(ATLAS_PARENT, 'HarvardOxford/HarvardOxford-sub-maxprob-thr0-1mm.nii.gz')
ID_MATRIX = '/usr/local/fsl_5.0.9/etc/flirtsch/ident.mat'
REF_SAMPLESIZE = '/data/shared/bvFTD/VBM/default/data/bvFTD/4908/structural/mri/smwp14908_T1_reoriented_time01.nii'
OUTPUT_DIR = '/data/shared/bvFTD/Machine_Learning/data/parcellated_GM_data'


def load_atlas_labels():
    cort_labels_path = os.path.join(ATLAS_PARENT, 'HarvardOxford-Cortical.xml')
    subcort_labels_path = os.path.join(ATLAS_PARENT, 'HarvardOxford-Subcortical.xml')
    cort_labels_tree, subcort_labels_tree = ET.parse(cort_labels_path), ET.parse(subcort_labels_path)
    cort_labels_root, subcort_labels_root = cort_labels_tree.getroot(), subcort_labels_tree.getroot()
    cort_labels, subcort_labels = {}, {}
    for elem in cort_labels_root.findall("./data/label"):
        cort_labels[int(elem.get('index'))] = elem.text
    for elem in subcort_labels_root.findall("./data/label"):
        subcort_labels[int(elem.get('index'))] = elem.text
    return cort_labels, subcort_labels


def cmd_flirt(input_atlas, output_atlas):
    cmd = FLIRT + ' -in ' + input_atlas + ' -applyxfm -init ' + ID_MATRIX + ' -out ' + \
          output_atlas + ' -paddingsize 0.0 -interp nearestneighbour -ref ' + REF_SAMPLESIZE
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    output, error = proc.communicate()
    print output


def resample_atlases():
    ensure_folder(OUTPUT_DIR)
    output_cort_atlas = os.path.join(OUTPUT_DIR, 'HarvardOxford-cort-resampled.nii')
    output_subcort_atlas = os.path.join(OUTPUT_DIR, 'HarvardOxford-subcort-resampled.nii')
    cmd_flirt(CORT_ATLAS, output_cort_atlas)
    cmd_flirt(SUBCORT_ATLAS, output_subcort_atlas)


if __name__ == '__main__':
    resample_atlases()
