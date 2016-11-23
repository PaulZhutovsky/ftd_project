from FTD_classification import run
import os.path as osp
from itertools import product

PARENT_DIR = '/data/shared/bvFTD/Machine_Learning/'
DATA_DIR = osp.join(PARENT_DIR, 'data')
SAVE_FOLDER = osp.join(PARENT_DIR, 'atlasStrucCov')
CLASSIFICATIONS = ['FTDvsNeurol', 'FTDvsNeurol', 'FTDvsRest', 'NeurolvsPsych']
SMOOTHING = [True]
PARCELLATION = [True]
COVARIATES = [False]
STRUCTURAL_COVARIANCE = True
RESAMPLING_ITERATIONS = 1000

if __name__ == '__main__':
    for i, (clf, smooth, atlas, cov) in enumerate(product(CLASSIFICATIONS, SMOOTHING, PARCELLATION, COVARIATES)):
        run(save_data=DATA_DIR, load_data=DATA_DIR, parcellation=atlas, smoothing=smooth, classification=clf,
            covariates=cov, save_dir=SAVE_FOLDER, num_resample_rounds=RESAMPLING_ITERATIONS,
            structural_covariance=STRUCTURAL_COVARIANCE)