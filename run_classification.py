from FTD_classification import run
import os.path as osp
from itertools import product

PARENT_DIR = '/data/shared/bvFTD/Machine_Learning/'
DATA_DIR = osp.join(PARENT_DIR, 'data')
SAVE_FOLDER = osp.join(PARENT_DIR, 'results')
CLASSIFICATIONS = ['FTDvsNeurol', 'FTDvsPsych', 'FTDvsRest', 'NeurolvsPsych']
SMOOTHING = [False]
PARCELLATION = [True]
COVARIATES = [False]
STRUCTURAL_COVARIANCE = False
SINGLE_CLASSIFICATION = ['RF']  # ''
RESAMPLING_ITERATIONS = 1000

if __name__ == '__main__':
    for i, (clf, smooth, atlas, cov, single_clf) in enumerate(product(CLASSIFICATIONS, SMOOTHING, PARCELLATION,
                                                                      COVARIATES, SINGLE_CLASSIFICATION)):
        run(save_data=DATA_DIR, load_data=DATA_DIR, parcellation=atlas, smoothing=smooth, classification=clf,
            covariates=cov, save_dir=SAVE_FOLDER, num_resample_rounds=RESAMPLING_ITERATIONS,
            structural_covariance=STRUCTURAL_COVARIANCE, single_classification=single_clf)
