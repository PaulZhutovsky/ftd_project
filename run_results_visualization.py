from visualize_results import run
import os.path as osp


DATA_FOLDER = '/data/shared/bvFTD/Machine_Learning/'
DATA_PATTERN = 'results_ftd_*no_Cov_no_Parc_Smoothed'
SAVE_FOLDER = osp.join(DATA_FOLDER, 'figures_no_Cov_no_Parc_Smoothed')

if __name__ == '__main__':
    args = {'FOLDER': DATA_FOLDER,
            'PATTERN': DATA_PATTERN,
            'SAVE_FOLDER': SAVE_FOLDER}
    run(args)