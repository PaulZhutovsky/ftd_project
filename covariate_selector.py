import numpy as np
import pandas as pd
from glob import glob
import os.path as osp


class CovariateSelector(object):

    def __init__(self, cov_csv_file='/data/shared/bvFTD/Machine_Learning/data/bvFTD_covariates.csv',
                 scans_dir='/data/shared/bvFTD/VBM/default_LOF5/data'):

        self.cov_csv_file = cov_csv_file
        self.scans_dir = scans_dir
        self.covariates = self.load_covariates()
        self.n_covariates = self.covariates.shape[1]

    def get_subjects_to_load(self):
        # unfortunately the following paths are hardcoded..
        ftd_files = sorted(glob(osp.join(self.scans_dir, 'bvFTD', '*', 'structural', 'mri', 'smw*')))
        neurological_files = sorted(glob(osp.join(self.scans_dir, 'neurological', '*', 'structural', 'mri', 'smw*')))
        psychiatry_files = sorted(glob(osp.join(self.scans_dir, 'psychiatric', '*', 'structural', 'mri', 'smw*')))
        return ftd_files + neurological_files + psychiatry_files

    def extract_IDs(self, ftd_csv_df, files_to_load_df):
        select_IDs = []
        # separate all file path strings with the slash character and extract element corresponding to subject ID
        splitted_list = [item for splits in np.array(files_to_load_df['raw'].str.split('/')) for item in splits]
        for split in splitted_list:
            if split.isdigit():
                select_IDs = np.append(select_IDs, int(split))
        ftd_csv_df = ftd_csv_df.loc[ftd_csv_df.ID_D.isin(list(select_IDs))]
        # sort data according to ascending path structures used for subject scan directories
        ftd_csv_df = ftd_csv_df.sort_values(by=['diagnosis_cat', 'ID_D'], ascending=[1, 1])
        return ftd_csv_df

    def load_covariates(self):
        files_to_load = self.get_subjects_to_load()

        cov_df = pd.read_csv(self.cov_csv_file)
        files_to_load_df = pd.DataFrame(files_to_load, columns=['raw'])
        cov_df = self.extract_IDs(cov_df, files_to_load_df)
        m_idx, f_idx = cov_df.Sex == 'm', cov_df.Sex == 'f'
        cov_df.loc[m_idx, 'Sex'], cov_df.loc[f_idx, 'Sex'] = 0., 1.
        cov_df.Sex = cov_df.Sex.astype(np.float)
        cov_df.age_Dx_T0 = cov_df.age_Dx_T0.str.replace(',', '.').astype(np.float)

        X = cov_df.loc[:, ['age_Dx_T0', 'Sex', 'TIV']].values
        return X

    def cov_from_id(self, select_id):
        return self.covariates[select_id]


class CovariateScaler(object):

    def __init__(self):
        self.train_covariates = np.nan
        self.test_covariates = np.nan
        self.min = np.nan
        self.max = np.nan

    def set_covariates(self, train_covariates, test_covariates):
        self.train_covariates = train_covariates
        self.test_covariates = test_covariates

    def fit(self, X, y=None):
        self.min = self.train_covariates.min(axis=0, keepdims=True)
        self.max = self.train_covariates.max(axis=0, keepdims=True)

    def transform(self, X, y=None):
        return self.__transform(X, self.test_covariates)

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.__transform(X, self.train_covariates)

    def __transform(self, data, covariates):
        normalized_covariates = (covariates - self.min) / (self.max - self.min)
        return np.column_stack((data, normalized_covariates))
