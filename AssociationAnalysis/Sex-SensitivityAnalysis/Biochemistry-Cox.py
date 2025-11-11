

import glob
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from tqdm import tqdm
import os
import re
from statsmodels.stats.multitest import fdrcorrection
from mne.stats import bonferroni_correction
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('error')

def results_summary(tgt_out_df):
    hr_out_lst, p_out_lst = [], []
    for i in range(len(tgt_out_df)):
        hr = f'{tgt_out_df.hr.iloc[i]:.2f}'
        lbd = f'{tgt_out_df.hr_lbd.iloc[i]:.2f}'
        ubd = f'{tgt_out_df.hr_ubd.iloc[i]:.2f}'
        hr_out_lst.append(hr + ' [' + lbd + '-' + ubd + ']')
        if tgt_out_df.pval_bfi.iloc[i] < 0.001:
            p_out_lst.append('***')
        elif tgt_out_df.pval_bfi.iloc[i] < 0.01:
            p_out_lst.append('**')
        elif tgt_out_df.pval_bfi.iloc[i] < 0.05:
            p_out_lst.append('*')
        else:
            p_out_lst.append('')
    return (hr_out_lst, p_out_lst)

def process(omics_f, mydf, cov_f_lst, my_formula):
    tmp_df = mydf[['eid', 'target_y', 'BL2Target_yrs', omics_f]+cov_f_lst]
    tmp_df.rename(columns={omics_f: 'x_omics'}, inplace=True)
    rm_eid_idx = tmp_df.index[tmp_df.x_omics.isnull() == True]
    tmp_df.drop(rm_eid_idx, axis=0, inplace=True)
    tmp_df.reset_index(inplace=True, drop=True)
    nb_all, nb_case = len(tmp_df), tmp_df.target_y.sum()
    prop_case = np.round(nb_case / nb_all * 100, 3)
    cph = CoxPHFitter(penalizer=1e-3)
    try:
        cph.fit(tmp_df, duration_col='BL2Target_yrs', event_col='target_y', formula=my_formula)
        hr = np.round(cph.hazard_ratios_.x_omics, 5)
        lbd = np.round(np.exp(cph.confidence_intervals_).loc['x_omics'][0], 5)
        ubd = np.round(np.exp(cph.confidence_intervals_).loc['x_omics'][1], 5)
        pval = cph.summary.p.x_omics
        tmpout = [omics_f, nb_all, nb_case, prop_case, hr, lbd, ubd, pval]
    except:
        tmpout = [omics_f, nb_all, nb_case, prop_case, np.nan, np.nan, np.nan, np.nan]
    return tmpout

dpath = '../'

omics = 'Biochemistry'
step = 1

sex_str = 'Female'
sex_id = 0

#sex_str = 'Male'
#sex_id = 1

omics_df = pd.read_csv(dpath + 'Data/BloodData/'+omics+'Data/'+omics+'Data.csv')
omics_f_lst = omics_df.columns.tolist()[1:]
eid_idx_df = pd.read_csv(dpath + 'Data/BloodData/BloodDataIndex.csv')
eid_idx_df = eid_idx_df.loc[eid_idx_df.Step == step]
omics_dict = pd.read_csv(dpath + 'Data/BloodData/BloodDict.csv')

tgt_df = pd.read_csv(dpath + 'Data/Target/PD/PD_outcomes.csv', usecols=['eid', 'target_y', 'BL2Target_yrs'])
cov_df = pd.read_csv(dpath + 'Data/Covariates/CovariatesImputed.csv')
cov_df['race'].replace([1,2,3,4], [1, 0, 0, 0], inplace = True)
cov_df = cov_df.loc[cov_df.sex == sex_id]
mydf = pd.merge(tgt_df, eid_idx_df, how='inner', on=['eid'])
mydf = pd.merge(mydf, cov_df, how='inner', on=['eid'])
mydf = pd.merge(mydf, omics_df, how='inner', on=['eid'])

cov_f_lst = ['age', 'race', 'educ', 'tdi', 'smk', 'alc', 'bmi', 'fastingtime']
my_formula = "age + C(race) + educ + tdi + C(smk) + C(alc) + bmi + fastingtime + x_omics"

myout_df = Parallel(n_jobs=10)(delayed(process)(omics_f, mydf, cov_f_lst, my_formula) for omics_f in omics_f_lst)
myout_df = pd.DataFrame(myout_df)
myout_df.columns = ['Omics_feature', 'nb_individuals', 'nb_case', 'prop_case(%)', 'hr', 'hr_lbd', 'hr_ubd', 'pval_raw']
_, p_f_fdr = fdrcorrection(myout_df.pval_raw.fillna(1), alpha=0.05)
myout_df['pval_fdr'] = p_f_fdr
myout_df.loc[myout_df['pval_fdr'] >= 1, 'pval_fdr'] = 1
_, p_f_bfi = bonferroni_correction(myout_df.pval_raw.fillna(1), alpha=0.05)
myout_df['pval_bfi'] = p_f_bfi
myout_df.loc[myout_df['pval_bfi'] >= 1, 'pval_bfi'] = 1
myout_df['hr_output'], myout_df['pval_significant'] = results_summary(myout_df)
myout_df = myout_df[['Omics_feature', 'nb_individuals', 'nb_case', 'prop_case(%)', 'hr', 'hr_lbd', 'hr_ubd',
                     'pval_raw', 'pval_fdr', 'pval_bfi', 'hr_output']]

myout_df = pd.merge(myout_df, omics_dict, how = 'left', on = 'Omics_feature')

myout_df.to_csv(dpath + 'Results/Association/'+sex_str+'/'+omics+'_Step'+str(step)+'.csv', index=False)


