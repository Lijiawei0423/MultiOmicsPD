
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
import statsmodels.formula.api as sm
from joblib import Parallel, delayed
from scipy import stats
pd.options.mode.chained_assignment = None

def get_sig_omic(omic, cov_df):
    omic_f_df = pd.read_csv(dpath + 'Results/Association/'+omic+'_Step1.csv')
    omic_f_df = omic_f_df.loc[omic_f_df.pval_fdr < 0.05]
    omic_f_lst = omic_f_df.Omics_feature.tolist()
    omic_df = pd.read_csv(dpath + 'Data/BloodData/'+omic+'Data/'+omic+'Data.csv', usecols=['eid'] + omic_f_lst)
    omic_df = pd.merge(omic_df, cov_df, how='inner', on=['eid'])
    return (omic_f_lst, omic_df)

def get_omic_pair(omic1, omic2, cov_df, cov_f_lst):
    _, omic1_df = get_sig_omic(omic1, cov_df)
    omic1_df.drop(cov_f_lst, axis = 1, inplace = True)
    _, omic2_df = get_sig_omic(omic2, cov_df)
    omic_pair_df = pd.merge(omic1_df, omic2_df, how='inner', on=['eid'])
    return omic_pair_df

def get_asss(mydf, omic1, omic2, omic_f1, omic_f2, cov_f_lst, my_formula1, my_formula2):
    tmpdf = mydf.copy()
    tmpdf = tmpdf[[omic_f1, omic_f2] + cov_f_lst]
    tmpdf.rename(columns = {omic_f1:'omic_y', omic_f2:'omic_x'}, inplace = True)
    tmpdf.dropna(axis = 0, how = 'any', inplace = True)
    nb2analy = len(tmpdf)
    corr, _ = stats.spearmanr(tmpdf.omic_y, tmpdf.omic_x)
    try:
        ols_mod1 = sm.ols(my_formula1, tmpdf).fit()
        beta_hat1 = ols_mod1.params.loc['omic_x']
        beta_sd1 = ols_mod1.bse.loc['omic_x']
        t_val = np.round(beta_hat1/beta_sd1, 5)
        pval = ols_mod1.pvalues.loc['omic_x']
        ci_mod1 = ols_mod1.conf_int(alpha=0.05)
        ci_lbd1, ci_ubd1 = ci_mod1.loc['omic_x'][0], ci_mod1.loc['omic_x'][1]
        beta_est1 = f'{beta_hat1:.3f}' ' [' + f'{ci_lbd1:.3f}' + '-' + f'{ci_ubd1:.3f}' + ']'
        ols_mod2 = sm.ols(my_formula2, tmpdf).fit()
        beta_hat2 = ols_mod2.params.loc['omic_y']
        ci_mod2 = ols_mod2.conf_int(alpha=0.05)
        ci_lbd2, ci_ubd2 = ci_mod2.loc['omic_y'][0], ci_mod2.loc['omic_y'][1]
        beta_est2 = f'{beta_hat2:.3f}' ' [' + f'{ci_lbd2:.3f}' + '-' + f'{ci_ubd2:.3f}' + ']'
        tmp_out = [omic1, omic2, omic_f1, omic_f2, nb2analy, corr, t_val, pval, beta_est1, beta_est2]
    except:
        tmp_out = [omic1, omic2, omic_f1, omic_f2, nb2analy, corr, np.nan, np.nan, np.nan, np.nan]
    return tmp_out

dpath = '../'

my_formula1 = 'omic_y ~ age + C(sex) + C(race) + educ + tdi + C(smk) + C(alc) + bmi + omic_x'
my_formula2 = 'omic_x ~ age + C(sex) + C(race) + educ + tdi + C(smk) + C(alc) + bmi + omic_y'

tgt_df = pd.read_csv(dpath + 'Data/Target/PD/PD_outcomes.csv', usecols=['eid', 'target_y'])
tgt_df = tgt_df.loc[tgt_df.target_y == 0]
cov_f_lst = ['age', 'sex', 'race', 'educ', 'tdi', 'smk', 'alc', 'bmi']
cov_df = pd.read_csv(dpath + 'Data/Covariates/CovariatesImputed.csv', usecols = ['eid']+cov_f_lst)
cov_df['race'].replace([1,2,3,4], [1, 0, 0, 0], inplace = True)
cov_df = pd.merge(cov_df, tgt_df[['eid']], how = 'inner', on = ['eid'])

cll_f_lst, _ = get_sig_omic('Biochemistry', cov_df)
gen_f_lst, _ = get_sig_omic('Genomic', cov_df)
met_f_lst, _ = get_sig_omic('Metabolomic', cov_df)
pro_f_lst, _ = get_sig_omic('Proteomic', cov_df)

cll_gen_df = get_omic_pair('Biochemistry', 'Genomic', cov_df, cov_f_lst)
cll_met_df = get_omic_pair('Biochemistry', 'Metabolomic', cov_df, cov_f_lst)
cll_pro_df = get_omic_pair('Biochemistry', 'Proteomic', cov_df, cov_f_lst)
met_gen_df = get_omic_pair('Metabolomic', 'Genomic', cov_df, cov_f_lst)
met_pro_df = get_omic_pair('Metabolomic', 'Proteomic', cov_df, cov_f_lst)
pro_gen_df = get_omic_pair('Proteomic', 'Genomic', cov_df, cov_f_lst)

print(('C_G', 'C_M', 'C_P', 'M_G', 'M_P', 'P_G'))
print((cll_gen_df.shape[0], cll_met_df.shape[0], cll_pro_df.shape[0], met_gen_df.shape[0], met_pro_df.shape[0], pro_gen_df.shape[0]))
print((cll_gen_df.shape[1], cll_met_df.shape[1], cll_pro_df.shape[1], met_gen_df.shape[1], met_pro_df.shape[1], pro_gen_df.shape[1]))
 
cross_omic_df, omic1, omic2, omic_f_lst1, omic_f_lst2 = cll_gen_df, 'Biochemistry', 'Genomic', cll_f_lst, gen_f_lst
cross_omic_df, omic1, omic2, omic_f_lst1, omic_f_lst2 = cll_met_df, 'Biochemistry', 'Metabolomic', cll_f_lst, met_f_lst
cross_omic_df, omic1, omic2, omic_f_lst1, omic_f_lst2 = cll_pro_df, 'Biochemistry', 'Proteomic', cll_f_lst, pro_f_lst
cross_omic_df, omic1, omic2, omic_f_lst1, omic_f_lst2 = met_gen_df, 'Metabolomic', 'Genomic', met_f_lst, gen_f_lst
#cross_omic_df, omic1, omic2, omic_f_lst1, omic_f_lst2 = met_pro_df, 'Metabolomic', 'Proteomic', met_f_lst, pro_f_lst
#cross_omic_df, omic1, omic2, omic_f_lst1, omic_f_lst2 = pro_gen_df, 'Proteomic', 'Genomic', pro_f_lst, gen_f_lst

cross_omic_ass_lst = []
for omic_f1 in tqdm(omic_f_lst1):
    cross_omic_ass_lst += Parallel(n_jobs=10)(delayed(get_asss)(cross_omic_df, omic1, omic2, omic_f1, omic_f2, cov_f_lst, my_formula1, my_formula2) for omic_f2 in omic_f_lst2)

cross_omic_ass_df = pd.DataFrame(cross_omic_ass_lst)
cross_omic_ass_df.columns = ['OmicGroup_y', 'OmicGroup_x', 'OmicFeature_y', 'OmicFeature_x', 'NB4Aanaly', 'Sp_corr', 't_stat', 'Pval_raw', 'Beta_est', 'Beta_est_rev']
cross_omic_ass_df.to_csv(dpath + 'Results/NetworkAnalysis/IncidentPDIndividuals/PairwiseAssociations/'+omic1+'-'+omic2+'-associations.csv', index=False)

