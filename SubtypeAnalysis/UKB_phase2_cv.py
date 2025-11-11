
import numpy as np
import pandas as pd
import sys
import os
sys.path.append('..')
from Utility.Training_Utilities import *
from lightgbm import LGBMClassifier
import warnings
import re
import shap
from tqdm import tqdm
from sklearn.metrics import roc_curve
pd.options.mode.chained_assignment = None  # default='warn'

def get_sig_omic(dpath, omic):
    omic_f_df = pd.read_csv(dpath + 'Association/'+omic+'_Step1.csv')
    omic_f_df = omic_f_df.loc[omic_f_df.pval_fdr < 0.05]
    omic_f_lst = omic_f_df.Omics_feature.tolist()
    omic_df = pd.read_csv(dpath +omic+'Data/'+omic+'Data.csv', usecols=['eid'] + omic_f_lst)
    return (omic_f_lst, omic_df)

def normal_imp(mydict):
    mysum = sum(mydict.values())
    mykeys = mydict.keys()
    for key in mykeys:
        mydict[key] = mydict[key]/mysum
    return mydict

dpath = '..'
outpath = dpath+'Output/multi_label/yaxing/'

bld_idx_df = pd.read_csv(dpath+'BloodDataIndex.csv',
                         usecols = ['eid','Proteomic','Metabolomic', 'ClinicalLab','Genomic','Step'])
bld_dict = pd.read_csv(dpath+'BloodDict.csv',
                       usecols = ['Omics_group', 'Omics_feature', 'Omics_fullname', 'Omics_code'])
yaxing_df = pd.read_csv(dpath+'labels.csv',usecols=['eid','cluster_label'])
yaxing_df['cluster_label'] = yaxing_df['cluster_label']-1
tgt_df = pd.read_csv(dpath + 'PD_outcomes.csv', usecols=['eid', 'target_y','BL2Target_yrs'])
cov_df = pd.read_csv('..')
cov_f_lst = ['age','sex','race','educ','tdi','smk','alc','bmi']
bld_idx_df = pd.read_csv(dpath+'BloodDataIndex.csv',
                         usecols = ['eid','Step','cv_fold'])

cli_f_lst,cli_df = get_sig_omic(dpath, 'ClinicalLab')
meta_f_lst,meta_df = get_sig_omic(dpath, 'Metabolomic')
tmp_f = cli_f_lst+meta_f_lst+cov_f_lst


mydf = pd.merge(cli_df,meta_df, how = 'inner', on=['eid'])
mydf = pd.merge(mydf,yaxing_df, how = 'inner', on=['eid'])
mydf = pd.merge(mydf,cov_df, how = 'inner', on = ['eid'])
mydf = pd.merge(mydf,bld_idx_df, how = 'inner', on=['eid'])
train_df = mydf.copy()

my_params = {'n_estimators': 500,
             'max_depth': 15,
             'num_leaves': 10,
             'subsample': 0.7,
             'learning_rate': 0.01,
             'colsample_bytree': 0.7}

AUC_cv = []
fold_id_lst = range(5)
all_prors_test_df = []  

for fold_id in fold_id_lst:
    train_idx = mydf[(mydf['cv_fold'] != fold_id)].index
    test_idx = mydf[(mydf['cv_fold'] == fold_id)].index
    X_train, X_test = mydf.iloc[train_idx][tmp_f], mydf.iloc[test_idx][tmp_f]
    y_train, y_test = mydf.iloc[train_idx]['cluster_label'], mydf.iloc[test_idx]['cluster_label']
    model = LGBMClassifier(**my_params)
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict_proba(X_train)[:, 1]
    y_pred_test = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_test)
    AUC_cv.append(auc)
    print(f'Fold {fold_id}: AUC = {auc:.3f}')
    
    prors_test_df = pd.DataFrame({'eid': mydf.iloc[test_idx].eid.tolist(),
                                   'cv_fold': mydf.iloc[test_idx].cv_fold.tolist(),
                                   'target_y': y_test.tolist(),
                                   'ProRS': y_pred_test.tolist()})
    
    all_prors_test_df.append(prors_test_df) 

all_prors_test_df = pd.concat(all_prors_test_df, axis=0)
outpath = '..'
all_prors_test_df.to_csv(outpath + 'all_folds_results.csv', index=False)

AUC_cv = np.array(AUC_cv)
print('AUC: ', AUC_cv)
print(f'AUC: {AUC_cv.mean():.3f} +/- {AUC_cv.std():.3f}')
print('finished')