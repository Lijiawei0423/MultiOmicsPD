
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
outpath = dpath+'Output/multi_label/yaxing_new/'

bld_idx_df = pd.read_csv(dpath+'BloodDataIndex.csv',
                         usecols = ['eid','Proteomic','Metabolomic', 'ClinicalLab','Genomic','Step'])
bld_dict = pd.read_csv(dpath+'BloodDict.csv',
                       usecols = ['Omics_group', 'Omics_feature', 'Omics_fullname', 'Omics_code'])
yaxing_df = pd.read_csv(dpath+'labels.csv',usecols=['eid','cluster_label'])
yaxing_df['cluster_label'] = yaxing_df['cluster_label']-1
yaxing_df.rename(columns = {'cluster_label':'target_y'}, inplace = True)
tgt_df = pd.read_csv(dpath + 'PD_outcomes.csv', usecols=['eid', 'target_y','BL2Target_yrs'])
cov_df = pd.read_csv('/Users/lijiawei/Desktop/Covariates/CovariatesImputed.csv')
cov_f_lst = ['age','sex','race','educ','tdi','smk','alc','bmi']

cli_f_lst,cli_df = get_sig_omic(dpath, 'ClinicalLab')
meta_f_lst,meta_df = get_sig_omic(dpath, 'Metabolomic')
tmp_f = cli_f_lst+meta_f_lst+cov_f_lst


mydf = pd.merge(cli_df,meta_df, how = 'inner', on=['eid'])
mydf = pd.merge(mydf,yaxing_df, how = 'inner', on=['eid'])
mydf = pd.merge(mydf,cov_df, how = 'inner', on = ['eid'])
train_df = mydf.copy()

test_df = bld_idx_df.loc[(bld_idx_df['Step'] == 1)]
test_df = pd.merge(test_df,cli_df, how = 'inner', on=['eid'])
test_df = pd.merge(test_df,meta_df, how = 'inner', on=['eid'])
test_df = pd.merge(test_df,tgt_df, how = 'inner', on=['eid'])
test_df = pd.merge(test_df,cov_df, how = 'inner', on = ['eid'])
test_df = test_df[test_df['target_y'] == 1]
test_df.reset_index(inplace = True, drop = True)

my_params = {'n_estimators': 500,
             'max_depth': 15,
             'num_leaves': 10,
             'subsample': 0.7,
             'learning_rate': 0.01,
             'colsample_bytree': 0.7}

X_train,X_test = train_df[tmp_f], test_df[tmp_f]
y_train = train_df['target_y']

model = LGBMClassifier(**my_params)
model.fit(X_train, y_train)
y_pred_train = model.predict_proba(X_train)[:, 1]
y_pred_test = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_train, y_pred_train)
youden_index = tpr - fpr
optimal_idx = np.argmax(youden_index)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold = 0.406

print(f'Optimal threshold to maximize Youden\'s index: {optimal_threshold:.3f}')

y_pred_test_labels = (y_pred_test >= optimal_threshold).astype(int)

output_df = pd.DataFrame({'eid': test_df['eid'], 'risk_score': y_pred_test,'yaxing_label': y_pred_test_labels})
label_distribution = output_df['yaxing_label'].value_counts()
print('Distribution of yaxing_label:')
print(label_distribution)

output_df.to_csv(outpath + 'predicted_labels.csv', index=False)
train_labels_df = pd.DataFrame({'eid': train_df['eid'], 'yaxing_label': y_train})

combined_df = pd.concat([train_labels_df, output_df], ignore_index=True)

combined_df.to_csv(outpath + 'combined_labels.csv', index=False)


