import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
import re
import shap

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
outpath = dpath+'Output/multi/'

bld_idx_df = pd.read_csv(dpath+'BloodDataIndex.csv',
                         usecols = ['eid','Proteomic','Metabolomic', 'ClinicalLab','Genomic','Step', 'cv_fold', 'in_cv_fold'])
bld_dict = pd.read_csv(dpath+'BloodDict.csv',
                       usecols = ['Omics_group', 'Omics_feature', 'Omics_fullname', 'Omics_code'])

tgt_df = pd.read_csv(dpath + 'PD_outcomes.csv', usecols=['eid', 'target_y','BL2Target_yrs'])

cll_f_lst, cll_df = get_sig_omic(dpath, 'ClinicalLab')
gen_f_lst, gen_df = get_sig_omic(dpath, 'Genomic')
gen_f_lst1 = [gen_f.replace(':', '_') for gen_f in gen_f_lst]
gen_df.columns = ['eid'] + gen_f_lst1
met_f_lst, met_df = get_sig_omic(dpath, 'Metabolomic')
pro_f_lst, pro_df = get_sig_omic(dpath, 'Proteomic')
omic_f_lst = cll_f_lst + gen_f_lst1 + met_f_lst + pro_f_lst
omic_df = pd.merge(cll_df, gen_df, how='inner', on=['eid'])
omic_df = pd.merge(omic_df, met_df, how='inner', on=['eid'])
omic_df = pd.merge(omic_df, pro_df, how='inner', on=['eid'])

mydf = pd.merge(tgt_df, bld_idx_df, how = 'inner', on=['eid'])
mydf = pd.merge(mydf, omic_df, how = 'inner', on=['eid'])
mydf['target_y'].loc[mydf.BL2Target_yrs > 5] = 0

tmp_f = omic_f_lst
fold_id_list = [i for i in range(5)]
my_params = {'n_estimators': 500,
             'max_depth': 15,
             'num_leaves': 10,
             'subsample': 0.7,
             'learning_rate': 0.01,
             'colsample_bytree': 0.7}
AUC_cv = []

print(f'Number of features: {len(tmp_f)}')


for fold_id in fold_id_list:
    train_idx = mydf[(mydf['cv_fold'] != fold_id) & (mydf['Step'] == 2)].index
    test_idx = mydf[(mydf['cv_fold'] == fold_id) & (mydf['Step'] == 2)].index
    X_train, X_test = mydf.iloc[train_idx][tmp_f], mydf.iloc[test_idx][tmp_f]
    
    y_train, y_test = mydf.iloc[train_idx]['target_y'], mydf.iloc[test_idx]['target_y']
    print('Number of training samples:', len(X_train))
    print('Number of test samples:', len(X_test))
    model = LGBMClassifier(**my_params)
    model.fit(X_train, y_train)

    # Predict and calculate AUC
    y_pred_train = model.predict_proba(X_train)[:, 1]
    y_pred_test = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_test)
    AUC_cv.append(auc)
    print(f'Fold {fold_id}: AUC = {auc:.3f}')
    
    prors_train_df = pd.DataFrame({'eid': mydf.iloc[train_idx].eid.tolist(),
                                   'cv_fold': mydf.iloc[train_idx].cv_fold.tolist(),
                                   'target_y': y_train.tolist(),
                                   'ProRS': y_pred_train.tolist()})
    prors_test_df = pd.DataFrame({'eid': mydf.iloc[test_idx].eid.tolist(),
                                   'cv_fold': mydf.iloc[test_idx].cv_fold.tolist(),
                                   'target_y': y_test.tolist(),
                                   'ProRS': y_pred_test.tolist()})
    prors_df = prors_test_df
    prors_df.to_csv(outpath + str(fold_id) + '_5y.csv', index = False)

AUC_cv = np.array(AUC_cv)
print('AUC: ', AUC_cv)
print(f'AUC: {AUC_cv.mean():.3f} +/- {AUC_cv.std():.3f}')
print('finished')






