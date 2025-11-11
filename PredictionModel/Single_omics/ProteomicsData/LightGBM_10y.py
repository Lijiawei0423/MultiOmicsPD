import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import numpy as np


def get_sig_omic(dpath, omic):
    omic_f_df = pd.read_csv(dpath + 'Association/'+omic+'_Step1.csv')
    omic_f_df = omic_f_df.loc[omic_f_df.pval_fdr < 0.05]
    omic_f_lst = omic_f_df.Omics_feature.tolist()
    omic_df = pd.read_csv(dpath +omic+'Data/'+omic+'Data.csv', usecols=['eid'] + omic_f_lst)
    return (omic_f_lst, omic_df)



dpath = '..'
outpath = dpath+'Output/'

bld_idx_df = pd.read_csv(dpath+'BloodDataIndex.csv',
                         usecols = ['eid', 'Step', 'cv_fold', 'in_cv_fold','Proteomic'])
bld_dict = pd.read_csv(dpath+'BloodDict.csv',
                       usecols = ['Omics_group', 'Omics_feature', 'Omics_fullname', 'Omics_code'])

tgt_df = pd.read_csv(dpath + 'PD_outcomes.csv', usecols=['eid', 'target_y','BL2Target_yrs'])

pro_f_lst, pro_df = get_sig_omic(dpath, 'Proteomic')
omic_f_lst = pro_f_lst
omic_df = pro_df

tmp_f = omic_f_lst
mydf = pd.merge(tgt_df, bld_idx_df, how = 'inner', on=['eid'])
mydf = pd.merge(mydf, omic_df, how = 'inner', on=['eid'])
mydf['target_y'].loc[mydf.BL2Target_yrs > 10] = 0



my_params = {'n_estimators': 500,
             'max_depth': 15,
             'num_leaves': 10,
             'subsample': 0.7,
             'learning_rate': 0.01,
             'colsample_bytree': 0.7}


train_eids = mydf[(mydf['Step'] == 1) & (mydf['Proteomic'] == 1)]['eid']
train_samples = mydf[mydf['eid'].isin(train_eids)]


test_eids = mydf[mydf['Step'] == 2]['eid']
test_samples = mydf[mydf['eid'].isin(test_eids)]

X_train,y_train = train_samples[tmp_f],train_samples['target_y']
X_test,y_test = test_samples[tmp_f],test_samples['target_y']


my_lgb = LGBMClassifier(objective='binary', metric='auc', is_unbalance=True, n_jobs=4, verbosity=-1, seed=2020)
my_lgb.set_params(**my_params)
my_lgb.fit(X_train, y_train)

y_pred_train = my_lgb.predict_proba(X_train)[:, 1]
prors_train_df = pd.DataFrame({'eid': train_samples['eid'].tolist(),
                                   'target_y': y_train.tolist(),
                                   'ProRS': y_pred_train.tolist()})


y_pred_test = my_lgb.predict_proba(X_test)[:, 1]
prors_test_df = pd.DataFrame({'eid': test_samples['eid'].tolist(),
                               'target_y': y_test.tolist(),
                               'ProRS': y_pred_test.tolist()})
auc_test = roc_auc_score(y_test, y_pred_test)
print(f'Test AUC: {auc_test:.3f}')

prors_df = pd.concat([prors_test_df, prors_train_df], axis = 0)
prors_df.to_csv(outpath + '/ProteomicData/LightGBM_10y'+ '.csv', index = False)


print('finished')