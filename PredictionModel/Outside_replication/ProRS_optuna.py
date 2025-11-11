import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
import sys 
import os
import warnings
sys.path.append(os.path.abspath(".."))
from s0_IncidentPredict1 import select_params_combo, get_top_nmr_lst, get_best_params, model_train_pred

nb_cpus = 10
nb_params = 500
my_seed = 2024
warnings.filterwarnings('ignore',category=DeprecationWarning)
dpath = '../BloodData/'
ppmipath = '../PPMI/'
outpath = dpath+'Output/'

datapath = '../PPMI_clinical/'
ppmipath = '../PPMI/'
outpath = '../BloodData/Output/'

ppmi_df = pd.read_csv(datapath+'...csv')

def get_sig_omic(dpath, omic):
    omic_f_df = pd.read_csv(dpath + 'Association/'+omic+'_Step1.csv')
    omic_f_df = omic_f_df.loc[omic_f_df.pval_fdr < 0.05]
    omic_f_lst = omic_f_df.Omics_feature.tolist()
    omic_df = pd.read_csv(dpath +omic+'Data/'+omic+'Data.csv', usecols=['eid'] + omic_f_lst)
    return (omic_f_lst, omic_df)

pro_f_lst, pro_df = get_sig_omic(dpath, 'Proteomic')
ppmi_df = pd.read_csv(ppmipath + 'PPMI_UKB_Proteomics_CommonFeatures.csv' )
ppmi_tgt_df = pd.read_csv(ppmipath + 'PPMI_PD_outcomes.csv', usecols=['PATNO', 'Pro_Hea'])
ppmi_tgt_df = ppmi_tgt_df[ppmi_tgt_df['Pro_Hea'].isin([0, 1])]
ppmi_df = pd.merge(ppmi_df, ppmi_tgt_df, how = 'inner', on = ['PATNO'])
ppmi_f_lst = ppmi_df.columns.tolist()
pro_common_features = list(set(ppmi_f_lst) & set(pro_f_lst))
tmp_f = pro_common_features
mydf = ppmi_df

params_dict = {'n_estimators': [100, 200, 300, 400, 500],
               'max_depth': np.linspace(5, 30, 6).astype('int32').tolist(),
               'num_leaves': np.linspace(5, 30, 6).astype('int32').tolist(),
               'subsample': np.linspace(0.6, 1, 9).tolist(),
               'learning_rate': [0.1, 0.05, 0.01, 0.001],
               'colsample_bytree': np.linspace(0.6, 1, 9).tolist()}
ini_params = {'n_estimators': 500, 'max_depth': 15, 'num_leaves': 10,
              'subsample': 0.7, 'learning_rate': 0.01, 'colsample_bytree': 0.7}
candidate_params_lst = select_params_combo(params_dict, nb_params, my_seed)
candidate_params_lst = candidate_params_lst + [ini_params]


region_fold_id_lst = range(5)
inner_cv_fold_id_lst = range(5)
eid_lst, y_test_lst = [], []
y_pred_nmr_lst, y_pred_cov_lst, y_pred_nmr_cov_lst = [], [], []

imp_df = pd.DataFrame({'feature_code':tmp_f})

for fold_id in region_fold_id_lst:
    traindf, testdf = mydf.loc[mydf.cv_fold != fold_id], mydf.loc[mydf.cv_fold == fold_id]
    traindf.reset_index(inplace=True, drop=True)
    testdf.reset_index(inplace=True, drop=True)
    traindf = traindf.copy() 
    testdf = testdf.copy()    
    top_nmr_f_lst, lgb_nmr = get_top_nmr_lst(traindf, tmp_f, my_seed)
    top_nmr_f_lst = tmp_f
    params_nmr = ini_params.copy()
    params_nmr = get_best_params(traindf, top_nmr_f_lst, inner_cv_fold_id_lst, candidate_params_lst, my_seed)
    print(f'Fold {fold_id} finished: {params_nmr}')
    y_pred_nmr_lst += model_train_pred(traindf, testdf, top_nmr_f_lst, params_nmr, my_seed)
    eid_lst += testdf.PATNO.tolist()
    y_test_lst += testdf.Pro_Hea.tolist()
    tg_imp = lgb_nmr.booster_.feature_importance(importance_type='gain')
    tg_imp_df = pd.DataFrame({'feature_code':lgb_nmr.booster_.feature_name(),
                                  'Imp_iter'+str(fold_id):lgb_nmr.booster_.feature_importance(importance_type='gain')})
    tg_imp_df['Imp_iter'+str(fold_id)] = tg_imp_df['Imp_iter'+str(fold_id)]/tg_imp_df['Imp_iter'+str(fold_id)].sum()#归一化
    imp_df = pd.merge(imp_df, tg_imp_df, how = 'left', on = 'feature_code')
pred_df = pd.DataFrame({'eid': eid_lst, 'target_y': y_test_lst, 
                            'y_pred_nmr': y_pred_nmr_lst})

pred_df.to_csv(dpath + 'Output/PPMI/pro_lightgbm_optuna'  + '.csv', index=False)


