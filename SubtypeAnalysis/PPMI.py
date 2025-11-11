import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
import re
import optuna

import optuna
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold

def get_sig_omic_new(dpath, omic):
    omic_f_df = pd.read_csv(dpath + 'Association/'+omic+'_Step1.csv')
    # omic_f_df = omic_f_df.loc[omic_f_df.pval_fdr < 0.05]
    omic_f_lst = omic_f_df.Omics_feature.tolist()
    omic_df = pd.read_csv(dpath +omic+'Data/'+omic+'Data_Raw.csv', usecols=['eid'] + omic_f_lst)
    return (omic_f_lst, omic_df)


ppmipath = '..'
 
ppmi_df = pd.read_csv(ppmipath+'clinical_filtered_features_with_targets_BLSC.csv')
info_df = pd.read_csv(ppmipath+'cli_common.csv')
listcode_values = info_df['LTSTCODE'].values
columns_to_keep = ['PATNO','target_y','COHORT']+ [col for col in ppmi_df.columns if col in listcode_values]
ppmi_df = ppmi_df[columns_to_keep]
ppmi_df = ppmi_df[ppmi_df['target_y'] == 1]

a1_df = pd.read_csv(ppmipath+'Apolipoprotein A1_data.csv', usecols=['PATNO', 'TESTVALUE', 'CLINICAL_EVENT'])
a1_df = a1_df[a1_df['CLINICAL_EVENT'] == 'BL']
a1_df = a1_df[['PATNO', 'TESTVALUE']]
a1_df.rename(columns = {'TESTVALUE':'30630-0.0'}, inplace = True)

hdl_df = pd.read_csv(ppmipath+'HDL_data.csv', usecols=['PATNO', 'TESTVALUE', 'CLINICAL_EVENT'])
hdl_df = hdl_df[hdl_df['CLINICAL_EVENT'] == 'BL']
hdl_df = hdl_df[['PATNO', 'TESTVALUE']]
hdl_df.rename(columns={'TESTVALUE': '30760-0.0'}, inplace=True)

ldl_df = pd.read_csv(ppmipath+'LDL_data.csv', usecols=['PATNO', 'TESTVALUE', 'CLINICAL_EVENT'])
ldl_df = ldl_df[ldl_df['CLINICAL_EVENT'] == 'BL']
ldl_df = ldl_df[['PATNO', 'TESTVALUE']]
ldl_df.rename(columns={'TESTVALUE': '30780-0.0'}, inplace=True)

tri_df = pd.read_csv(ppmipath+'Triglycerides_data.csv', usecols=['PATNO', 'TESTVALUE', 'CLINICAL_EVENT'])
tri_df = tri_df[tri_df['CLINICAL_EVENT'] == 'BL']
tri_df = tri_df[['PATNO', 'TESTVALUE']]
tri_df.rename(columns={'TESTVALUE': '30870-0.0'}, inplace=True)

cho_df = pd.read_csv(ppmipath+'Total_Cholesterol_data.csv', usecols=['PATNO', 'TESTVALUE', 'CLINICAL_EVENT'])
cho_df = cho_df[cho_df['CLINICAL_EVENT'] == 'BL']
cho_df = cho_df[['PATNO', 'TESTVALUE']]
cho_df.rename(columns={'TESTVALUE': '30690-0.0'}, inplace=True)

ppmi_df = pd.merge(ppmi_df, a1_df, how='inner', on=['PATNO'])
ppmi_df = pd.merge(ppmi_df, hdl_df, how='inner', on=['PATNO'])
ppmi_df = pd.merge(ppmi_df, ldl_df, how='inner', on=['PATNO'])
ppmi_df = pd.merge(ppmi_df, tri_df, how='inner', on=['PATNO'])
ppmi_df = pd.merge(ppmi_df, cho_df, how='inner', on=['PATNO'])
print(ppmi_df.shape)

# 'g/dL->g/L: RCT13 RCT12'
ppmi_df['RCT13'] = ppmi_df['RCT13'] * 10
ppmi_df['RCT12'] = ppmi_df['RCT12'] * 10

# 'mg/dL -> mmol/L'
ppmi_df['RCT183'] = ppmi_df['RCT183'] * 0.2495
ppmi_df['RCT6'] = ppmi_df['RCT6'] * 0.357
ppmi_df['30760-0.0'] = ppmi_df['30760-0.0'] * 0.0259
ppmi_df['30780-0.0'] = pd.to_numeric(ppmi_df['30780-0.0'], errors='coerce')
ppmi_df['30780-0.0'] = ppmi_df['30780-0.0'] * 0.0259
ppmi_df['30870-0.0'] = pd.to_numeric(ppmi_df['30870-0.0'], errors='coerce')
ppmi_df['30870-0.0'] = ppmi_df['30870-0.0'] * 0.0113
ppmi_df['30690-0.0'] = pd.to_numeric(ppmi_df['30690-0.0'], errors='coerce')
ppmi_df['30690-0.0'] = ppmi_df['30690-0.0'] * 0.0259
ppmi_df['RCT392'] = ppmi_df['RCT392'] * 88.4
ppmi_df['RCT1'] = pd.to_numeric(ppmi_df['RCT1'], errors='coerce') * 17.1
ppmi_df['30630-0.0'] = ppmi_df['30630-0.0'] * 1e-2

listcode_to_omics = dict(zip(info_df['LTSTCODE'], info_df['Omics_feature']))
new_columns = ppmi_df.columns[:3].tolist() + [listcode_to_omics.get(col, col) for col in ppmi_df.columns[3:]]
ppmi_df.columns = new_columns

tmp_f_lst = ppmi_df.columns[3:].tolist()

cov_df = pd.read_csv(ppmipath+'covariates.csv')
cov_df = cov_df[['PATNO', 'age', 'SEX']]
cov_df.rename(columns={'SEX': 'sex'}, inplace=True)
ppmi_df = pd.merge(ppmi_df, cov_df, how='inner', on=['PATNO'])

dpath = '..'
outpath = dpath+'Output/'

cli_f_lst,cli_df = get_sig_omic_new(dpath,'ClinicalLab')
bld_idx_df = pd.read_csv(dpath+'BloodDataIndex.csv',
                         usecols = ['eid','Step'])
tgt_df = pd.read_csv(dpath+'PD_outcomes.csv',usecols=['eid','BL2Target_yrs'])
yaxing_df = pd.read_csv(dpath+'labels.csv',usecols=['eid','cluster_label'])
yaxing_df['cluster_label'] = yaxing_df['cluster_label']-1
mydf = pd.merge(cli_df,yaxing_df, how = 'inner', on=['eid'])
mydf= pd.merge(mydf,bld_idx_df, how = 'inner', on=['eid'])
mydf = mydf[mydf['Step'] == 2]#（322，27）
covdf = pd.read_csv(dpath+'Covariates.csv')
covdf = covdf[['eid', 'age', 'sex']]
mydf = pd.merge(mydf, covdf, how='inner', on=['eid'])
mydf = pd.merge(mydf,tgt_df, how = 'inner', on=['eid'])
mydf = mydf[mydf['BL2Target_yrs'] <= 5]

my_params = {'n_estimators': 500,
             'max_depth': 15,
             'num_leaves': 10,
             'subsample': 0.7,
             'learning_rate': 0.01,
             'colsample_bytree': 0.7}
tmp_f_lst = ['30130-0.0', '30150-0.0', '30160-0.0', '30080-0.0', '30200-0.0', '30180-0.0', '30190-0.0', '30210-0.0', '30220-0.0', '30010-0.0', '30000-0.0', '30140-0.0', '30120-0.0', '30840-0.0', '30860-0.0', '30600-0.0', '30680-0.0', '30700-0.0', '30620-0.0', '30650-0.0', '30670-0.0', '30630-0.0', '30760-0.0', '30780-0.0', '30870-0.0', '30690-0.0']
tmp_f_lst+= ['age','sex']
X_train,X_test = mydf[tmp_f_lst], ppmi_df[tmp_f_lst]
y_train = mydf['cluster_label']


model = LGBMClassifier(**my_params)
model.fit(X_train, y_train)
y_pred_train = model.predict_proba(X_train)[:, 1]
y_pred_test = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_train, y_pred_train)
youden_index = tpr - fpr
optimal_idx = np.argmax(youden_index)
optimal_threshold = thresholds[optimal_idx]

y_pred_test_labels = (y_pred_test >= optimal_threshold).astype(int)
print(f'Optimal threshold to maximize Youden\'s index: {optimal_threshold:.3f}')

output_df = pd.DataFrame({'PATNO': ppmi_df['PATNO'], 'yaxing_label': y_pred_test_labels})
output_df.to_csv(outpath + 'PPMI_predicted_labels.csv', index=False)
label_counts = output_df['yaxing_label'].value_counts()
print("yaxing_label distribution:")
print(label_counts)
print('done!')