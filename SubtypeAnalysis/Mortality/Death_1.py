import numpy as np
import pandas as pd

yaxing_df = pd.read_csv('..') 


mor_path =' /Users/lijiawei/Desktop/Mortality-2-LJW/'
dpath = '/Users/lijiawei/Desktop/BloodData/'
death_df = pd.read_csv('/Users/lijiawei/Desktop/Mortality-2-LJW/All-cause.csv',usecols=['eid','BL2Death_yrs'])
outcomes_df = pd.read_csv(dpath+'PD_outcomes.csv',usecols=['eid', 'BL2Target_yrs'])
tgt_df = pd.read_csv('/Users/lijiawei/Desktop/Mortality-2-LJW/Primary-Secondary-G.csv',usecols=['eid','target_date'])


mydf = pd.merge(yaxing_df,death_df, how='left', on=['eid'])
mydf = pd.merge(mydf,tgt_df, how='left', on=['eid'])
mydf = pd.merge(mydf,outcomes_df, how='left', on=['eid'])


mydf['Death_Target_Diff'] = np.where(mydf['BL2Death_yrs'].isna(), np.nan, mydf['BL2Death_yrs'] - mydf['BL2Target_yrs'])
print(mydf['Death_Target_Diff'].describe())

grouped = mydf.groupby('yaxing_label')
results = {}

for label, group in grouped:
    total_count = len(group)
    death_count = len(group[ ~group['target_date'].isna()])
    print('Total Count:', total_count)
    print('Death Count:', death_count)
    death_rate = death_count / total_count
    five_year_death_count = len(group[
    (~group['target_date'].isna()) &
    ((group['BL2Death_yrs'] - group['BL2Target_yrs']) <= 5)
    ])
    five_year_death_rate = five_year_death_count / total_count
    print('Five Year Death Count:', five_year_death_count)
    results[label] = {
        'death_rate': death_rate,
        'five_year_death_rate': five_year_death_rate
    }

for label, rates in results.items():
    print(f"Group {label}:")
    print(f"  Death Rate: {rates['death_rate']:.4f}")
    print(f"  Five-Year Death Rate: {rates['five_year_death_rate']:.4f}")

