import numpy as np
import pandas as pd

yaxing_df = pd.read_csv('..')
mor_path =' ..'
dpath = '..'

death_df = pd.read_csv('/..',usecols=['eid','BL2Death_yrs'])
tgt_yrs_df = pd.read_csv(dpath+'PD_outcomes.csv',
                         usecols=['eid', 'BL2Target_yrs'])

mydf = pd.merge(yaxing_df,tgt_yrs_df, how='left', on=['eid'])
mydf = pd.merge(mydf,death_df, how='left', on=['eid'])
grouped = mydf.groupby('yaxing_label')
results = {}

for label, group in grouped:
    total_count = len(group)
    death_count = len(group[ ~group['BL2Death_yrs'].isna()])
    death_rate = death_count / total_count
    print('Total Count:', total_count)
    print('Death Count:', death_count)
    five_year_death_count = len(group[
    (~group['BL2Death_yrs'].isna()) &
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
