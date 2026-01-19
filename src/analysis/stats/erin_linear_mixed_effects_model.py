import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from scipy.stats import ttest_rel
import statsmodels.api as sm
import statsmodels.formula.api as smf

# mixed linear effects model
# PostErrorRT ~ PreviousErrorType (is or cr) * thisTrialCongruency (i or c) * thisTrialSwitchType (s or r) 
# + IncongruentProportion (25% or 75%) + SwitchProportion (25% or 75%) + (1 | Subject)

# setting up dataframe to have necessary columns
raw_data = pd.read_csv('/Users/erinburns/Library/CloudStorage/Box-Box/CoganLab/D_Data/GlobalLocal/combinedData.csv')

raw_data['prev_acc'] = raw_data.groupby('subject_ID')['acc'].shift(1)
raw_data['prev_congruency'] = raw_data.groupby('subject_ID')['congruency'].shift(1)
raw_data['prev_switch'] = raw_data.groupby('subject_ID')['switchType'].shift(1)

conditions = [
    (raw_data['prev_acc'] == 0) & (raw_data['prev_congruency'] == 'i') & (raw_data['prev_switch'] == 'r'),
    (raw_data['prev_acc'] == 0) & (raw_data['prev_congruency'] == 'c') & (raw_data['prev_switch'] == 's')
]
choices = ['ir', 'cs']

raw_data['PreviousErrorType'] = np.select(conditions, choices, default='None')

# new model dataframe with only the necessary rows with either iR or cS as previous error
model_df = raw_data[raw_data['PreviousErrorType'].isin(['ir', 'cs']) & (raw_data['acc'] == 1)].copy()

##fix blockType stuff to make new IncongruentProp and SwitchProp columns
block_map = {
    'A': {'CongruentProp': 0.75, 'SwitchProp': 0.25},
    'B': {'CongruentProp': 0.25, 'SwitchProp': 0.75},
    'C': {'CongruentProp': 0.75, 'SwitchProp': 0.25},
    'D': {'CongruentProp': 0.25, 'SwitchProp': 0.75}
}
model_df['CongruentProp'] = model_df['blockType'].map(lambda x: block_map[x]['CongruentProp'])
model_df['SwitchProp'] = model_df['blockType'].map(lambda x: block_map[x]['SwitchProp'])

##to make categorical - do I need??
model_df['CongruentProp'] = model_df['CongruentProp'].astype(str)
model_df['SwitchProp'] = model_df['SwitchProp'].astype(str)

#model
formula = "RT ~ PreviousErrorType * congruency * switchType + CongruentProp + SwitchProp"

model = smf.mixedlm(formula, data=model_df, groups=model_df['subject_ID'])
result = model.fit()

print(result.summary())