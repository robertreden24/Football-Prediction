import pandas as pd
import numpy as np
import scipy
import sys
import bookie_package as bp
import warnings
warnings.filterwarnings('ignore')

df_home = pd.read_excel('df_both_seasons_home.xlsx')
df_away = pd.read_excel('df_both_seasons_away.xlsx')

cols_to_use = df_home.columns.difference(df_away.columns)


df_both = pd.merge(df_away, df_home[cols_to_use], left_index=True, right_index=True, how='outer')

del df_both['Unnamed: 0']

df_both['pred_HTGDIFF'] = df_both['FTHG'] - df_both['FTAG']
df_both['pred_ATGDIFF'] = df_both['FTAG'] - df_both['FTHG']

df_both.rename(columns={"HTGDIFF": "test_HTGDIFF", "ATGDIFF": "test_ATGDIFF", 'FTHG': 'pred_FTHG', 'FTAG':'pred_FTAG'}, inplace=True)

df_both = df_both.reindex(columns = ['Day', 'Month', 'Year', 'HomeTeam', 'AwayTeam', 'pred_FTHG', 'pred_FTAG',
       'test_HTGDIFF', 'pred_HTGDIFF', 'test_ATGDIFF', 'pred_ATGDIFF', 'AVGATGDIFF', 'AVGFTAG','AVGFTHG', 'AVGHTGDIFF'])

df_both.to_excel('both.xlsx')

errors = abs(df_both['pred_HTGDIFF'] - df_both['test_HTGDIFF'])
accuracy = (errors==0).sum() / len(errors) * 100
print('MAE:', round(np.mean(errors),2), 'Goals.')
print('Accuracy:', round(accuracy, 2), '%.')

total_wins=(df_both["pred_HTGDIFF"] > 0).sum()
total_draw=(df_both["pred_HTGDIFF"] == 0).sum()
total_loss=(df_both["pred_HTGDIFF"] < 0).sum()


common_win = ((df_both["test_HTGDIFF"] > 0) & (df_both["pred_HTGDIFF"] > 0)).sum()
common_draw = ((df_both["test_HTGDIFF"] == 0) & (df_both["pred_HTGDIFF"] == 0)).sum()
common_lost = ((df_both["test_HTGDIFF"] < 0) & (df_both["pred_HTGDIFF"] < 0)).sum()

print('Correct Prediction Total: {} %'.format(np.round(((common_win+common_draw+common_lost)/df_both.shape[0]) * 100,2)))
print('Correct Prediction Share Wins: {} %'.format(np.round((common_win /total_wins)*100, 2)))
print('Correct Prediction Share Draws: {} %'.format(np.round((common_draw / total_draw)*100,2)))
print('Correct Prediction Share Lost: {} %'.format(np.round((common_lost / total_loss)*100,2)))