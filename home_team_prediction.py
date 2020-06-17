import pandas as pd
import seaborn as sns
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.style as style
from sklearn.ensemble import RandomForestRegressor
import sys
from sklearn.model_selection import train_test_split
import pydot

import bookie_package as bp
from sklearn.tree import export_graphviz

import warnings
warnings.filterwarnings('ignore')

pd.options.display.max_columns = None

df_both_seasons_home = pd.read_pickle('df_both_seasons_essentials')

d_both_seasons = bp.averages.avg_goal_diff(df_both_seasons_home, 'AVGHTGDIFF', 'HomeTeam', 'H')

df_both_seasons = bp.averages.from_dict_value_to_df(d_both_seasons)

df_both_seasons=df_both_seasons.sort_values(['Year', 'Month','Day'], ascending=False)

avg_fthg_per_team=bp.averages.avg_goals(df_both_seasons, 'AVGFTHG', 'HomeTeam', 'H')


df_both_seasons = bp.averages.from_dict_value_to_df(avg_fthg_per_team)

df_both_seasons=df_both_seasons.sort_values(['Year', 'Month','Day'], ascending=False)

print(df_both_seasons.head())

team_with_past_HTGDIFF=bp.averages.previous_data(df_both_seasons, 'HomeTeam', 'HTGDIFF')

df_team_with_past_HTGDIFF = bp.averages.from_dict_value_to_df(team_with_past_HTGDIFF)

columns_HTGDIFF = [
    'Day', 'Month', 'Year', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG',
    'HTGDIFF', 'ATGDIFF', 'AVGHTGDIFF','AVGFTHG', 'HST', 'AST',  'HTGDIFF_1', 'HTGDIFF_2', 'HTGDIFF_3', 'HTGDIFF_4', 'HTGDIFF_5', 'HTGDIFF_6', 'HTGDIFF_7',
    'HTGDIFF_8', 'HTGDIFF_9', 'HTGDIFF_10'
]
df_team_with_past_HTGDIFF = df_team_with_past_HTGDIFF.reindex(columns=columns_HTGDIFF)

df_team_with_past_HTGDIFF.fillna(0, inplace=True)

team_with_past_HST=bp.averages.previous_data(df_team_with_past_HTGDIFF, 'HomeTeam', 'HST')

df_team_with_past_HST = bp.averages.from_dict_value_to_df(team_with_past_HST)

columns_HST =  ['HST_1', 'HST_2', 'HST_3', 'HST_4', 'HST_5', 'HST_6', 'HST_7', 'HST_8', 'HST_9', 'HST_10']
columns_HTGDIFF_HST = columns_HTGDIFF + columns_HST

df_team_with_past_HST = df_team_with_past_HST.reindex(columns=columns_HTGDIFF_HST)

df_team_with_past_HST.fillna(0, inplace=True)

team_with_past_FTHG = bp.averages.previous_data(df_team_with_past_HST, 'HomeTeam', 'FTHG')

df_team_with_past_FTHG = bp.averages.from_dict_value_to_df(team_with_past_FTHG)

columns_FTHG = ['FTHG_1', 'FTHG_2', 'FTHG_3', 'FTHG_4', 'FTHG_5', 'FTHG_6', 'FTHG_7', 'FTHG_8', 'FTHG_9', 'FTHG_10']
columns_HTGDIFF_HST_FTHG = columns_HTGDIFF_HST + columns_FTHG

df_team_with_past_FTHG = df_team_with_past_FTHG.reindex(columns=columns_HTGDIFF_HST_FTHG)

df_team_with_past_FTHG.fillna(0, inplace=True)

df_result = df_team_with_past_FTHG.copy()

df_result = df_result.drop(['HomeTeam', 'AwayTeam'], axis = 1)

print(df_result.head())


target = df_result['FTHG']


df_result= df_result.drop([
    'FTHG','FTAG', 'HTGDIFF', 'ATGDIFF', 'HST', 'AST', 'HTGDIFF_7', 'HTGDIFF_9',
    'HTGDIFF_10', 'FTHG_5', 'FTHG_8', 'FTHG_10', 'FTHG_4', 'FTHG_7', 'HST_6',
    'FTHG_3'
], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(
    df_result, target, test_size = 0.25,random_state = 42
)

print('X_train Shape:', X_train.shape)
print('y_train Shape:', y_train.shape)
print('X_test Shape:', X_test.shape)
print('y_test Shape:', y_test.shape)



features_names = list(df_result.columns)
X_train = np.array(X_train)
base = X_train[:, features_names.index('AVGFTHG')]

baseline_errors = abs(base - y_train)
print('MAE: ', round(np.mean(baseline_errors), 2), 'Goals.')

rf = bp.prediction.random_forrest(X_train, y_train, n_estimators=1000,random_state = 42)

print(bp.prediction.performance_accuracy(y_test,X_test, rf))

features=np.array(df_result)
predictions_FTHG = rf.predict(features)
next_games_predictions=np.round(predictions_FTHG,0)

df_both_seasons['FTHG'] = next_games_predictions
print(df_both_seasons.head())

rf_depth_4 = bp.prediction.random_forrest(X_train, y_train, n_estimators=10,random_state = 42, max_depth = 4)


tree_4 = rf_depth_4.estimators_[7]


export_graphviz(tree_4, out_file = 'tree_4_home.dot', feature_names = features_names, rounded = True, precision = 1)


(graph, ) = pydot.graph_from_dot_file('tree_4_home.dot')
graph.write_png('tree_4_home.png')

print('Depth of this tree:', tree_4.tree_.max_depth)


importance = np.round(rf.feature_importances_,4)
dictionary = dict(zip(features_names, importance))
sorted_dictionary=sorted(dictionary.items(), key=lambda x:x[1], reverse=True)
names=[]
values=[]
for i in range(0, len(importance)):
    print('Variable Importance: {:15} {}%'.format(
        sorted_dictionary[i][0], np.round(sorted_dictionary[i][1]*100,4))
         )
    names.append(sorted_dictionary[i][0])
    values.append(np.round(sorted_dictionary[i][1]*100,4))

sns.set(style='whitegrid', rc={'figure.figsize':(11.7,8.27)})
sns.set_context('talk')


import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
bottom, top = plt.ylim()
bottom = 0
cum_values=np.cumsum(values)
plt.plot(names,cum_values, '--bo', color='r')

plt.axhline(95,color='black')
plt.xticks(rotation=90)
plt.xlabel('Feature')
plt.ylabel('Percentage')
plt.title('Cumulative Feature Importance')

from sklearn.model_selection import RandomizedSearchCV

rs = bp.prediction.random_search(X_train,y_train,cv=10)

best_params = rs.best_params_

best_params

rfc = bp.prediction.random_forrest(
    X_train, y_train,
    n_estimators=best_params['n_estimators'],
    random_state = 42,
    min_samples_split = best_params['min_samples_split'],
    max_leaf_nodes = best_params['max_leaf_nodes'],
    max_features = best_params['max_features'],
    max_depth = best_params['max_depth'],
    bootstrap = best_params['bootstrap']
)

print(bp.prediction.performance_accuracy(y_test,X_test, rfc))

next_games=df_result

predictions_next_games = rf.predict(next_games)
next_games_predictions=np.round(predictions_next_games,0)

df_both_seasons['FTHG'] = next_games_predictions
print(df_both_seasons.head())


df_both_seasons.to_excel('df_both_seasons_home.xlsx')