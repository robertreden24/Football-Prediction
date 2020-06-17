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
#imports the own created package
import bookie_package as bp
from sklearn.tree import export_graphviz

import warnings
warnings.filterwarnings('ignore')

pd.options.display.max_columns = None

df_both_seasons_away = pd.read_pickle('df_both_seasons_essentials')

print(df_both_seasons_away.head())

d_both_seasons = bp.averages.avg_goal_diff(df_both_seasons_away, 'AVGATGDIFF', 'AwayTeam', 'A')


df_both_seasons = bp.averages.from_dict_value_to_df(d_both_seasons)


df_both_seasons=df_both_seasons.sort_values(['Year', 'Month','Day'], ascending=False)

avg_ftag_per_team=bp.averages.avg_goals(df_both_seasons, 'AVGFTAG', 'AwayTeam', 'A')

df_both_seasons = bp.averages.from_dict_value_to_df(avg_ftag_per_team)


df_both_seasons=df_both_seasons.sort_values(['Year', 'Month','Day'], ascending=False)

team_with_past_AST=bp.averages.previous_data(df_both_seasons, 'AwayTeam', 'AST')

df_team_with_past_AST = bp.averages.from_dict_value_to_df(team_with_past_AST)

columns_AST = [
    'Day', 'Month', 'Year', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HTGDIFF', 'ATGDIFF', 'AVGATGDIFF', 'AVGFTAG',
    'HST', 'AST', 'AST_1', 'AST_2', 'AST_3', 'AST_4', 'AST_5', 'AST_6', 'AST_7', 'AST_8', 'AST_9', 'AST_10'
    ]

df_team_with_past_AST = df_team_with_past_AST.reindex(columns=columns_AST)

df_team_with_past_AST.sort_values(['Year', 'Month','Day'], ascending=False,inplace=True)

df_team_with_past_AST.fillna(0, inplace=True)

team_with_past_FTAG = bp.averages.previous_data(df_team_with_past_AST, 'AwayTeam', 'FTAG')


df_team_with_past_FTAG = bp.averages.from_dict_value_to_df(team_with_past_FTAG)

columns_FTAG = ['FTAG_1', 'FTAG_2', 'FTAG_3', 'FTAG_4', 'FTAG_5', 'FTAG_6', 'FTAG_7', 'FTAG_8', 'FTAG_9', 'FTAG_10']
columns_AST_FTHG = columns_AST + columns_FTAG


df_team_with_past_FTAG = df_team_with_past_FTAG.reindex(columns=columns_AST_FTHG)

df_team_with_past_FTAG.sort_values(['Year', 'Month','Day'], ascending=False,inplace=True)

df_team_with_past_FTAG.fillna(0, inplace=True)

print(df_team_with_past_FTAG.columns)


df_result = df_team_with_past_FTAG.copy()

df_result = df_result.drop(['HomeTeam', 'AwayTeam'], axis = 1)

print('Shape of features:', df_result.shape)


target = df_result['FTAG']


df_result= df_result.drop([
    'Day','AST_4','AST_5','AST_3', 'AST_7', 'AST_8','AST_6',  'AST_10', 'AST_9', 'Year','FTAG','FTHG', 'HTGDIFF', 'ATGDIFF', 'HST', 'AST'
], axis = 1)


# splitting arrays into random train and test subsets
X_train, X_test, y_train, y_test = train_test_split(
    df_result, target, test_size = 0.25,random_state = 42
)


print('X_train Shape:', X_train.shape)
print('y_train Shape:', y_train.shape)
print('X_test Shape:', X_test.shape)
print('y_test Shape:', y_test.shape)

features_names = list(df_result.columns)
X_train = np.array(X_train)
base = X_train[:, features_names.index('AVGFTAG')]
# subtracting train result from test data
baseline_errors = abs(base - y_train)
print('MAE: ', round(np.mean(baseline_errors), 2), 'Goals.')

rf = bp.prediction.random_forrest(X_train, y_train, n_estimators=1000,random_state = 42)

print(bp.prediction.performance_accuracy(y_test,X_test, rf))

next_games=df_result
predictions_next_games = rf.predict(next_games)
next_games_predictions=np.round(predictions_next_games,0)

del df_both_seasons['FTHG']

df_both_seasons['FTAG'] = next_games_predictions
print(df_both_seasons.head())


rf_depth_4 = bp.prediction.random_forrest(X_train, y_train, n_estimators=10,random_state = 42, max_depth = 4)

# randomly pick one tree from ten
tree_4 = rf_depth_4.estimators_[7]

# use export_graphviz to save the tree as a dot file first as indicated:
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html
export_graphviz(tree_4, out_file = 'tree_4_away.dot', feature_names = features_names, rounded = True, precision = 1)

# then use the dot file to create a png file
(graph, ) = pydot.graph_from_dot_file('tree_4_away.dot')
graph.write_png('tree_4_away.png');

print('Depth of this tree:', tree_4.tree_.max_depth)

# creates a list of feature names and their importance
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
rs = bp.prediction.random_search(X_train,y_train, cv=5)

best_params = rs.best_params_

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

df_both_seasons['FTAG'] = next_games_predictions
print(df_both_seasons.head())

df_both_seasons.to_excel('df_both_seasons_away.xlsx')