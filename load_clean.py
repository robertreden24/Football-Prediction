import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import sys

import warnings
warnings.filterwarnings('ignore')

pd.options.display.max_columns = None

df_season_1=pd.read_csv('D1.csv')
df_season_2 = pd.read_csv('D1_last.csv')
df_season_2_second =pd.read_csv('D2.csv')

df_union_home= df_season_2_second[df_season_2_second['HomeTeam']=='Union Berlin']
df_paderborn_home= df_season_2_second[df_season_2_second['HomeTeam']=='Paderborn']
df_koln_home = df_season_2_second[df_season_2_second['HomeTeam']=='FC Koln']
df_union_away= df_season_2_second[df_season_2_second['AwayTeam']=='Union Berlin']
df_paderborn_away = df_season_2_second[df_season_2_second['AwayTeam']=='Paderborn']
df_koln_away = df_season_2_second[df_season_2_second['AwayTeam']=='FC Koln']
df_season_2_second = pd.concat([df_union_home, df_union_away, df_paderborn_home, df_paderborn_away, df_koln_home, df_koln_away], axis=0)

df_season_2= pd.concat([df_season_2,df_season_2_second], axis=0)

teams_s1 = df_season_1.columns.unique()

teams_s2 = df_season_2.columns.unique()

same_columns = np.intersect1d(teams_s1, teams_s2)

df_season_1 = df_season_1[same_columns]
df_season_2 = df_season_2[same_columns]

df_both_seasons = pd.concat([df_season_1, df_season_2], axis=0)

df_both_seasons['Date'] = pd.to_datetime(df_both_seasons['Date'], errors='coerce')


df_both_seasons['Day'] = df_both_seasons['Date'].dt.day
df_both_seasons['Month'] = df_both_seasons['Date'].dt.month
df_both_seasons['Year'] = df_both_seasons['Date'].dt.year

df_both_seasons_essentials = df_both_seasons[['Day','Month','Year','HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HST', 'AST']]

df_both_seasons_essentials['HTGDIFF'] = df_both_seasons_essentials['FTHG'] - df_both_seasons_essentials['FTAG']
df_both_seasons_essentials['ATGDIFF'] = df_both_seasons_essentials['FTAG'] - df_both_seasons_essentials['FTHG']


print(df_both_seasons_essentials.head())

df_both_seasons_essentials= df_both_seasons_essentials.sort_values(['Year', 'Month','Day'], ascending=False)

# Counts the amount of home + away games every team had during Bundesliga for the 2019/2020 season and 2018/2019
df_both_seasons_essentials.groupby('HomeTeam').count()

from pandas.plotting import scatter_matrix

print(df_both_seasons_essentials[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HST', 'AST', 'HTGDIFF', 'ATGDIFF']].describe())

import pickle
df_both_seasons_essentials.to_pickle('df_both_seasons_essentials')