import pandas as pd
import os

path = os.path.join(os.path.dirname(__file__), 'exp_results')

df_iris = pd.read_csv(os.path.join(path,
                      'results_logapso_fuku_sugeno_iris.csv'))
df_wine = pd.read_csv(os.path.join(path,
                      'results_logapso_fuku_sugeno_wine.csv'))
df_wdbc = pd.read_csv(os.path.join(path,
                      'results_logapso_fuku_sugeno_wdbc.csv'))
df_ionosphere = pd.read_csv(os.path.join(path,
                            'results_logapso_fuku_sugeno_ionosphere.csv'))
df_diabetes = pd.read_csv(os.path.join(path,
                          'results_logapso_fuku_sugeno_diabetes.csv'))

df_iris.drop(['fitness', 'run'], axis=1, inplace=True)
df_wine.drop(['fitness', 'run'], axis=1, inplace=True)
df_wdbc.drop(['fitness', 'run'], axis=1, inplace=True)
df_ionosphere.drop(['fitness', 'run'], axis=1, inplace=True)
df_diabetes.drop(['fitness', 'run'], axis=1, inplace=True)

print('Iris: {}'.format(
        len(df_iris.groupby(list(df_iris.columns)).nunique())))
print('Wine: {}'.format(
        len(df_wine.groupby(list(df_wine.columns)).nunique())))
print('Breast cancer: {}'.format(
        len(df_wdbc.groupby(list(df_wdbc.columns)).nunique())))
print('Ionosphere: {}'.format(
        len(df_ionosphere.groupby(list(df_ionosphere.columns)).nunique())))
print('Diabetes: {}'.format(
        len(df_diabetes.groupby(list(df_diabetes.columns)).nunique())))
