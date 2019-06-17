import pandas as pd
import os
import json

#def get_best_result(df_results):
#    index_best = df['fitness'].idxmax()
#    df_results.iloc[index_best].columns
#    df_results.groupby
#    print('Iris: {}'.format(
#        len(df_iris.groupby(list(df_iris.columns)).nunique())))

def load_results(alg, func_eval, dataset=None):
    if dataset is None:
        file_name = 'results_{}_{}.csv'.format(alg, func_eval)
    else:
        file_name = 'results_{}_{}_{}.csv'.format(alg, func_eval, dataset)
    full_file_name = os.path.join('exp_results_backup', file_name)
    return pd.read_csv(full_file_name, delimiter=',')
    
#def create_dat_file(df_results):

def create_dat_file(df_results):
    params = json.load('parameters.json')
    algorithms = ['PSO', 'HGAPSO','LOGAPSO']
    datasets = params['clustering'].keys()
    clustering_eval = ['xie_beni', 'daviels_bouldin']
    benchmark_funcs = params['clustering']
    
    #for alg in algorithms:    
    #    for b in benchmark_funcs:
            #load_results(alg, )
    

if __name__=='__main__':
    print('a')