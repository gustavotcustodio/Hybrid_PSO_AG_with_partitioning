import pandas as pd
import os
import json

def get_best_result(df_results):
    index_best = df_results['fitness'].idxmin()
    df_duplicates = df_results.drop(['fitness', 'run'], axis=1)
    best_row = df_duplicates.iloc[index_best]

    filters = " & ".join(["(df_duplicates['{0}'] == best_row['{0}'])".
                         format(col) for col in df_duplicates.columns])
    return df_results.loc[eval(filters)]


def get_average_fitness(df_results, n_runs):
    n_iters = int(len(df_results) / n_runs)
    iters = [i%n_iters for i in range(len(df_results))]
    df_results['iter'] = iters
    return df_results.groupby(df_results['iter']).mean()['fitness']


def load_results(alg, func_eval, dataset=None):
    if dataset is None:
        file_name = 'results_{}_{}.csv'.format(alg, func_eval)
    else:
        file_name = 'results_{}_{}_{}.csv'.format(alg, func_eval, dataset)
    full_file_name = os.path.join('exp_results_backup', file_name)
    return pd.read_csv(full_file_name, delimiter=',')


def plot_results_chart(df_fitness):
    #df_fitness.max()
    tex_chart = open('model_tex_chart.tex').read()
    #text_chart.format()


def save_dat_file(dict_fitness, func_eval, dataset=None):
    if dataset is not None:
        file_name = '{}_{}.dat'.format(func_eval, dataset)
    else:
        file_name = '{}.dat'.format(func_eval)
    full_save_path = os.path.join('dat_files', file_name)
    pd.DataFrame(dict_fitness).to_csv(full_save_path, sep=',')


def save_fitness_values(df_results):
    params = json.load('parameters.json')
    algorithms = ['PSO', 'HGAPSO','LOGAPSO']
    datasets = params['clustering'].keys()
    clustering_eval = ['xie_beni', 'daviels_bouldin']
    benchmark_funcs = params['clustering']
    
    for cl_eval in clustering_eval:
        for dataset in datasets:
            dict_fitness = {}
            for alg in algorithms:
                df_results = load_results(alg, cl_eval, dataset)
                mean_fitness_vals = get_average_fitness(df_results, 5)
                dict_fitness.update({alg: mean_fitness_vals})

        #for b in benchmark_funcs:
        #    load_results(alg, )
    

if __name__=='__main__':
    df_results = load_results('logapso', 'davies_bouldin', 'iris')
    
    best_result = get_best_result(df_results)
    average_fitness = get_average_fitness(best_result, 5)
    save_dat_file({'logapso':average_fitness}, 'davies_bouldin', 'iris')
    #save_dat_file()
