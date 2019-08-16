import pandas as pd
import os
import numpy as np
import json

def get_best_result(df_results, n_dims):
    if 'particle_size' in df_results.columns:
        index_best = df_results.loc[
                        df_results['particle_size']==n_dims,
                        'fitness'].idxmin()
    else:
        index_best = df_results.loc[
                        df_results['n_clusters']==n_dims, 
                        'fitness'].idxmin()
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
    full_file_name = os.path.join('exp_tests', file_name)   
    try:
        df_results = pd.read_csv(full_file_name, delimiter=',')
    except:
        return None
    return df_results


def plot_results_chart(df_fitness, func_eval, file_name):
    tex_chart = open('model_tex_chart.tex').read()
    min_val = df_fitness.values.min()
    max_val = df_fitness.values.max()
    func_eval = func_eval.replace('_','-').capitalize()

    output = tex_chart.format(func_eval, min_val, max_val, file_name)
    output_file = open(os.path.join('graphs', file_name+'.tex'), 'w')
    
    output_file.write(output)
    output_file.close()


def save_dat_file(dict_fitness, func_eval, dataset=None):
    if dataset is not None:
        file_name = '{}_{}.dat'.format(func_eval, dataset)
    else:
        file_name = '{}.dat'.format(func_eval)

    full_save_path = os.path.join('graphs', 'dat_files')
    full_save_path = os.path.join(full_save_path, file_name)
    
    pd.DataFrame(dict_fitness).to_csv(full_save_path, sep=',', index=False)


def analyze_results():
    params = json.load(open('parameters.json'))
    algorithms = ['pso', 'hgapso','logapso']
    datasets = params['clustering'].keys()
    clustering_eval = ['xie_beni', 'fuku_sugeno']
    benchmark_funcs = params['function']
    
    #for cl_eval in clustering_eval:
    #    for dataset in datasets:
    #        dict_fitness = {}
    #        for alg in algorithms:
    #            df_results = load_results(alg, cl_eval, dataset)
                #get_best_result(df_results, n_dims)
                #mean_fitness_vals = get_average_fitness(df_results, 5)
                #dict_fitness.update({alg: mean_fitness_vals})
                #save_dat_file(dict_fitness, cl_eval, dataset)

    particle_sizes = params['pso']['particle_size']
    for b in benchmark_funcs:
       
        list_df_results = [load_results(alg, b) for alg in algorithms]
        for n_dims in particle_sizes: 
            dict_fitness = {}
            for alg, df_results in zip(algorithms, list_df_results):
                if df_results is not None:
                    best_values = get_best_result(df_results, n_dims)
                    average_values = get_average_fitness(best_values, n_runs=5)
                    dict_fitness.update({alg: average_values})
            save_dat_file(dict_fitness, b)
                
if __name__=='__main__':
    analyze_results()
    #print(plot_results_chart(pd.DataFrame(dict_fitness), 
    #                   eval_func, eval_func + '_' + dataset))
