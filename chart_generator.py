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
    full_file_name = os.path.join('exp_results_out', file_name)
    return pd.read_csv(full_file_name, delimiter=',')


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
    eval_func = 'xie_beni' 
    dataset = 'diabetes'

    df_logapso = load_results('logapso', eval_func, dataset)
    #df_hgapso = load_results('hgapso', eval_func, dataset)
    df_pso = load_results('pso', eval_func, dataset)
    
    best_logapso = get_best_result(df_logapso)
    average_logapso = list(get_average_fitness(best_logapso, 5))
    #best_hgapso = get_best_result(df_hgapso)
    #average_hgapso = list(get_average_fitness(best_hgapso, 5))
    best_pso = get_best_result(df_pso)
    average_pso = list(get_average_fitness(best_pso, 5))

    dict_fitness = {'logapso':average_logapso, 'pso':average_pso}#'hgapso':average_hgapso,

    save_dat_file(dict_fitness, eval_func, dataset)

    print(plot_results_chart(pd.DataFrame(dict_fitness), 
                       eval_func, eval_func + '_' + dataset))
