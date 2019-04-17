import numpy as np
import pandas as pd
import json
import pso
import hybrid_pso_ga
import logapso
import functions
import os
import itertools as it

def read_json(file_name):
    with open(file_name) as json_file:
        params = json.load (json_file)
    return params


def save_results(algorithm, function, index_params, df_results):
    """Save the results of experiments to a csv file."""
    results_dir = os.path.join (os.path.dirname(__file__), 'exp_results')
    file_name = f"{algorithm}_params_{function}_{index_params}.csv"

    df_results.to_csv(os.path.join(results_dir, file_name), index=False)
    print(f'{file_name} succesfully saved.')


def create_grid_params(list_params):
    """Transform a dictionary that returns lists to a list
    of dictionaries containing all possible combination of parameters.

    Example:
        input: {'max_iters':[100,200], 'pop_size':[30,50]}
        output:[{'max_iters':100, 'pop_size':30},
                {'max_iters':100, 'pop_size':50},
                {'max_iters':200, 'pop_size':30},
                {'max_iters':200, 'pop_size':50}]
    """
    # TODO
    #res = [[]]
    #for _, vals in list_params.items():
    #    res = [x+[y] for x in res for y in vals]
    #print(res)
    return list_params

def run_pso_experiments(list_params, benchmark_funcs, n_runs):
    """Execute experiments 'n_runs' times for each group of
        PSO parameters and different benchmark functions.
    """
    pso_params = create_grid_params(list_params)
    for func in benchmark_funcs: 
        eval_func, l_bound, u_bound, task = functions.get_function(
                                            params['function'])
        for param in pso_params:
            pso.run_pso(
                #param['']
            )
    #benchmark_funcs = {'eval_func': eval_func,
    #                    'l_bound': l_bound,
    #                    'u_bound': u_bound,
    #                    'task': task}
    return None

def run_experiments(n_runs, params):
    """Run a group of experiments for each optimisation algorithm."""
    # Number of times each group of experiments is run
    n_runs = 5
    algorithms = ['pso', 'hgapso', 'logapso']
    benchmark_funcs = params['function']
    
    for alg in algorithms:
        if alg == 'pso':
            df_results = run_pso_experiments(
                    params['pso'], benchmark_funcs, n_runs)
    
if __name__ == '__main__':
    params = read_json ('parameters.json')
