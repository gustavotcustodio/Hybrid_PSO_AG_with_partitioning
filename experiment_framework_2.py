import numpy as np
import pandas as pd
import json
import pso
import hybrid_pso_ga
import logapso
import functions
import os

def read_json(file_name):
    """Read a Json file with the parameters for optimization algorithms."""
    with open(file_name) as json_file:
        params = json.load (json_file)
    return params


def save_results(algorithm, benchmark_func, df_results):
    """Save the results of experiments to a csv file.
    
    Parameters
    ----------
    algorithm: string
        Name of optimization algorithm.
    index_params: int
        Index of parameters used for the optimization algorithm.
    df_results: Dataframe
        Dataframe containing the results of experiments.
    """
    results_dir = os.path.join(os.path.dirname(__file__), 'exp_results')
    file_name = f"results_{algorithm}_{benchmark_func}.csv"

    df_results.to_csv(os.path.join(results_dir, file_name), index=False)
    print(f'{file_name} succesfully saved.')


def create_grid_params(dict_params):
    """Transform a dictionary that returns lists to a list
    of dictionaries containing all possible combination of 
    parameters (cartesian product).

    Example:
        input: {'max_iters':[100,200], 'pop_size':[30,50]}
        output:[{'max_iters':100, 'pop_size':30},
                {'max_iters':100, 'pop_size':50},
                {'max_iters':200, 'pop_size':30},
                {'max_iters':200, 'pop_size':50}]
    """
    cartesian_params = [[]]
    # Generate the cartesian product of all possible parameters
    for vals in dict_params.values():
        cartesian_params = [p+[v] for p in cartesian_params for v in vals]

    keys = dict_params.keys()
    final_params = [dict(zip(keys, params)) for params in cartesian_params]
    return final_params


def run_logapso_experiments(list_pso_params, list_ga_params, func_name, n_runs):
    """Execute experiments with the LOGAPSO 'n_runs' times for each
    combination of PSO and GA parameters for a given benchmark function.
    """
    # Remove the key 'n_gens' because this param won't be used by the
    # LOGAPSO algorithm.
    del list_ga_params['n_gens']
    all_params = {}
    all_params.update(list_pso_params)
    all_params.update(list_ga_params)
    
    pso_ga_params = create_grid_params(all_params)
    eval_func, l_bound, u_bound, task = functions.get_function(func_name)

    df_results = pd.DataFrame(columns=['run', 'fitness', 'omega', 'c1', 'c2'])

    for p in pso_ga_params:
        for run in range(n_runs):
            print(f'-------- LOGAPSO - run {run+1} --------')

            #_, _, best_evals = logapso.run_logapso(

            #    )
            #df_new_res = pd.DataFrame(
            #    {'run':[run+1]*len(best_evals),
            #    'fitness':best_evals,
            #    'omega':[p['consts'][0]]*len(best_evals),
            #    'c1':[p['consts'][1]]*len(best_evals),
            #    'c2':[p['consts'][2]]*len(best_evals)
            #    })
            #df_results = df_results.append(df_new_res, ignore_index=True)
    return df_results


def run_pso_experiments(list_params, func_name, n_runs):
    """Execute experiments with the PSO algorithm 'n_runs' times for each
    group of PSO parameters and a given benchmark function.
    """
    pso_params = create_grid_params(list_params)
    eval_func, l_bound, u_bound, task = functions.get_function(func_name)

    df_results = pd.DataFrame(columns=['run', 'fitness', 'omega', 'c1', 'c2'])

    for p in pso_params:
        for run in range(n_runs):
            print(f'-------- PSO - run {run+1} --------')

            _, _, best_evals = pso.run_pso(
                eval_func=eval_func, consts=p['consts'], 
                max_iters=p['max_iters'], pop_size=p['pop_size'],
                particle_size=p['particle_size'], l_bound=l_bound,
                u_bound=u_bound, task=task
                )
            df_new_res = pd.DataFrame(
                {'run':[run+1]*len(best_evals),
                'fitness':best_evals,
                'omega':[p['consts'][0]]*len(best_evals),
                'c1':[p['consts'][1]]*len(best_evals),
                'c2':[p['consts'][2]]*len(best_evals)
                })
            df_results = df_results.append(df_new_res, ignore_index=True)
    return df_results


def run_experiments(n_runs, params):
    """Run a group of experiments for each optimisation algorithm."""
    # Number of times each group of experiments is run
    n_runs = 5
    algorithms = ['pso']
    benchmark_funcs = params['function']
    
    for alg in algorithms:
        for func in benchmark_funcs:
            if alg == 'pso':
                df_results = run_pso_experiments(
                                params['pso'], func, n_runs)
            save_results(alg, func, df_results)


if __name__ == '__main__':
    params = read_json ('parameters.json')
