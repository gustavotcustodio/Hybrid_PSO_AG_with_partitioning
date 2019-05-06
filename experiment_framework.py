import numpy as np
import pandas as pd
import json
import pso
import hybrid_pso_ga
import logapso
import hgapso
import functions
import os

def read_json(file_name):
    """
    Read a Json file with the parameters for optimization algorithms.
    """
    with open(file_name) as json_file:
        params = json.load (json_file)
    return params


def save_results(algorithm, benchmark_func, df_results):
    """Save the results of experiments to a csv file.
    
    Parameters
    ----------
    algorithm: string
        Name of optimization algorithm.
    benchmark_func: string
        Name of benchmark function.
    df_results: Dataframe
        Dataframe containing the results of experiments.
    """
    results_dir = os.path.join(os.path.dirname(__file__), 'exp_results')
    file_name = f"results_{algorithm}_{benchmark_func}.csv"

    df_results.to_csv(os.path.join(results_dir, file_name), index=False)
    print(f'{file_name} succesfully saved.')


def create_grid_params(dict_params):
    """
    Transform a dictionary that returns lists to a list
    of dictionaries containing all possible combination of 
    parameters (cartesian product).

    Example:
        input: {'max_iters':[100,200], 'pop_size':[30,50]}
        output:[{'max_iters':100, 'pop_size':30},
                {'max_iters':100, 'pop_size':50},
                {'max_iters':200, 'pop_size':30},
                {'max_iters':200, 'pop_size':50}]

    Parameters
    ----------
    dict_params: dict

    Returns
    -------
    final_params: list[dict]
        List containing a dict for each permutation of parameters.
    """
    cartesian_params = [[]]
    # Generate the cartesian product of all possible parameters
    for vals in dict_params.values():
        cartesian_params = [p+[v] for p in cartesian_params for v in vals]

    keys = dict_params.keys()
    final_params = [dict(zip(keys, params)) for params in cartesian_params]
    return final_params


def run_cluster_pso_experiments(list_pso_params, list_cluster_params,
                                func_name, n_runs, dataset_name):
    """
    Execute experiments using the PSO algorithm 'n_runs' times for each group 
    of PSO parameters, a clustering evaluation index and a dataset.

    Parameters
    ----------
    list_pso_params: dict
        Dictionary containing lists of different values for parameters.
    list_cluster_params: dict
        Parameters for clustering data (n_clusters, n_attrib).
    func_name: string
        Name of benchmark function.
    n_runs: int
        Number of runs for same parameter.
    dataset_name: string

    Returns
    -------
    df_results: Dataframe
        Dataframe containing the results for the group of parameters.
    """
    n_attrib = list_cluster_params[dataset_name]['n_attrib']
    particle_sizes = [n_attrib*c 
            for c in list_cluster_params[dataset_name]['n_clusters']]

    list_pso_params[dataset_name]['particle_size'] = particle_sizes

    eval_func, task = functions.get_cluster_index(func_name, dataset_name)

    df_results = pd.DataFrame(columns=['run', 'fitness', 'omega', 'c1', 'c2',
                              'n_clusters', 'max_iters', 'pop_size'])
    # Create all permutations of params.
    pso_params = create_grid_params(list_pso_params[dataset_name])
    
    n_params = len(pso_params)
    index_params = 1
    print(f'======== Clustering eval index: {func_name} ========')
    for p in pso_params:
        print(f'======== Parameters {index_params} of {n_params} ========')
        for run in range(n_runs):
            
            print(f'-------- PSO - run {run+1} --------')
            _, _, best_evals = pso.run_pso(
                    eval_func=eval_func, consts=p['consts'], 
                    max_iters=p['max_iters'], pop_size=p['pop_size'],
                    particle_size=p['particle_size'], l_bound=-1.0,
                    u_bound=1.0, task=task
            )
            n_iters = len(best_evals)
            n_clusters = int(p['particle_size']/n_attrib)
            df_new_res = pd.DataFrame(
                    {'run':[run+1]*n_iters,
                    'fitness':best_evals,
                    'omega':[p['consts'][0]]*n_iters,
                    'c1':[p['consts'][1]]*n_iters,
                    'c2':[p['consts'][2]]*n_iters,
                    'n_clusters':[n_clusters]*n_iters}
                    )
            df_results = df_results.append(df_new_res, ignore_index=True)
        index_params += 1
        #TODO
    return df_results
        

def run_pso_experiments(list_params, func_name, n_runs):
    """
    Execute experiments with the PSO algorithm 'n_runs' times for each
    group of PSO parameters and a given benchmark function.

    Parameters
    ----------
    list_params: dict
        Dictionary containing lists of different values for parameters.
    func_name: string
        Name of benchmark function.
    n_runs: int
        Number of runs for same parameter.

    Returns
    -------
    df_results: Dataframe
        Dataframe containing the results for the group of parameters.
    """
    pso_params = create_grid_params(list_params)
    eval_func, l_bound, u_bound, task = functions.get_function(func_name)

    df_results = pd.DataFrame(columns=['run', 'fitness', 'omega', 'c1', 'c2'])

    print(f'======== Benchmark function: {func_name} ========')
    
    n_params = len(pso_params)
    index_params = 1
    for p in pso_params:
        print(f'======== Parameters {index_params} of {n_params} ========')
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
                    'c2':[p['consts'][2]]*len(best_evals)}
                    )
            df_results = df_results.append(df_new_res, ignore_index=True)
        index_params += 1
    return df_results


def run_cluster_hgapso_experiments(list_pso_params, list_ga_params,
                                   list_cluster_params, func_name, n_runs,
                                   dataset_name):
    """
    """



def run_hgapso_experiments(list_pso_params, list_ga_params, func_name,
                           n_runs):
    """
    Run experiments with the HGAPSO algorithm 'n_run' times for
    each set of parameters.

    Parameters
    ----------
    list_pso_params: dict
        Dictionary containing all PSO parameters tested.
    list_ga_params: didct
        Dcitionary containing all GA parameters.
    func_name: string
        Name of function.
    n_runs: int
        Number of times the experiment is run.
    """
    all_params = {}
    all_params.update(list_pso_params)
    del list_ga_params['prob_cross']
    del list_ga_params['c']
    del list_ga_params['n_gens']
    all_params.update(list_ga_params)

    pso_ga_params = create_grid_params(all_params)
    eval_func, l_bound, u_bound, task = functions.get_function(func_name)
    func_params = {"eval_func":eval_func, "l_bound":l_bound, 
            "u_bound":u_bound, "task":task}

    df_results = pd.DataFrame(
            columns=['run', 'fitness', 'omega', 'c1', 'c2', 'prob_mut'])

    print(f'======== Benchmark function: {func_name} ========')
    n_params = len(pso_ga_params)
    index_params = 1
    for p in pso_ga_params:
        print(f'======== Parameters {index_params} of {n_params} ========')
        for run in range(n_runs):
            print(f'-------- HGAPSO - run {run+1} --------')
            _, _, best_evals = hgapso.run_hgapso(alg_params=p,
                                                 func_params=func_params)
            df_new_res = pd.DataFrame(
                    {'run':[run+1]*len(best_evals),
                    'fitness':best_evals,
                    'omega':[p['consts'][0]]*len(best_evals),
                    'c1':[p['consts'][1]]*len(best_evals),
                    'c2':[p['consts'][2]]*len(best_evals),
                    'prob_mut':[p['prob_mut']]*len(best_evals)}
                    )
            df_results = df_results.append(df_new_res, ignore_index=True)
        index_params += 1
    return df_results


def run_logapso_experiments(list_pso_params, list_ga_params,
                            list_logapso_params, func_name, n_runs):
    """
    Execute experiments with the LOGAPSO 'n_runs' times for each
    combination of PSO and GA parameters for a given benchmark function.

    Parameters
    ----------
    list_pso_params: dict
        Dictionary containing the PSO parameters.
    list_ga_params: dict
        Dictionary containing the GA parameters.
    list_logapso: dict
        Dictionary containing the LOGAPSO parameters.
    func_name: string
    n_runs: int

    Returns
    -------
    df_results: Dataframe
        Dataframe containing the results of experiments.
    """
    all_params = {}
    all_params.update(list_pso_params)
    all_params.update(list_ga_params)
    all_params.update(list_logapso_params)
    
    pso_ga_params = create_grid_params(all_params)
    eval_func, l_bound, u_bound, task = functions.get_function(func_name)
    func_params = {"eval_func":eval_func, "l_bound":l_bound,
            "u_bound":u_bound, "task":task}

    df_results = pd.DataFrame(
            columns=['run', 'fitness', 'omega', 'c1', 'c2'])
    for p in pso_ga_params:
        for run in range(n_runs):
            print(f'-------- LOGAPSO - run {run+1} --------')
            _, _, best_evals = logapso.run_logapso(
                    alg_params=p, func_params=func_params,
                    prob_run_ga=p['prob_run_ga'],
                    step_size=p['step_size']
                    )
            df_new_res = pd.DataFrame(
                    {'run':[run+1]*len(best_evals),
                    'fitness':best_evals,
                    'omega':[p['consts'][0]]*len(best_evals),
                    'c1':[p['consts'][1]]*len(best_evals),
                    'c2':[p['consts'][2]]*len(best_evals),
                    'prob_run_ga':[p['prob_run_ga']]*len(best_evals),
                    'step_size':[p['step_size']]*len(best_evals)}
                    )
            df_results = df_results.append(df_new_res, ignore_index=True)
    return df_results


def run_experiments(n_runs, params):
    """Run a group of experiments for each optimisation algorithm."""
    # Number of times each group of experiments is run
    algorithms = ['logapso']
    benchmark_funcs = params['function']
    
    for alg in algorithms:
        for func in benchmark_funcs:
            if alg == 'pso':
                df_results = run_pso_experiments(params['pso'], func, n_runs)
            elif alg == 'hgapso':
                df_results = run_hgapso_experiments(
                        params['pso'], params['ga'], func, n_runs)
            elif alg == 'logapso':
                df_results = run_logapso_experiments(
                        params['pso'], params['ga'], params['logapso'], 
                        func, n_runs)

            save_results(alg, func, df_results)


if __name__ == '__main__':
    params = read_json('parameters.json')
    run_experiments(5, params)