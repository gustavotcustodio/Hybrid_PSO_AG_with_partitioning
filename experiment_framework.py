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

    Parameters
    ----------
    file_name: string

    Return
    ------
    params: dict
        Dictionary with parameters.
    """
    with open(file_name) as json_file:
        params = json.load(json_file)
    return params


def save_results(algorithm, benchmark_func, df_results):
    """
    Save the results of experiments to a csv file.
    
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
        Dictionary containing parameters for experiments.

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


def merge_and_clean_params(lists_of_params_dicts, algorithm):
    """
    Merge the parameters used in an experiment.

    Parameters
    ----------
    list_of_params_dicts: list[dict]
        List containing all dicts of params for an algorithm.
    algorithm: string
        Algorithm for the parameteres.

    Returns
    -------
    all_params: dict
        Dictionary containing all parameters for the current experiment.
    """
    all_params = {}
    for dict_params in lists_of_params_dicts:
        all_params.update(dict_params.copy())
    
    if algorithm == 'hgapso':
        del all_params['prob_cross']
        del all_params['c']
        del all_params['n_gens']
        del all_params['pop_ga']
    return all_params


def run_experiment(algorithm, parameters, func_name, n_runs,
                   df_results, dataset_name=None, n_attrib=1):
    """
    Parameters
    ----------
    algorithm: string
        Name of algorithm.
    parameters: dict
        Parameters for the algorithm in the current experiment.
    func_name: string
    n_runs: int
    df_results: DataFrame
        Dataframe for saving the experiments results.
    dataset_name: string
        Takes the value 'None' if it's not a clustering optimisation problem.
    n_attrib: int
        Number of attributes for the dataset (Default value = 1).

    Returns
    -------
    df_results: DataFrame
        DataFrame containing the results of experiments.
    """
    if dataset_name is not None:
        eval_func, task = functions.get_cluster_index(func_name, dataset_name)
        l_bound, u_bound = -1.0, 1.0
        print(f'======== Clustering eval index: {func_name} ========')
    else:
        eval_func, l_bound, u_bound, task = functions.get_function(func_name)
        print(f'======== Benchmark function: {func_name} ========')

    func_params = {"eval_func":eval_func, "l_bound":l_bound,
                   "u_bound":u_bound, "task":task}
    # create all permutations of parameters
    grid_params = create_grid_params(parameters)
    
    n_params = len(grid_params)
    index_params = 1

    for p in grid_params:
        print(f'======== Parameters {index_params} of {n_params} ========')
        for run in range(n_runs):
            print(f'-------- {algorithm} - run {run+1} --------')
            if algorithm == 'pso':
                _, _, best_evals = pso.run_pso(
                        eval_func=eval_func, consts=p['consts'],
                        max_iters=p['max_iters'], pop_size=p['pop_size'],
                        particle_size=p['particle_size'], l_bound=l_bound,
                        u_bound=u_bound, task=task)
            elif algorithm == 'hgapso':
                _, _, best_evals = hgapso.run_hgapso(alg_params=p,
                                                     func_params=func_params)
            else: # logapso
                _, _, best_evals = logapso.run_logapso(
                                    alg_params=p, func_params=func_params, 
                                    prob_run_ga=p['prob_run_ga'],
                                    step_size=p['step_size'])
            n_iters = len(best_evals)
            # This is a clustering optimization problem
            if dataset_name is not None: 
                # Each particle has C X F dimensions, where C is the number
                # of clusters and F is the number of features for the dataset.
                n_clusters = int(p['particle_size']/n_attrib)
                df_results = add_results_to_df(
                        p, df_results, n_iters, best_evals, run, algorithm,
                        n_clusters)

            else: # This is problem of benchmark function optimization
                df_results = add_results_to_df(p, df_results, n_iters, 
                                               best_evals, run, algorithm)
        index_params += 1
    return df_results


def add_results_to_df(params, df_results, n_iters, best_evals, run,
                      algorithm, n_clusters=None):
    """
    Add the results of current experiments to the final Dataframe.

    Parameters
    ----------
    params: dict
        Dictionary containing the parameters of experiments.
    df_results: Dataframe
        Dataframe containing the results of experiments.
    n_iters: int
        Number of iterations.
    best_evals: list[float]
        Fitness of best candidate solution for each iteration.
    run: int
        Current execution of the algorithm.
    algorithm: string
        Name of algorithm.
    n_clusters: int or None
        Number of clusters for clustering optimization problems
        (None if it is not a clustering optimization problem).

    Returns
    -------
    df_results: Dataframe
        Dataframe containg the results of experiments so far.
    """
    info_to_input = {'c1':[params['consts'][1]]*n_iters,
                    'c2':[params['consts'][2]]*n_iters,
                    'fitness':best_evals,
                    'max_iters':[params['max_iters']]*n_iters,
                    'omega':[params['consts'][0]]*n_iters,
                    'pop_size':[params['pop_size']]*n_iters,
                    'run':[run+1]*n_iters}
    if algorithm == 'logapso':
        info_to_input.update({'prob_mut':[params['prob_mut']]*n_iters,
                              'prob_run_ga':[params['prob_run_ga']]*n_iters,
                              'step_size':[params['step_size']]*n_iters})
    elif algorithm == 'hgapso':
        info_to_input.update({'prob_mut':[params['prob_mut']]*n_iters})

    if n_clusters is not None:
        info_to_input.update({'n_clusters':[n_clusters]*n_iters})
    else:
        info_to_input.update(
                {'particle_size':[params['particle_size']]*n_iters})
    return df_results.append(pd.DataFrame(info_to_input), ignore_index=True)


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
    df_results = pd.DataFrame(columns=['c1', 'c2', 'fitness', 'max_iters',
                              'omega', 'pop_size', 'run', 'particle_size'])
    return run_experiment('pso', list_params, func_name, n_runs, df_results)


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
    n_attrib = list_cluster_params['n_attrib']
    particle_sizes = [n_attrib*c  for c in list_cluster_params['n_clusters']]

    list_pso_params['particle_size'] = particle_sizes

    df_results = pd.DataFrame(columns=['c1', 'c2', 'fitness','max_iters',
                              'omega', 'pop_size', 'run', 'n_clusters'])
       
    return run_experiment('pso', list_pso_params, func_name, n_runs, 
                          df_results, dataset_name, n_attrib)


def run_hgapso_experiments(list_pso_params, list_ga_params, func_name,
                           n_runs):
    """
    Run experiments with the HGAPSO algorithm 'n_run' times for
    each set of parameters.

    Parameters
    ----------
    list_pso_params: dict
        Dictionary containing all PSO parameters tested.
    list_ga_params: dict
        Dictionary containing all GA parameters.
    func_name: string
        Name of function.
    n_runs: int
        Number of times the experiment is executed.

    Returns
    -------
    df_results: Dataframe
        Dataframe containing the results for the group of parameters.
    """
    all_params = merge_and_clean_params(
            [list_pso_params, list_ga_params], 'hgapso')

    df_results = pd.DataFrame(
            columns=['c1', 'c2', 'fitness', 'max_iters', 'omega',
                     'pop_size', 'run', 'prob_mut', 'particle_size'])

    return run_experiment('hgapso', all_params, func_name, n_runs, df_results)


def run_cluster_hgapso_experiments(list_pso_params, list_ga_params,
                                   list_cluster_params, func_name, n_runs,
                                   dataset_name):
    """
    Run experiments with the HGAPSO algorithm for clustering optimization
    problems. Each experiment for each set of params executes 'n_run' times.

    Parameters
    ----------
    list_pso_params: dict
        Dictionary containing all PSO parameters tested.
    list_ga_params: dict
        Dictionary containing all GA parameters.
    list_cluster_params: dict
        Dictionary containing clustering params.
    func_name: string
        Name of function.
    n_runs: int
        Number of times the experiment is executed.
    dataset_name: string

    Returns
    -------
    df_results: Dataframe
        Dataframe containing the results for the group of parameters.
    """
    # Merge parameters of pso and ga in a single dict
    all_params = merge_and_clean_params(
            [list_pso_params, list_ga_params], 'hgapso')
            
    # Number of the datasets' attributes
    n_attrib = list_cluster_params['n_attrib']
    particle_sizes = [n_attrib*c for c in list_cluster_params['n_clusters']]

    all_params['particle_size'] = particle_sizes

    df_results = pd.DataFrame(
            columns=['c1', 'c2', 'fitness','max_iters', 'omega', 'pop_size',
                     'run', 'prob_mut', 'n_clusters'])
    return run_experiment('hgapso', all_params, func_name, n_runs,
                          df_results, dataset_name, n_attrib)


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
        Dataframe containing the results for the group of parameters.
    """
    all_params = merge_and_clean_params(
            [list_pso_params, list_ga_params, list_logapso_params], 'logapso')
    
    df_results = pd.DataFrame(
            columns=['c1', 'c2', 'fitness', 'max_iters', 'omega', 'pop_size', 
            'run', 'prob_mut', 'prob_run_ga', 'step_size', 'particle_size'])

    return run_experiment('logapso', all_params, func_name, n_runs,
                          df_results)


def run_cluster_logapso_experiments(list_pso_params, list_ga_params,
                                    list_logapso_params, list_cluster_params,
                                    func_name, n_runs, dataset_name):
    """
    Execute experiments with LOGAPSO algorithm for clustering optimization
    problems. Each experiment for each set of params executes 'n_run' times.

    Parameters
    ----------
    list_pso_params: dict
        Dictionary containing the PSO parameters.
    list_ga_params: dict
        Dictionary containing the GA parameters.
    list_logapso: dict
        Dictionary containing the LOGAPSO parameters.
    list_cluster_params: dict
        Dictionary containing clustering params.
    func_name: string
    n_runs: int
    dataset_name: string

    Returns
    -------
    df_results: Dataframe
        Dataframe containing the results for the group of parameters.
    """
    return pd.DataFrame()


def run_all_experiments(n_runs, params):
    """
    Run a group of experiments for each optimisation algorithm.

    Parameters
    ----------
    n_runs: int
        Number of executions of the same algorithm with the same set of parameters.
    params: dict
        Dictionary containing the parameters for the experiments.
    """
    algorithms = ['pso', 'hgapso', 'logapso']
    benchmark_funcs = params['function']

    # Indices for clustering evalutation
    indices_clust_eval = ['davies_bouldin', 'xie_beni']
    datasets = params['clustering'].keys()
    
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

        for cl in indices_clust_eval:
            for dataset in datasets:
                if alg == 'pso':
                    df_results = run_cluster_pso_experiments(
                            params['pso'], params['cluster'][dataset],
                            cl, n_runs, dataset)
                elif alg == 'hgapso':
                    df_results = run_cluster_hgapso_experiments(
                            params['pso'], params['ga'],
                            params['cluster'][dataset], cl, n_runs, dataset)
                elif alg == 'logapso':
                    df_results = run_cluster_logapso_experiments(
                            params['pso'], params['ga'], params['logapso'],
                            params['cluster'][dataset], cl, n_runs, dataset)
                save_results(alg, cl, df_results)


if __name__ == '__main__':
    params = read_json('parameters.json')
    run_all_experiments(5, params)