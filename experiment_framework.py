import numpy as np
import pandas as pd
import json
import hybrid_pso_ga
import pso
import functions

def read_json (file_name):
    with open(file_name) as json_file:
        params = json.load (json_file)
    return params

def run_hybrid_pso_ag_experiment (n_runs, params, eval_function, index_params):
    i = index_params

    hybrid_pso_ga.partitioned_pso ( 
        n_partitions = params ['hybrid']['n_partitions'][i], 
        n_particles = params ['pso']['pop_sizes'][i], 
        n_vars = params ['pso']['particle_sizes'][i], 
        n_particles_part = params ['hybrid']['n_particles_partition'][i], 
        n_vars_part = params ['hybrid']['n_vars_partition'][i], 
        consts = params ['pso']['constants'][i], 
        eval_func = eval_function,
        max_iters_hybrid = params ['hybrid']['max_iters'][i],
        max_iters_pso = params ['pso']['max_iters'][i], 
        u_bound = params ['pso']['u_bounds'][i], 
        l_bound = params ['pso']['l_bounds'][i],
        task = params ['pso']['task'][i],
        prob_cross = params ['ga']['prob_cross'][i], 
        prob_mut = params ['ga']['prob_mut'][i], 
        c = params ['ga']['c'][i] )
    return 'None'


def run_pso_experiment (n_runs, params, eval_function, index_params):
    i = index_params

    df_results = pd.DataFrame (columns = ['run', 'fitness', 'omega', 'c1', 'c2'])

    for run in range(n_runs):
        _, _, best_evals = pso.run_pso ( 
            eval_func = eval_function, 
            consts = params['pso']['constants'][i],
            max_iter = params['pso']['max_iters'][i], 
            pop_size = params['pso']['pop_sizes'][i], 
            particle_size = params['pso']['particle_sizes'][i], 
            l_bound = params['pso']['l_bounds'][i], 
            u_bound = params['pso']['u_bounds'][i],
            task = params['pso']['task'][i] )

        df_new_res = pd.DataFrame ({'run': [run + 1] * len(best_evals), 
                        'fitness': best_evals, 
                        'omega': [params['pso']['constants'][i][0]] * len(best_evals), 
                        'c1': [params['pso']['constants'][i][1]] * len(best_evals), 
                        'c2': [params['pso']['constants'][i][2]] * len(best_evals) })
    df_results  = df_results.append (df_new_res, ignore_index=True)

    return df_results

    
def run_experiments ():
    params = read_json ('parameters.json')

    '''Lists the experiments to be executed.
    The tuple determines the algorithm to be executed and the index of params.
    The algorithm and index (starting 0) are determined by the parameters.json file.
    Example: ('pso', 0)
        It will run the pso algorithm with the values of first positions in lists.'''
    experiments = [('pso', 0), ('pso', 1)]

    # Number of times each group of experiments is run
    n_runs = 10

    for alg, index_params in experiments:
        
        eval_function = functions.get_function (
                            params['pso']['eval_funcs'][index_params])
        if alg == 'hybrid':
            df_results = run_hybrid_pso_ag_experiment (
                            n_runs, params, eval_function, index_params)
        elif alg == 'pso':
            df_results = run_pso_experiment (
                            n_runs, params, eval_function, index_params)
        elif alg == 'logapso':
            df_results = None
        print (df_results)

if __name__ == '__main__':
    try:
        run_experiments ()

    #Value error exception
    except:
       # File not found exception
       print ('Erro durante a execução dos experimentos')

