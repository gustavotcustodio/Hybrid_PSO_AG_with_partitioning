import numpy as np 
import pandas as pd
import pso
import genetic as ga

def run_hgapso (pso_params, eval_func, particle_size, max_iters = 100):
    pop_size= 4* particle_size
    population = pso.generate_particles (pop_size, particle_size, 
                        pso_params['l_bound'], pso_params['u_bound'])

    evals_parts = pso.evaluate_particles (eval_func, population)

    best_parts = np.copy (population)
    evals_best = np.copy (evals_parts)

    global_best, eval_global = pso.get_best_particle (
                                population, evals_parts,  pso_params['task'])

    velocities = pso.generate_velocities (pop_size, particle_size, 
                                pso_params['l_bound'], pso_params['u_bound'])
    consts = [0.5, 2.0, 2.0]

    # Evaluate the particles
    evals_parts = pso.evaluate_particles (eval_func, population)

    ######################### GA part ##############################

    # Select the 2N best particles (N = particle_size)
    top_pop, top_indices = ga.selection (population, evals_parts, particle_size)
    # Select the 2N worst particles
    bottom_pop = np.delete (population, top_indices)

    cross_pop = ga.random_arith_crossover (top_pop)

    mut_pop = ga.normal_mutation (cross_pop, prob_mut=0.2, 
                l_bound=pso_params['l_bound'], u_bound=pso_params['u_bound'])

    ######################### PSO part #############################
    velocities = pso.update_velocities (bottom_pop, best_parts, 
                                        global_best, velocities, consts)

    bottom_pop = pso.update_positions (bottom_pop, velocities)    
    evals_parts = pso.evaluate_particles (eval_func, bottom_pop)

    pso.update_best_solutions (
        bottom_pop, best_parts, evals_parts, evals_best, pso_params['task']
    )

    global_best, eval_global = pso.update_global_best (
        bottom_pop, global_best, evals_parts, eval_global, pso_params['task']
    )
    return population, evals_parts

if __name__ == '__main__':
    print('aaa')