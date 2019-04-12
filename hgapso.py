import numpy as np 
import pandas as pd
import pso
import genetic as ga
import functions

def run_hgapso (pso_params, eval_func, particle_size, max_iters = 100):
    pop_size= 4* particle_size
    population = pso.generate_particles (pop_size, particle_size, 
                        pso_params['l_bound'], pso_params['u_bound'])
    # Evaluate the particles
    evals_parts = pso.evaluate_particles (eval_func, population)
    consts = [0.5, 2.0, 2.0]

    for iter in range (max_iters):
        ######################### GA part ##############################
        # Select the 2N best particles (N = particle_size)
        top_pop, top_indices = ga.selection (population, evals_parts, 
                                    particle_size * 2)
        # Select the 2N worst particles
        bottom_pop = np.delete (population, top_indices[:], axis=0)

        cross_pop = ga.random_arith_crossover (top_pop)
        top_pop_mut = ga.normal_mutation(cross_pop, 
                                         prob_mut=0.2, 
                                         l_bound=pso_params['l_bound'], 
                                         u_bound=pso_params['u_bound'])
        # Evaluate the top 2N particles
        evals_top = pso.evaluate_particles (eval_func, top_pop_mut)

        ######################### PSO part #############################
        if iter == 0:
            pbest = np.copy (bottom_pop)
            # Evaluate the worst 2N particles
            evals_bottom = pso.evaluate_particles (eval_func, bottom_pop)
            evals_pbest = np.copy (evals_bottom)

            global_best, eval_global = pso.get_best_particle (bottom_pop,
                                           evals_bottom,  pso_params['task'])

            velocities = pso.generate_velocities (int(pop_size/2), particle_size, 
                             pso_params['l_bound'], pso_params['u_bound'])

        velocities = pso.update_velocities (bottom_pop, 
                            pbest, global_best, velocities, consts)

        bottom_pop = pso.update_positions (bottom_pop, velocities)
        # Evaluate the worst 2N particles
        evals_bottom = pso.evaluate_particles (eval_func, bottom_pop)

        pso.update_best_solutions (bottom_pop, 
            pbest, evals_bottom, evals_pbest, pso_params['task']
        )
        global_best, eval_global = pso.update_global_best ( bottom_pop, 
            global_best, evals_bottom, eval_global, pso_params['task']
        )
        population  = np.append (top_pop, bottom_pop, axis= 0)
        evals_parts = np.append (evals_top, evals_bottom)
        print (np.min (evals_parts))
    # if it is a minimization problem, get the minimum value.
    if pso_params['task'] == 'min':
        arg_gbest = np.argmin (evals_parts)
    else:
        arg_gbest = np.argmax (evals_parts)
    return population[arg_gbest], evals_parts[arg_gbest]

if __name__ == "__main__":
    pso_params = {"u_bound": 100.0,
                  "l_bound": -100.0,
                  "task": 'min'}

    eval_func = functions.rastrigin

    run_hgapso (pso_params, eval_func, particle_size=30, max_iters=100)