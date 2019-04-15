import numpy as np 
import pandas as pd
import pso
import genetic as ga
import functions
import random

def run_hgapso (pso_params, ga_params, eval_func, particle_size, max_iters = 100):
    pop_size = 4 * particle_size
    population = pso.generate_particles (pop_size, particle_size, 
                        pso_params['l_bound'], pso_params['u_bound'])
    # Evaluate the particles
    evals_parts = pso.evaluate_particles (eval_func, population)
    best_fitness =  float('inf') if pso_params['task'
                                    ] == 'min' else -float('inf')
    for iter in range (max_iters):
        ######################### GA part ##############################
        # Select the 2N best particles (N = particle_size)
        top_pop, top_indices = ga.selection (population, evals_parts, 
                                    int(particle_size*2), pso_params['task'])
        # Select the 2N worst particles (bottom particles)
        bot_indices = np.delete (range(pop_size), top_indices)
        bot_pop = population [bot_indices]

        # Crossover operation
        top_pop = ga.random_arith_crossover (top_pop)

        # Mutation
        top_pop = ga.normal_mutation( top_pop,
                                      prob_mut=ga_params['prob_mut'],
                                      l_bound=pso_params['l_bound'],
                                      u_bound=pso_params['u_bound'])
        # Evaluate the top 2N particles
        evals_top = pso.evaluate_particles (eval_func, top_pop)
        population [top_indices] = top_pop

        ######################### PSO part #############################
        if iter == 0:
            evals_parts = pso.evaluate_particles (eval_func, population)
            evals_bot = evals_parts [bot_indices]
            pbest = np.copy (population)
            evals_pbest = np.copy (evals_parts)
            gbest, eval_gbest = pso.get_best_particle (bot_pop, 
                                        evals_bot, pso_params['task'])
            velocities = pso.generate_velocities (pop_size, particle_size, 
                                pso_params['l_bound'], pso_params['u_bound'])

        velocities[bot_indices] = pso.update_velocities (
                                    bot_pop, pbest[bot_indices], gbest, 
                                    velocities[bot_indices], pso_params['consts'])

        bot_pop = pso.update_positions (bot_pop, velocities[bot_indices])
        # Evaluate the worst 2N particles
        evals_bot = pso.evaluate_particles (eval_func, bot_pop)
        
        pbest[bot_indices], evals_pbest[bot_indices] = pso.update_best_solutions (
                bot_pop, pbest[bot_indices], evals_bot, evals_pbest[bot_indices], 
                pso_params['task']
        )
        gbest, eval_gbest = pso.update_global_best ( bot_pop, gbest, 
                evals_bot, eval_gbest, pso_params['task']
        )
        population[bot_indices] = bot_pop
        evals_parts = pso.evaluate_particles (eval_func, population)

        if pso_params['task']=='min':
            min_index = np.argmin (evals_parts)
            if evals_parts [min_index] < best_fitness:
                best_fitness = evals_parts [min_index]
        else:
            max_index = np.argmax (evals_parts) 
            if evals_parts [max_index] > best_fitness:
                best_fitness = evals_parts [max_index]
        print (best_fitness)
    return population, evals_parts

if __name__ == "__main__":
    pso_params = {"u_bound": 100.0,
                  "l_bound":-100.0,
                  "task": 'min',
                  "consts": [0.7, 1.4, 1.4]}
    ga_params = {"prob_mut": 0.02}
    eval_func = functions.rastrigin

    run_hgapso (pso_params, ga_params, eval_func, 
                    particle_size=30, max_iters=1000)