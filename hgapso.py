import numpy as np 
import pandas as pd
import pso
import genetic as ga
import functions
import random

def run_hgapso(alg_params, func_params):
    """Run the Hgapso algorithm presentend by
    ***Reference***

    Parameters
    ----------
    alg_params: dictionary
        Dictionary with the parameters for the PSO, GA and LOGAPSO
        showed in file parameters.json.
        alg_params = {pop_size:int, particle_size:int, max_iters:int,
            consts:list[float](size=3), }
    func_params: dictionary
        Dictionary containing the benchmark function, lower bound,
        upper bound and the type of problem (minimization or maximization).
        func_params = {eval_func:1d_function, l_bound:float, u_bound:float,
            task:'min' or 'max'}
    """
    particle_size = alg_params['particle_size']
    pop_size = alg_params['pop_size']
    eval_func = func_params['eval_func']

    population = pso.generate_particles(pop_size, particle_size,
            func_params['l_bound'], func_params['u_bound'])
    # Evaluate the particles
    evals_parts = pso.evaluate_particles(eval_func, population)
    best_fitness = (float('inf') if func_params['task']=='min'
                                 else -float('inf'))

    best_pop, best_fitness_values = [], []   
    for iter in range(alg_params['max_iters']):
        # ----------------------- GA part of algorithm -----------------------
        # Select the 2N best particles (N = particle_size)
        top_pop, top_indices = ga.selection(population, evals_parts,
                                            int(pop_size/2),
                                            func_params['task'])
        # Select the 2N worst particles (bottom particles)
        bot_indices = np.delete(range(pop_size), top_indices)
        bot_pop = population[bot_indices]

        # Crossover operation
        top_pop = ga.random_arith_crossover(top_pop)

        # Mutation
        top_pop = ga.normal_mutation(population=top_pop,
                                     prob_mut=alg_params['prob_mut'],
                                     l_bound=func_params['l_bound'],
                                     u_bound=func_params['u_bound'])
        population[top_indices] = top_pop
        # ---------------------- PSO part of algorithm -----------------------
        if iter == 0:
            evals_parts = pso.evaluate_particles(eval_func, population)
            evals_bot = evals_parts[bot_indices]
            pbest = np.copy(population)
            evals_pbest = np.copy(evals_parts)
            gbest, eval_gbest = pso.get_best_particle(bot_pop, evals_bot,
                    func_params['task']
                    )
            velocities = pso.generate_velocities(pop_size, particle_size,
                    func_params['l_bound'], func_params['u_bound']
                    )
        velocities[bot_indices] = pso.update_velocities(
                bot_pop, pbest[bot_indices], gbest, velocities[bot_indices],
                alg_params['consts']
                )
        bot_pop = pso.update_positions(bot_pop, velocities[bot_indices])
        # Evaluate the worst 2N particles
        evals_bot = pso.evaluate_particles(eval_func, bot_pop)
        
        pbest[bot_indices], evals_pbest[bot_indices] = pso.update_best_solutions(
                bot_pop, pbest[bot_indices], evals_bot, 
                evals_pbest[bot_indices], func_params['task']
                )
        gbest, eval_gbest = pso.update_global_best(bot_pop, gbest, evals_bot,
                                                   eval_gbest,
                                                   func_params['task'])
        population[bot_indices] = bot_pop
        evals_parts = pso.evaluate_particles(eval_func, population)

        if func_params['task']=='min':
            best_index = np.argmin(evals_parts)
            if evals_parts[best_index] < best_fitness:
                best_fitness = evals_parts[best_index]
        else:
            best_index = np.argmax(evals_parts) 
            if evals_parts[best_index] > best_fitness:
                best_fitness = evals_parts[best_index]
        best_pop.append(population[best_index])
        best_fitness_values.append(best_fitness)
    return population, best_pop, best_fitness_values


if __name__ == "__main__":
    alg_params = {"particle_size": 30,
                  "pop_size":120,
                  "max_iters": 300, 
                  "consts": [0.7, 1.4, 1.4],
                  "prob_mut": 0.02}
    func_params = {"eval_func": functions.rastrigin,
                  "u_bound": 100.0,
                  "l_bound":-100.0,
                  "task": 'min'}

    run_hgapso (alg_params, func_params)