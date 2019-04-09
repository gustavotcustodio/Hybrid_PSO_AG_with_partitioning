import numpy as np
import pso
import genetic as ga
import random
import functions

def logapso_ga_func (eval_func, particle):
    def wrapper (chromosome):
        return eval_func (particle + chromosome) 
    return wrapper

def ga_random_walk (eval_func, best_parts, evals_best, task, ind_apply_ga, 
    prob_run_ga, step_size, pso_params, ga_params):
    '''
    Executes the genetic algorithm to move the best position found by 
    particle i in a random direction. If the new position found is a better 
    solution than best_parts[i], replace best_parts[i] by the best solution.
    '''
    op = min if task == 'min' else max

    for i in ind_apply_ga:
        if random.random() < prob_run_ga:
            # Use a decorator to store the particle's position before
            # moving it randomly thorugh the search space.
            ga_func = logapso_ga_func (eval_func, best_parts[i])

            # Apply the GA to get the step direction where the best solution moves.
            step_direction = ga.run_ga (pop_size = pso_params['pop_size'], 
                                        chrom_size = pso_params['particle_size'],
                                        n_gens = ga_params['n_gens'],
                                        fitness_func = ga_func, 
                                        prob_cross = ga_params['prob_cross'],
                                        c = ga_params['c'], 
                                        prob_mut = ga_params['prob_mut'], 
                                        l_bound = pso_params['l_bound'], 
                                        u_bound = pso_params['u_bound'],
                                        task = pso_params['task'] )

            test_new_best = best_parts [i] + step_size * step_direction
            eval_new = eval_func (test_new_best)

            # Check if the fitness of the new solution is better than 
            # the current best (better if the fitness is higher or lower,
            # depending on the type of problem.)
            if op (evals_best [i], eval_new) == eval_new:
                best_parts [i] = test_new_best

def run_logapso (pso_params, ga_params, eval_func, prob_run_ga = 0.3, 
    step_size=0.2, initial_particles = None):
    
    if initial_particles is None:
        particles = pso.generate_particles (
                            pso_params['pop_size'], 
                            pso_params['particle_size'], 
                            pso_params['l_bound'], 
                            pso_params['u_bound'])
    else:
        particles = initial_particles

    pop_size = particles.shape[0]
    particle_size = particles.shape[1]
        
    evals_parts = pso.evaluate_particles (eval_func, particles)
    
    best_parts, evals_best = np.copy (particles), np.copy (evals_parts)
    global_best, eval_global = pso.get_best_particle (
                                    particles, evals_parts, pso_params['task'])

    velocities = pso.generate_velocities (pop_size, particle_size, 
                        pso_params['l_bound'], pso_params['u_bound'])
    global_solutions, best_evals = [], []

    for _ in range (pso_params['max_iters']):
        velocities = pso.update_velocities (particles, best_parts, 
                            global_best, velocities, pso_params['constants'])
        # Limit velocity to bounds
        #pso.limit_bounds (velocities, pso_params['l_bound'], pso_params['u_bound'])

        particles = pso.update_positions (particles, velocities)
        # Limit particles to bounds
        #pso.limit_bounds (particles, pso_params['l_bound'], pso_params['u_bound'])

        evals_parts = pso.evaluate_particles (eval_func, particles)

        best_copy = np.copy (best_parts)
        pso.update_best_solutions ( particles, best_parts, 
                                    evals_parts, evals_best, pso_params['task'])
        # Indices of best solutions to apply the GA
        ind_apply_ga = np.unique (np.where(best_copy != best_parts)[0])

        ga_random_walk (eval_func, best_parts, evals_best, pso_params['task'], 
                        ind_apply_ga, prob_run_ga, step_size, pso_params, ga_params)

        global_best, eval_global = pso.update_global_best (particles, global_best, 
                                        evals_parts, eval_global, pso_params['task'])
        global_solutions.append (global_best)
        best_evals.append (eval_global)
        #print (eval_global)
    return particles, np.array (global_solutions), best_evals

if __name__ == "__main__":
    pso_params = {"pop_size": 100,
                  "particle_size": 30,
                  "max_iters": 10,
                  "consts": [0.7, 1.4, 1.4],
                  "u_bound": 100.0,
                  "l_bound": -100.0,
                  "eval_func": 'rastrigin',
                  "task": 'min'}

    ga_params =  {"prob_cross": 0.8,
                  "prob_mut": 0.1,
                  "c": 0.5,
                  "n_gens":10}

    eval_func = functions.rastrigin

    run_logapso (pso_params, ga_params, eval_func)