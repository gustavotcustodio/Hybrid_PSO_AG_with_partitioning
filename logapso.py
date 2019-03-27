
import numpy as np
import pso
import genetic as ga
import random 

def ga_random_walk (eval_func, best_parts, evals_best, task, 
    ind_apply_ga, prob_run_ga, step_size, genetic_alg):
    '''
    Executes the genetic algorithm to move the best position found by 
    particle i in a random direction. If the new position found is a better 
    solution than best_parts[i], replace best_parts[i] by the best solution.
    '''
    op = min if task == 'min' else max

    for i in ind_apply_ga:
        if random.random() < prob_run_ga:
            step_direction = genetic_alg (best_parts[i])

            test_new_best = best_parts [i] + step_size * step_direction
            eval_new = eval_func (test_new_best)

            # Check if the fitness of the new solution is better than 
            # the current best (better if the fitness is higher or lower,
            # depending on the type of problem.)
            if op (evals_best [i], eval_new) == eval_new:
                best_parts [i] = test_new_best

def run_logapso (eval_func, consts, max_iter=100, pop_size=100, particle_size=10, 
    initial_particles = None, l_bound=-1.0, u_bound=1.0, task='min'):
    if initial_particles is None:
        particles = pso.generate_particles (
                            pop_size, particle_size, l_bound, u_bound)
    else:
        particle_size = initial_particles.shape[1]
        pop_size = initial_particles.shape[0]
        particles = initial_particles

    evals_parts = pso.evaluate_particles (eval_func, particles)
    
    best_parts = np.copy (particles)
    evals_best = np.copy (evals_parts)
 
    global_best, eval_global = pso.get_best_particle (particles, evals_parts, task)

    velocities = pso.generate_velocities (pop_size, particle_size, l_bound, u_bound)

    global_solutions, best_evals = [], []

    for _ in range (max_iter):
        velocities = pso.update_velocities (particles, best_parts, 
                            global_best, velocities, consts)
        # Limit velocity to bounds
        pso.limit_bounds (velocities, l_bound, u_bound)

        particles = pso.update_positions  (particles, velocities)
        # Limit particles to bounds
        pso.limit_bounds (particles, l_bound, u_bound)

        evals_parts = pso.evaluate_particles (eval_func, particles)

        best_copy = np.copy (best_parts)
        pso.update_best_solutions (
                particles, best_parts, evals_parts, evals_best, task)

        # Indices of best solutions to apply the GA
        ind_apply_ga = np.unique (np.where(best_copy != best_parts)[0])
        best_parts [ind_apply_ga] =  ga_random_walk

        global_best, eval_global = pso.update_global_best (particles, global_best, 
                                            evals_parts, eval_global, task)
        global_solutions.append (global_best)
        best_evals.append (eval_global)
    return particles, np.array (global_solutions), best_evals

