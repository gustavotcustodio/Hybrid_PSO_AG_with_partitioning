
import numpy as np
import pso

def run_logapso ():
    if initial_particles is None:
        particles = pso.generate_particles (
                        pop_size, particle_size, l_bound, u_bound)
    else:
        particle_size = initial_particles.shape[1]
        pop_size = initial_particles.shape[0]
        particles = initial_particles

    evals_parts = evaluate_particles (eval_func, particles)
    
    best_parts = np.copy (particles)
    evals_best = np.copy (evals_parts)
 
    global_best, eval_global = pso.get_best_particle (particles, evals_parts, task)

    velocities = generate_velocities (pop_size, particle_size, l_bound, u_bound)

    global_solutions, best_evals = [], []

    for _ in range (max_iter):
        velocities = update_velocities (
                        particles, best_parts, global_best, velocities, consts)
        velocities = pso.limit_bounds (velocities, l_bound, u_bound)

        particles  = update_positions  (particles, velocities)
        particles = pso.limit_bounds (velocities, l_bound, u_bound)

        evals_parts = pso.evaluate_particles (eval_func, particles)

        pso.copy_particleupdate_best_solutions (
                    particles, best_parts, evals_parts, evals_best, task)

        global_best, eval_global = pso.update_global_best (
                    particles, global_best, evals_parts, eval_global, task)
        
        global_solutions.append (global_best)
        best_evals.append (eval_global)
    return particles, np.array (global_solutions), best_evals

