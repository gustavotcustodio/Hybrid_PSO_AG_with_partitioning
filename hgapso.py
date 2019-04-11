import numpy as np 
import pandas as pd
import pso
import genetic as ga

def run_hgapso (pso_params, ga_params, eval_func, particle_size, max_iters = 100):
    pop_size = 4 * particle_size

    population = pso.generate_particles ( pop_size, particle_size, 
                        pso_params['l_bound'], pso_params['u_bound'])

    evals_parts = pso.evaluate_particles (eval_func, population)

    cross_pop = ga.random_arith_crossover (population)

    
    
    return population, evals_parts