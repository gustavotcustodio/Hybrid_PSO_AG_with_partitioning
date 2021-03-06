import numpy as np
import genetic as ga
import pso
import functions

def split_particles (particles, n_particles_part, n_vars_part):
    """
    Split all PSO particles in partitions.
    Example of two particles: [[1,2,6,3,1,1], [9,9,1,2,4,7]]
    splitted in 2 subpops and 3 subspaces:

    1 2 | 6 3 | 1 1
    ---------------
    9 9 | 1 2 | 4 7

    Attributes
    ----------
    particles: 2d array
        All PSO particles.
    n_particles_part: int
        Number of particles in each partition.
    n_vars_part: int
        Number of variables in each partition.

    Returns
    -------
    partitions: 3d array [partition, particle, variable]
        Particles splitted in different partitions.
    """
    n_subpops = int (particles.shape[0] / n_particles_part)
    n_subspaces = int (particles.shape[1] / n_vars_part)

    subpops = np.array (np.split (particles, n_subpops))
    subspaces = np.array (np.split (subpops, n_subspaces, axis=2))
    return np.concatenate (subspaces)
 
def merge_particles (partitions, n_subspaces):
    """
    Merge particles splitted in split_particles method.

    Parameters
    ----------
    partitions: 3d array
        PSO particles splitted in partitions.
    n_subspaces: int
        Number of groups that each particle is splitted.

    Returns
    -------
    particles: 2d array
        Partitions of particles splitted in a 3d array
        merged again in a 2d array.
    """
    # 2d array containing variables of particles
    particles_vars = np.concatenate(partitions)

    # 3d array with slices of particles splitted in n_subspaces
    particles_subspaces = np.split(particles_vars, n_subspaces)
    return np.concatenate(particles_subspaces, axis=1)

def split_and_crossover (population, n_particles_part, n_vars_part, 
    prob_cross, c):
    """
    """
    cross_pop = None
    n_particles = population.shape[0]
    n_vars = population.shape[1]

    for chrom in range (0, n_particles, n_particles_part):
        for var in range (0, n_vars, n_vars_part):
            # index of last solution in subpop
            last_chrom = chrom + n_particles_part
            # index of last variable in subspace
            last_var = var + n_vars_part

            subpop = population [chrom:last_chrom, var:last_var]
            # Get the offsprings and their parents
            offsprings, parents = ga.arith_crossover(subpop, prob_cross, c)

            if offsprings is not None:
                # Since the offsprings are related to a subspace, they 
                # have less variables than a candidate solution. Therefore, 
                # the parents are copied and the variables related to the 
                # offsprings are updated.
                new_chroms = np.copy (population [chrom:last_chrom][parents])
                new_chroms [:,var:last_var] = offsprings
                if cross_pop is None:
                    cross_pop = new_chroms
                else:
                    cross_pop = np.append (cross_pop, new_chroms, axis=0)
    return cross_pop

def run_hpsoga (pso_params, ga_params, eval_func, n_particles_part, 
    n_vars_part, max_iters = 100):
    """
    """                
    population = None
    global_best = None
    eval_global = float('inf') if pso_params['task'] == 'min' else -float('inf')
    for _ in range (max_iters):         
        # Apply the standard PSO to all particles 
        population,_,_= pso.run_pso (
                            eval_func = eval_func,
                            consts = pso_params['consts'],
                            max_iters = pso_params['max_iters'],
                            pop_size = pso_params['pop_size'], 
                            particle_size = pso_params['particle_size'], 
                            initial_particles = population,
                            u_bound = pso_params['u_bound'],
                            l_bound = pso_params['l_bound'],
                            task = pso_params['task'] )
        # Evaluate all candidate solutions
        fitness_vals = np.apply_along_axis (eval_func, 1, population) 
        # Apply the selection operator in all solutions
        selected_pop = ga.roulette_selection (population, pso_params['pop_size'], 
                            fitness_vals, pso_params['task'])
        # Split population in sub-partitions and apply the crossover in each one.
        new_particles = split_and_crossover (selected_pop, n_particles_part, 
                            n_vars_part, ga_params['prob_cross'], ga_params['c'])
        if new_particles is not None:
            # Add new candidate solutions to population
            selected_pop = np.append (selected_pop, new_particles, axis=0)
        # Apply mutation
        population = ga.arith_mutation(selected_pop, ga_params['prob_mut'], 
                            pso_params['l_bound'], pso_params['u_bound'])

        fitness_vals = np.apply_along_axis (eval_func, 1, population) 
        global_best, eval_global = pso.update_global_best (population, global_best, 
                                    fitness_vals, eval_global, pso_params['task'])
        print (eval_global)
    return population

if __name__ == "__main__":
    pso_params = {"pop_size": 100,
                  "particle_size": 30,
                  "max_iters": 100,
                  "consts": [0.7, 1.4, 1.4],
                  "u_bound": 100.0,
                  "l_bound": -100.0,
                  "task": 'min'}

    ga_params =  {"prob_cross": 0.6,
                  "prob_mut": 0.01,
                  "c": 0.5}

    eval_func = functions.rastrigin

    n_particles_part = 20
    n_vars_part = 10

    run_hpsoga (pso_params, ga_params, eval_func, n_particles_part, 
                n_vars_part, max_iters = 100)
