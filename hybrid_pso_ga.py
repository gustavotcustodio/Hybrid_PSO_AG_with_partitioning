import numpy as np
import pso
import genetic as ga

def split_particles (particles, n_particles_part, n_vars_part):
    '''
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
    '''
    n_subpops = int (particles.shape[0] / n_particles_part)
    n_subspaces = int (particles.shape[1] / n_vars_part)

    subpops = np.array (np.split (particles, n_subpops))
    subspaces = np.array (np.split (subpops, n_subspaces, axis=2))

    return np.concatenate (subspaces)

 
def merge_particles (partitions, n_subspaces):
    '''
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
    '''
    # 2d array containing variables of particles
    particles_vars = np.concatenate ( partitions )

    # 3d array with slices of particles splitted in n_subspaces
    particles_subspaces = np.split (particles_vars, n_subspaces)
    return np.concatenate (particles_subspaces, axis=1)

def split_and_crossover (population, n_particles_part, n_vars_part, 
                            n_particles, n_vars, prob_cross, c):
    '''
    '''
    cross_pop = None
    for chrom in range (0, n_particles, n_particles_part):
        for var in range (0, n_vars, n_vars_part):
            # index of last solution in subpop
            last_chrom = chrom + n_particles_part
            # index of last variable in subspace
            last_var = var + n_vars_part

            subpop = population [chrom:last_chrom, var:last_var]
            # Get the offsprings and their parents
            offsprings, parents = ga.crossover (subpop, prob_cross, c)
            
            if offsprings is not None:
                ''' Since the offsprings are related to a subspace, they 
                have less variables than a candidate solution. Therefore, 
                the parents are copied and the variables related to the 
                offsprings are updated.'''
                new_chroms = np.copy (population [chrom:last_chrom][parents])
                new_chroms [:,var:last_var] = offsprings
                if cross_pop is None:
                    cross_pop = new_chroms
                else:
                    cross_pop = np.append (cross_pop, new_chroms)
    return cross_pop

def partitioned_pso (n_partitions, n_particles, n_vars, n_particles_part, n_vars_part, 
    consts, eval_func, max_iters_hybrid=100, max_iters_pso=100, u_bound=1.0, 
    l_bound=-1.0, task ='min', prob_cross = 0.6, prob_mut = 0.01, c = 0.5):
    '''
    '''                
    population = None
    n_subpops = int (n_particles / n_particles_part)
    n_subspaces = int (n_vars / n_vars_part)

    for _ in range(max_iters_hybrid):         
        # Apply the standard PSO to all particles 
        population,_,_ = pso.run_pso (eval_func, consts, max_iter= max_iters_pso, 
                                    pop_size= n_particles, particle_size= n_vars, 
                                    initial_particles= population)
        # Evaluate all candidate solutions
        fitness_solutions = np.apply_along_axis (eval_func, 0, population)

        # Apply the selection operator in all solutions
        changed_pop = ga.roulette_selection (
                                population, n_particles, fitness_solutions)

        # Split population in sub-partitions and apply the crossover in each one.
        new_particles = split_and_crossover (changed_pop, n_particles_part, 
                            n_vars_part, n_particles, n_vars, prob_cross, c)

        if new_particles is not None:
            # Add new candidate solutions to population
            changed_pop = np.append (changed_pop, new_particles, axis = 0)

        # Apply mutation
        population = ga.mutation (changed_pop, prob_mut, u_bound, l_bound)
    return population

if __name__ == '__main__':
    consts = [0.7, 1.4, 1.4]
    eval_func = lambda p: np.sum (p**2)

    partitioned_pso (n_partitions=9, n_particles=6, n_vars=9, 
                        n_particles_part=2, n_vars_part=3, 
                        consts=consts, eval_func=eval_func)