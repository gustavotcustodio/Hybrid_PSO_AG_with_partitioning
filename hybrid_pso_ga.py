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


def partitioned_pso (n_partitions, n_particles, n_vars, n_particles_part, 
    n_vars_part, consts, eval_func, iters_hybrid=100, u_bound=1.0, l_bound=-1.0,
    prob_cross = 0.5, c = 0.5):
    '''
    '''                
    population = None

    for _ in range(iters_hybrid):
                
        # Apply the standard PSO to all particles 
        population, _, _ = pso.run_pso (
                            eval_func, consts, max_iter=100, pop_size=8, 
                            particle_size=9, initial_particles = population)

        n_subspaces = int (population.shape[1] / n_vars_part)

        # Number of particles to select
        num_to_select = int (n_partitions * n_particles_part / n_subspaces)

        # Apply the selection operator in all particles
        selected_particles = ga.selection (population, num_to_select) # mudar para índices***

        splitted_pop = split_particles (selected_particles, 
                                        n_particles_part, n_vars_part)

        # Apply crossover in all sub-partitions
        for i in range(n_partitions):
            splitted_pop[i] = ga.crossover (splitted_pop[i], prob_cross, c)

        # Apply mutation
        # Merge sub-partitions
        merged_pop = merge_particles (splitted_pop, n_subspaces)

        print (population[:,-1])
        print (merged_pop[:,-1])
        
        print ('==================================================================')

    return population


if __name__ == '__main__':
    consts = [0.7, 1.4, 1.4]
    eval_func = lambda p: np.sum (p**2)

    partitioned_pso (n_partitions=9, n_particles=6, n_vars=9, 
                        n_particles_part=2, n_vars_part=3, 
                        consts=consts, eval_func=eval_func)