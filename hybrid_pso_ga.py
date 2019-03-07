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
    n_vars_part, consts, eval_func, iters_hybrid=100, u_bound=1.0, l_bound=-1.0):
    '''
    '''                
    particles = None

    for _ in range(iters_hybrid):
                
        # Apply the standard PSO to all particles 
        particles, _, _ = pso.run_pso (
                            eval_func, consts, max_iter=100, pop_size=100, 
                            particle_size=10, initial_particles = particles)

        print (particles)

        n_subspaces = int (particles.shape[1] / n_vars_part)

        n_subpops =   (n_partitions * n_particles_part) / n_subspaces

        # Number of particles to select
        num_to_select = int (n_particles_part * n_particles_part / n_subpops)

        # Apply the selection operator in all particles
        selected_parts = ga.selection (particles, num_to_select)

        # Partition the population in sub-partitions
        splitted_parts = split_particles (
                            selected_parts, n_particles_part, n_vars_part)

        # Apply crossover in all sub-partitions

        # Apply mutation

        # Merge sub-partitions
        merged_particles = merge_particles (
                            split_particles, n_subspaces)

    return particles


if __name__ == '__main__':
    consts = [0.5, 0.5, 0.5]

    eval_func = lambda p: np.sum (p**2)

    partitioned_pso (n_partitions=9, n_particles=6, n_vars=9, 
                        n_particles_part=2, n_vars_part=3, 
                        consts=consts, eval_func=eval_func)