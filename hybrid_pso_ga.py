import numpy as np

def split_particles (particles, n_particles, n_vars):
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
    n_particles: int
        Number of particles in each partition.
    n_vars: int
        Number of variables in each partition.

    Returns
    -------
    partitions: 3d array [partition, particle, variable]
        Particles splitted in different partitions.
    '''
    n_subpops = int (particles.shape[0] / n_particles)
    n_subspaces = int (particles.shape[1] / n_vars)

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


def partitioned_pso (n_particles, n_vars, n_particles_part, n_vars_part,
                    consts, fitness, max_iter=100, u_bound=1.0, l_bound=-1.0):

    # Apply the standard PSO to all particles    

    # Apply the selection operator in all particles

    # Partition the population in sub-partitions

    # Apply crossover in all sub-partitions

    # Apply mutation

    return None