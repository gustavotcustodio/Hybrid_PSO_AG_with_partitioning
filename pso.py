import numpy as np


def generate_single_array (n_dimensions, l_bound, u_bound):
    '''
    Generates a single array with float values ranging from
    l_bound to u_bound

    Parameters
    ----------
    n_dimensions: int
        Number of array dimensions.
    l_bound: float
        Min value allowed for a position in the array.
    u_bound: float
        Max value allowed for a position in the array.

    Returns
    -------
    single_array: 1d array
    '''
    return np.random.uniform (l_bound, u_bound, n_dimensions)

def generate_particles (pop_size, particle_size, l_bound, u_bound):
    '''
    Parameters
    ----------
    pop_size: int
        Number or particles to be generated.
    particle_size: int
        Number of dimensions of particles.
    l_bound: float
        Min value allowed for the position of a particle.
    u_bound: float
        Max value allowed for the position of a particle.

    Returns
    -------
    positions: 2d array
        Matrix containing the positions of all PSO particles.
    velocities: 2d array
        Matrix containing particles' velocities.
    '''
    # generate multiple arrays to represent PSO particles
    particles = np.array(
        [generate_single_array (particle_size, l_bound, u_bound)
        for _ in range(pop_size)]
    )
    return particles


def generate_velocities (pop_size, particle_size, l_bound, u_bound):
    '''
    Generate an array of arrays containing the initial velocities
    for all PSO particles.

    Parameters
    ----------
    pop_size: int
        Number of particles.
    particle_size: int
        Number of dimensions for each particle.
    l_bound: float
        Min value allowed for the position of a particle in space,
        used to calculate the min and max velocity.
    u_bound: float
        Max value allowed for the position of a particle in space,
        used to calculate the min and max velocity.

    Returns
    -------
    velocities: 2d array
        Initial velocities for all PSO particles.
    '''
    u_bound_vel = abs(u_bound - l_bound)
    l_bound_vel = -u_bound_vel

    velocities = np.array(
        [generate_single_array (particle_size, l_bound_vel, u_bound_vel)
        for _ in range(pop_size)])
    return velocities

def evaluate_particle (fitness, particle):
    '''
    Evaluate a particle using a fitness function.
    '''
    return fitness (particle)

def copy_particle (particle):
    return np.copy (particle)

def get_best_particle (particles, evals_parts):
    '''
    Get the particle with best evaluation.
    '''
    i_max = np.argmax (evals_parts)
    return particles [i_max]

def update_velocities (particles, best_parts, global_best, 
                                        velocities, consts):
    '''
    Parameters
    ----------
    particles: 2d array

    best_parts: 2d array
    global_best: 1d array
    velocities: 2d array
    const: list [float]
    '''
    pop_size = particles.shape[0]
    particle_size = particles.shape[1]

    # Matrices of random numbers
    r1 = np.random.uniform (0, 1, (pop_size, particle_size))
    r2 = np.random.uniform (0, 1, (pop_size, particle_size))

    # Tranform the global best solution in a 2d array,
    # repeating the array containing the global solution in each row
    mat_global_best = np.tile (global_best, (pop_size, 1))

    return           (consts[0] * velocities
            ) + (r1 * consts[1] * (best_parts-particles)
            ) + (r2 * consts[2] * (mat_global_best-particles))

def update_positions (positions, velocities):
    '''
    Update the position of a particle in the space
    acording to its velocity.

    Parameters
    ----------
    positions: 2d array
        Positions of PSO particles in the space.
    velocities: 2d array
        Velocities of PSO particles.
    '''
    return positions + velocities

def update_best_solutions (positions, best_parts, f_pos, f_best):
    '''
    Update the best known positions of PSO particles with the
    new best positions found.

    If the position x_i of a particle is better than p_i, update p_i.
    Repeat that for all particles.

    Parameters
    ----------
    positions: 2d array
        Positions of PSO particles in space.
    f_pos: 1d array
        Evaluations of particles according to their positions.
    f_best: 1d array
        Evaluations of best solutions found by particles.
    '''
    # Indices where the particles have better evaluation than the current best.
    indices_better = np.where (f_pos > f_best)[0]
    
    best_parts [indices_better] = np.copy (positions [indices_better])
    f_best [indices_better] = np.copy (f_pos [indices_better])

    return best_parts

def update_global_best (positions, global_best, f_pos, fg):
    '''
    Update the best known global solution, if a better particle is found.

    Parameters
    ----------
    positions: 2d array
        Positions of PSO particles in space.
    global_best: 1d array
        Best solution found so far by the PSO.
    f_pos: 1d array
        Evaluations of particles according to their positions.
    fg: float
        Evaluation of the best global solution.

    Returns
    -------
    new_global_best: 1d array
        Returns the same global_solution if no other best solution is found,
        otherwise replace the current global_solution with the best particle.
    new_fg: float
        New evaluation value of the global solution.
    '''
    index_best = np.argmax (f_pos)

    if f_pos [index_best] > fg:
        return np.copy( positions[index_best] ), f_pos[index_best]
    return global_best, fg


def split_particles (particles, n_subpops, n_subspaces):
    '''
    Split all PSO particles in partitions.
    Example of two particles: [[1,2,6,3,1,1], [9,9,1,2,4,7]]
        split in 2 subpops and 3 subspaces:

    1 2 | 6 3 | 1 1
    ---------------
    9 9 | 1 2 | 4 7

    Attributes
    ----------
    particles: 2d array
        All PSO particles.
    n_subpops: int
        Number of groups which the particles are divided.
    n_subspaces: int
        Number of parts to split each particle.

    Returns
    -------
    partitions: 3d array   
    '''

    subpops = np.array (np.split (particles, n_subpops))
    subspaces = np.array (np.split (subpops,2,axis=2))
    return np.concatenate (subspaces)

 
def merge_particles (particles, n_parts, n_dims):
    
    n_subpops     = n_parts / particles.shape[1]
    n_subspaces   = n_dims /  particles.shape[2]
    n_particles_per_block = particles.shape[1]

    total = n_subspaces * n_subpops
        
    merged_particles = np.array (
        [[particles [i:i+n_subspaces, j, :] for j in range(n_particles_per_block)] 
        for i in range(0, total, n_subspaces)]
    )

    return merged_particles.reshape (n_parts, n_dims)


def run_partitioned_pso (n_particles, n_dims, n_subpops, n_subspaces,
            consts, fitness, max_iter=100, u_bound=1, l_bound=-1):

    if n_particles % n_subpops != 0 or n_dims % n_subspaces != 0:
        return -float('infinity'), np.array([])

    particles = generate_particles (
                    n_particles, n_dims, l_bound, u_bound)
    velocities = generate_velocities (
                    n_particles, n_dims, l_bound, u_bound)
    best_parts = np.apply_along_axis (copy_particle, 1, particles)

    evals_parts = np.apply_along_axis (fitness, 1, particles)
    evals_best = np.copy (evals_parts)

    g = get_best_particle (best_parts, evals_best)
    eval_g  = fitness (g)

    best_fitness = []

    for _ in range(max_iter):
        
        update_velocities (particles[i], best_parts[i], g, velocities[i], consts)

        particles[i] = update_positions (particles[i], velocities[i])
        evals_parts[i] = fitness (particles[i])

        #best_parts[i], g, evals_best[i], eval_g = update_best (
        #    particles[i], best_parts[i], g, evals_parts[i], evals_best[i], eval_g)

        best_fitness.append (eval_g)

    return g, np.array(best_fitness)


if __name__ == '__main__':
    selection = np.array([0,1,4,6])

    selection = selection.reshape( selection.shape[0]/2, 2)

    for i in selection:
        print (i[0])
        print (i[1])

    n_particles = 6
    n_dims = 4
    n_subpops = 2
    n_subspaces = 2
    consts = [0.7, 1.4, 1.4]
    fitness = lambda positions: np.sum(positions**2)

    #g, res = run_pso (n_particles, n_dims, n_subpops, 
    #                n_subspaces, consts, fitness)

    #print (res)