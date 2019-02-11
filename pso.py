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


def generate_particles (n_particles, n_dimensions, l_bound, u_bound):
    '''
    Generate the initial set of particle positions in a PSO.

    Parameters
    ----------
    n_particles: int
        Number of particles.
    n_dimensions: int
        Size of each particle.
    l_bound: float
        Min value allowed for the position of a particle in space.
    u_bound: float
        Max value allowed for the position of a particle in space.

    Returns
    -------
    particles: 2d array
        Array containing initial particle positions in a PSO.
    '''
    particles = np.array(
        [generate_single_array (n_dimensions, l_bound, u_bound)
        for _ in range(n_particles)]
    ) 
    return particles


def generate_velocities (n_particles, n_dimensions, l_bound, u_bound):
    '''
    Generate an array of arrays containing the initial velocities
    for all PSO particles.

    Parameters
    ----------
    n_particles: int
        Number of particles.
    n_dimensions: int
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
        [generate_single_array (n_dimensions, l_bound_vel, u_bound_vel)
        for _ in range(n_particles)])

    return velocities


def eval_particle (fitness, particle):
    '''
    Evaluate a particle using a fitness function.
    '''
    return fitness (particle)


def copy_particle (particle):
    return np.copy (particle)


def get_best_particle (particles, evals_p):
    '''
    Get the particle with best evaluation.
    '''
    i_max = np.argmax (evals_p)
    return particles [i_max]


def update_velocity (x_i, p_i, g, v_i, consts):
    n_dimensions = x_i.shape[0]

    r1 = np.random.uniform (0, 1, n_dimensions)
    r2 = np.random.uniform (0, 1, n_dimensions)

    return consts[0]*v_i + r1*consts[1]*(p_i-x_i) + r2*consts[2]*(g-x_i)


def update_position (position, velocity):
    '''
    Update the position of a particle in the space
    acording to its velocity.

    Parameters
    ----------
    position: 1d array
        Position of a PSO particle in the space.
    velocity: 1d array
        Velocity of a PSO particle.
    '''
    return position + velocity


def update_best (x_i, p_i, g, fx_i, fp_i, fg):
    '''
    Update the best known position p_i of a particle
    and the best global solution g.

    If the position x_i of a particle is better than p_i, update p_i.
    If the position x_i of a particle is better than g, update g.

    Parameters
    ---------
    x_i: 1d array
        Position of a particle i.
    p_i: 1d array
        Best solution found by particle i.
    g: 1d array
        Best global solution.
    fx_i, fp_i, g: float
        Evaluation of particle position 
        x_i, best solution p_i and global solution g.

    Returns
    -------
    x_i, x_i, fx_i, fx_i
        If the position x_i is better than p_i and g.
    x_i, g, fx_i, fg
        If the position x_i is better than p_i, but not g.
    p_i, g, fp_i, fg
        If the position x_i is better than none.
    '''
    if fx_i > fp_i:
        if fx_i > fg:
            return copy_particle(x_i), copy_particle(x_i), fx_i, fx_i
        return copy_particle(x_i), g, fx_i, fg
    return p_i, g, fp_i, fg


def split_particles (particles, n_subpops, n_subspaces):
    '''
    Split all PSO particles in partitions according to the
    proposal by ***.

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
    partitions: 3d array [partition, particle, particle_slice]
        PSO particles splitted in different partitions.
    '''
    n = particles.shape[0] / n_subpops
    m = particles.shape[1] / n_subspaces

    part_split = []

    for i in range(0, particles.shape[0], n):
        for j in range(0, particles.shape[1], m):
            part_split.append ( particles[i:i+n, j:j+m] )

    return np.array (part_split)

 
def merge_particles (particles, n_parts, n_dims):
    
    n_subpops     = n_parts / particles.shape[1]
    n_subspaces   = n_dims /  particles.shape[2]
    n_parts_block = particles.shape[1]

    total = n_subspaces * n_subpops
        
    merged_particles = np.array (
        [[particles [i:i+n_subspaces, j, :] for j in range(n_parts_block)] 
        for i in range(0, total, n_subspaces)]
    )

    return merged_particles.reshape (n_parts, n_dims)


def run_pso (n_particles, n_dims, n_subpops, n_subspaces,
                consts, fitness, max_iter=100, u_bound=1, l_bound=-1):

    if n_particles % n_subpops != 0 or n_dims % n_subspaces != 0:
        return -float('infinity'), np.array([])

    x = generate_particles  (n_particles, n_dims, l_bound, u_bound)
    v = generate_velocities (n_particles, n_dims, l_bound, u_bound)
    p = np.apply_along_axis (copy_particle, 1, x)

    evals_x = np.apply_along_axis (fitness, 1, x)
    evals_p = np.copy (evals_x)

    g = get_best_particle (p, evals_p)
    eval_g  = fitness (g)

    best_fitness = []

    for _ in range(max_iter):

        for i in range(n_particles):
            v[i] = update_velocity (x[i], p[i], g, v[i], consts)
            x[i] = update_position (x[i], v[i])
            evals_x[i] = fitness (x[i])

            p[i], g, evals_p[i], eval_g = update_best (
                x[i], p[i], g, evals_x[i], evals_p[i], eval_g)

        best_fitness.append (eval_g)

    return g, np.array(best_fitness)


if __name__ == '__main__':
    selection = np.array([0,1,4,6])

    selection = selection.reshape( selection.shape[0]/2, 2)

    for i in selection:
        print i[0]
        print i[1]

    n_particles = 6
    n_dims = 4
    n_subpops = 2
    n_subspaces = 2
    consts = [0.7, 1.4, 1.4]
    fitness = lambda x: np.sum(x**2)


    g, res = run_pso (n_particles, n_dims, n_subpops, 
                    n_subspaces, consts, fitness)

    print res