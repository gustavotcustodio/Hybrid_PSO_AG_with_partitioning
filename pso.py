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


def copy_particle (particle):
    return np.copy (particle)


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


def evaluate_particles (eval_func, particles):
    '''
    Evaluate particles using an evaluation function.

    Returns
    -------
    evals_particles: 1d array
        Evaluation of PSO particles.
    '''
    evals = [eval_func (p) for p in particles]

    return np.array (evals)


def get_best_particle (particles, evals_parts, task = 'min'):
    '''
    Get the particle with best evaluation.
    The task parameter says if it is a minimization or
    maximization problem.

    Returns
    -------
    best_particle: 1d array
        Position of particle with best evaluation.
    eval_best: float
        Evaluation of best particle.
    '''
    if task == 'min':    
        i_min = np.argmax (evals_parts)
        return np.copy (particles [i_min]), evals_parts[i_min]
    if task == 'max':    
        i_max = np.argmax (evals_parts)
        return np.copy (particles [i_max]), evals_parts[i_max]

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


def update_best_solutions (
        positions, best_parts, evals_parts, evals_best, task='min'):
    '''
    Update the best known positions of PSO particles with the
    new best positions found.

    If the position x_i of a particle is better than p_i, update p_i.
    Repeat that for all particles.

    Parameters
    ----------
    particles: 2d array
        Positions of PSO particles in space.
    evals_parts: 1d array
        Evaluations of particles according to their positions.
    evals_best: 1d array
        Evaluations of best solutions found by particles.
    task: string
        Min if the better solution is the one with lower evaluation, 
        max if the better solution is the one with higher evaluation.
    '''
    if task == 'min':
        # Indices where the particles have better eval than the current best.
        indices_better = np.where (evals_parts < evals_best)[0]
    else:
        indices_better = np.where (evals_parts > evals_best)[0]
    best_parts [indices_better] = np.copy (
                                positions [indices_better])
    evals_best [indices_better] = np.copy (
                                evals_parts [indices_better])


def update_global_best (
    particles, global_best, evals_parts, eval_global, task='min'):
    '''
    Update the best known global solution, if a better particle is found.

    Parameters
    ----------
    particles: 2d array
        Positions of PSO particles in space.
    global_best: 1d array
        Best solution found so far by the PSO.
    evals_parts: 1d array
        Evaluations of particles according to their positions.
    eval_global: float
        Evaluation of the best global solution.

    Returns
    -------
    new_global_best: 1d array
        Returns the same global_solution if no other best solution is found,
        otherwise replace the current global_solution with the best particle.
    new_eval_global: float
        New evaluation value of the global solution.
    '''
    if task =='min':
        index_best = np.argmin (evals_parts)
        if evals_parts [index_best] < eval_global:
            return np.copy (particles[index_best] ), evals_parts[index_best]
    else:
        index_best = np.argmax (evals_parts)
        if evals_parts [index_best] > eval_global:
            return np.copy (particles[index_best] ), evals_parts[index_best]

    return global_best, eval_global


def run_pso (eval_func, consts, max_iter = 100, pop_size=100, particle_size=10,
            initial_particles = None, l_bound=-1.0, u_bound=1.0, task='min'):
    '''
    Run the PSO algorithm for max_iter iterations.

    Parameters
    ----------
    eval_func: function_1_param
        Function to evaluate particles.
    max_iter: int
        Number of PSO iterations.
    consts: list [float]
        Constants for updating PSO velocity.
    pop_size: int
        Number of PSO particles.
    particle_size: int
        Number of dimensions for each PSO particle.
    initial_particles: 2d array or None
        Initial particles for the PSO, if there are
        none, start the particles randomly.
    l_bound: float
        Minimum value for a particle.
    u_bound: float
        Maximum value for a particle.
    task: string
        'min' for minimization problems and 'max' for maximization.

    Returns
    -------
    particles: 2d array
        2d array with each particle in a row after the PSO execution.
    global_solutions: 2d array
        Best global solutions found by PSO in each iter.
    best_evals: list[float]
        Evalulations for global best solution in each iteration.
    '''
    if initial_particles is None:
        particles = generate_particles (pop_size, particle_size, l_bound, u_bound)
    else:
        particle_size = initial_particles.shape[1]
        pop_size = initial_particles.shape[0]
        particles = initial_particles

    evals_parts = evaluate_particles (eval_func, particles)
    
    best_parts = np.copy (particles)
    evals_best = np.copy (evals_parts)

    global_best, eval_global = get_best_particle (particles, evals_parts, task)

    velocities = generate_velocities (pop_size, particle_size, l_bound, u_bound)

    global_solutions, best_evals = [], []

    for _ in range (max_iter):
        velocities = update_velocities (
                        particles, best_parts, global_best, velocities, consts)
        particles  = update_positions  (particles, velocities)
        evals_parts = evaluate_particles (eval_func, particles)

        update_best_solutions (
                    particles, best_parts, evals_parts, evals_best, task)

        global_best, eval_global = update_global_best (
                    particles, global_best, evals_parts, eval_global, task)
        
        global_solutions.append (global_best)
        best_evals.append (eval_global)
    return particles, np.array (global_solutions), best_evals

if __name__ == '__main__':
    selection = np.array([0,1,4,6])

    selection = selection.reshape( int(selection.shape[0]/2), 2)

    n_particles = 6
    n_dims = 4
    n_subpops = 2
    n_subspaces = 2
    consts = [0.7, 1.4, 1.4]
    eval_func = lambda p: np.sum (p**2)

    particles, global_solutions, best_evals = run_pso (
                    eval_func, consts, max_iter=50, pop_size=10)