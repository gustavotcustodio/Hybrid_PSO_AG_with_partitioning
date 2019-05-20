import random
import numpy as np


def normalize_fitness(fitness_vals, task='max'):
    """
    Normalize fitness values between 1.0 and 2.0.

    When the GA problem is to minimize an objective function, invert the
    fitness values. therefore, the particles with lower fitness have a
    higher probebility of being selected for the next generation.
    """
    fitness_norm = ((fitness_vals-min(fitness_vals))/
                    (max(fitness_vals)-min(fitness_vals)+1.0) + 1.0)

    return (1.0/fitness_norm) if task=='min' else fitness_norm


def selection(population, fitness_vals, n_to_select, task):
    """
    Select the top n_to_select chromosomes according to 
    their fitness values.
    """
    fitness_norm = normalize_fitness (fitness_vals, task)
    top_indices = np.argsort (fitness_norm)[::-1] [0:n_to_select]
    return population [top_indices], top_indices


def roulette_selection(population, n_to_select, fitness_vals, task):
    """
    Roulette selection of genetic algorithm.

    Parameters
    ----------
    population: 2d array
        Candidate solutions.
    n_to_select: int
        Number of candidate solutions to keep for the next generation.
    fitness_vals: 1d array
        Fitness values of candidate solutions.
    task: 'min' or 'max'
        Minimization or maximization problem.

    Returns
    -------
    selected_pop: 2d array
        'n_to_select' best selected chromosomes.
    """
    indices = []
    fitness_norm = normalize_fitness(fitness_vals, task)
    f_cumsum = np.cumsum(fitness_norm / sum (fitness_norm))

    for _ in range(n_to_select):
        # Get the index of the first element equal or lower 
        # than a random number from 0 to 1.
        index = np.searchsorted(f_cumsum, random.random())
        indices.append(index)        
    return population[indices]


def random_arith_crossover(population):
    """."""
    rd = np.random.uniform(0, 1, size=(population.shape))
    cross_pop = rd[:-1]*population[:-1] + (1-rd[:-1])*population[1:]
    # Last chromosome
    last_chrom = rd[-1]*population[-1] + (1-rd[-1])*population[0]
    return np.append(cross_pop, [last_chrom], axis = 0)


def arith_crossover(population, prob_cross, c):
    """
    Combine multiple chromosomes to genereate new ones.

    Attributes
    ----------
    population: 2d array
        Chromosomes representing candidate solutions (parents).
    prob_cross: float
        Probability of occurring a crossover between
        two chromosomes ranging from 0 to 1.
    c: float
        Weighting factor for crossover.

    Returns
    -------
    offsprings: 2d array
        Chromosomes resulting from the parents' crossover.
    """
    pop_size = population.shape[0]
    parents = np.where(np.random.uniform(0, 1, pop_size)<prob_cross)[0]
    # if only one chromosome is selected, there is no crossover  
    if parents.size == 0:
        return None, None

    offsprings = np.empty(shape=(len(parents), population.shape[1]))
    # if an odd number of chromosomes is selected, 
    # don't include the last one in the crossover.
    if len(parents) % 2 != 0:
        last_parent = parents[-1]
        indices = parents[:-1]
    else: 
        last_parent = None
        indices = parents
    index_off = 0
    indices = indices.reshape(int(indices.shape[0]/2), 2)
    for i in indices:
        p0, p1 = population[i[0]], population[i[1]]
        offsprings[index_off] = c*p0 + (1-c)*p1
        offsprings[index_off+1] = c*p1 + (1-c)*p0
        index_off += 2
    # if there is an odd number of parents, the last one is cloned
    if last_parent is not None:
        offsprings[-1] = np.copy(population[last_parent])
    return offsprings, parents


def single_point_crossover(population):
    """
    Perform single-point crossover operation between multiple particles.

    Parameters
    ----------
    population: 2d array
        Population of candidate solutions.

    Returns
    -------
    offsprings: 2d array
        Population generated from crossover operation.
    """
    n_chroms = population.shape[0]
    chrom_size = population.shape[1]
    if chrom_size <= 1:
        return np.copy(population)  

    offsprings = np.empty(shape=(n_chroms, chrom_size))
    for c in range(0, n_chroms, 2):
        if c < n_chroms-1:
            pos_cross = random.randint(1, chrom_size-1)
            offsprings[c] = np.append(population[c, :pos_cross],
                                      population[c+1, pos_cross:])
            offsprings[c+1] = np.append(population[c+1, :pos_cross],
                                        population[c, pos_cross:])
        else:
            offsprings[c] = np.copy(population[c])
    return offsprings


def arith_mutation(population, prob_mut, l_bound=-1, u_bound=1):
    """
    Change the value of random positions in chromosomes.

    Attributes
    ----------
    population: 2d array
        Group of chromosomes.
    prob_mut: float
        Probability to change the value in a chromosome's position.
    u_bound: float
        Maximum value allowed to a chromosome.
    l_bound: float
        Minimum value allowed to a chromosome.

    Returns
    -------
    mut_population: 2d array
        Array with the new population after mutation operator applied.
    """
    n_chromosomes = population.shape[0]
    n_dims = population.shape[1]
    mut_population = np.copy(population)

    # create 2d array with random values
    random_2d_array = np.random.uniform (0, 1, size=(n_chromosomes, n_dims))

    # get indices in 2d array with lower values than prob_mut
    indices_change = np.where (random_2d_array < prob_mut)
    n_mutations = indices_change[0].shape[0]

    changed_values = [random.uniform (l_bound, u_bound
                        ) for _ in range(n_mutations)]
    mut_population[indices_change] = changed_values
    return mut_population


def gauss_mutation(population, prob_mut, u_bound=1, l_bound=-1):
    n_chromosomes = population.shape[0]
    mut_population = np.copy(population)

    # create an array with random values
    random_array = np.random.uniform(0, 1, size=n_chromosomes)

    # get chromosomes with lower values than prob_mut
    indices_change = np.where(random_array < prob_mut)[0]
    # random normal distribution number
    r = random.gauss(mu=0, sigma=1)
    mut_population[indices_change] = mut_population[indices_change] + r
    return mut_population


def mutation(population, prob_mut, possible_values):
    """
    Randomly change the value of chromosomes positions to a value in a list.

    Parameters
    ----------
    population: 2d array
    prob_mut: float
    possible_values: list

    Returns
    -------
    final_pop: 2d array
    """
    n_chroms = population.shape[0]
    chrom_size = population.shape[1]
    # mut_pop is a copy of the population with all values changed.
    mut_pop = np.copy(population)
    final_pop = np.copy(population)

    # create 2d array with random values
    random_2d_array = np.random.uniform(0, 1, size=(n_chroms, chrom_size))
    # get indices in 2d array with lower values than prob_mut
    indices_mut = np.where(random_2d_array < prob_mut)
    for v in possible_values:
        indices_change = np.where(population==v)
        pv = possible_values.copy()
        # remove the values v from list
        del pv[v]
        mut_pop[indices_change] = np.random.choice(
                                    pv, size=len(indices_change[0]))
    final_pop[indices_mut] = mut_pop[indices_mut]
    return final_pop
    

def run_single_ga_iter(population, prob_cross, prob_mut, fitness_func,
                       c=0.5, l_bound=-1, u_bound=1, task='min'):
    """."""
    ga_pop = np.copy(population)
    n_to_select = ga_pop.shape[0]

    offsprings, _ = arith_crossover(ga_pop, prob_cross, c)
    if offsprings is not None:
        ga_pop = np.append(ga_pop, offsprings, axis=0)
    ga_pop = arith_mutation(ga_pop, prob_mut, l_bound, u_bound)

    fitness_vals = np.apply_along_axis(fitness_func, 1, ga_pop)
    return roulette_selection(ga_pop, n_to_select, fitness_vals, task)


def run_ga(pop_size, chrom_size, n_gens, fitness_func, prob_mut,
           possible_values, task='min'):
    """."""
    indices_pop = np.random.randint(0, len(possible_values), 
                                    size=(pop_size, chrom_size))
    ga_pop = np.array([[possible_values[i] for i in row] 
                       for row in indices_pop])
    for _ in range(n_gens):
        offsprings = single_point_crossover(ga_pop)
        if offsprings is not None:
            ga_pop = np.append(ga_pop, offsprings, axis=0)

        ga_pop = mutation(ga_pop, prob_mut, possible_values)
        fitness_vals = np.apply_along_axis(fitness_func, 1, ga_pop)

        ga_pop = roulette_selection(ga_pop, pop_size, fitness_vals, task)

    fitness_vals = np.apply_along_axis(fitness_func, 1, ga_pop)
    best_chrom = np.argmin(fitness_vals)
    return ga_pop[best_chrom]