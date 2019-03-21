import numpy as np
import random

def roulette_selection (population, n_to_select, fitness_vals):
    indices = []
    f_norm = (fitness_vals - min (fitness_vals)) / (
                max (fitness_vals) - min (fitness_vals))
    f_norm =  f_norm / sum (f_norm)
    f_cumsum = np.cumsum (f_norm)

    for _ in range (n_to_select):
        ''' get the index of the first element equal or lower 
        than a random number from 0 to 1 '''
        index = np.searchsorted (f_cumsum, random.random())
        indices.append (index)        
    return population [indices]

def crossover (population, prob_cross, c):
    '''
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
    '''
    pop_size = population.shape[0]
    parents = np.where (np.random.uniform (0, 1, pop_size) < prob_cross)[0]
    # if only one chromosome is selected, there is no crossover  
    if parents.size == 0:
        return None, None

    offsprings = np.empty( shape= (len(parents), population.shape[1]) )
    ''' if an odd number of chromosomes is selected, 
    don't include the last one in the crossover '''
    if len (parents) % 2 !=0:
        last_parent = parents[-1]
        indices = parents[:-1]
    else: 
        last_parent = None
        indices = parents

    index_off = 0
    indices = indices.reshape ( int( indices.shape[0]/2 ), 2)
    for i in indices:
        p0, p1 = population[i[0]], population[i[1]]
        offsprings [index_off]     = c * p0 + (1 - c) * p1
        offsprings [index_off + 1] = c * p1 + (1 - c) * p0
        index_off += 2
    # if there is an odd number of parents, the last one is cloned
    if last_parent is not None:
        offsprings[-1] = np.copy (population [last_parent])
    return offsprings, parents

def mutation (population, prob_mut, u_bound = 1, l_bound = -1):
    '''
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
    '''
    n_chromosomes = population.shape[0]
    n_dims = population.shape[1]
    mut_population = np.copy (population)

    # create 2d array with random values
    random_2d_array = np.random.uniform (0, 1, size=(n_chromosomes, n_dims))

    # get indices in 2d array with lower values than prob_mut
    indices_change = np.where (random_2d_array < prob_mut)
    n_mutations = indices_change[0].shape[0]

    changed_values = [random.uniform (l_bound, u_bound
                                        ) for _ in range(n_mutations)]
    mut_population [indices_change] = changed_values
    return mut_population

def run_single_ga_iter (population, prob_cross, prob_mut, fitness_func, 
                        c = 0.5, l_bound = -1, u_bound = 1):
    '''
    '''
    ga_pop = np.copy (population)
    n_to_select = ga_pop.shape[0]

    offsprings, parents = crossover (ga_pop, prob_cross, c)
    if offsprings is not None:
        ga_pop = np.append (population, offsprings, axis = 0)
    #mut_pop = mutation (population, prob_mut)
    print (ga_pop)
    print (parents)

    fitness_vals = np.apply_along_axis (fitness_func, 0, ga_pop)
    return roulette_selection (ga_pop, n_to_select, fitness_vals)