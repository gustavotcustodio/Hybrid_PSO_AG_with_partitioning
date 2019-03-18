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
        Chromosomes representing candidate solutions.
    prob_cross: float
        Probability of occurring a crossover between
        two chromosomes ranging from 0 to 1.
    c: float
        Weighting factor for crossover.

    Returns
    -------
    cross_pos: 2d array
        Population of chromosomes after crossover.
    '''
    size_pop = population.shape[0]
    indices = np.where (
                np.random.uniform (0, 1, size_pop) < prob_cross)[0]
    # This is used if the chromosomes selected for crossover is an odd number.
    extra_index = None
    # if only one chromosome is selected, there is no crossover  
    if indices.shape[0] <= 1:
        return population
    ''' if a odd number of chromosomes is selected, don't include
    the last one in the crossover '''
    if indices.shape[0] % 2 !=0:
        indices = indices[:-1]
        extra_index = indices[-1]
    indices = indices.reshape ( int( indices.shape[0]/2 ), 2)

    for i in indices:
        p0 = c * population[i[0]] + (1 - c) * population[i[1]]
        p1 = c * population[i[1]] + (1 - c) * population[i[0]]
        population = np.append (population, [p0], axis = 0)
        population = np.append (population, [p1], axis = 0)
    # if there is an odd number of chromosomes, the last one is cloned
    if extra_index is not None:
        population = np.append (population, 
                                [population[extra_index]], axis = 0)
    return population

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
    random_2d_array = np.random.uniform (
                        0, 1, size=(n_chromosomes, n_dims))

    # get indices in 2d array with lower values than prob_mat
    indices_change = np.where (random_2d_array < prob_mut)

    n_mutations = indices_change[0].shape[0]

    changed_values = [
        random.uniform (l_bound, u_bound) for _ in range(n_mutations)]

    mut_population [indices_change] = changed_values
    return mut_population

def run_ga (partitions, prob_cross, prob_mut, 
                fitness_func, c = 0.5, l_bound = -1, u_bound = 1):
    '''
    '''
    for i in range (len(partitions)):
        n_to_select = partitions[i].shape[0]
        cross_pop = crossover (partitions[i], prob_cross, c)

        mut_pop = mutation (cross_pop, prob_mut)
    
        fitness_vals = np.apply_along_axis (
                            fitness_func, 0, mut_pop)
        partitions[i] = roulette_selection (
                            partitions[i], n_to_select, fitness_vals)