import numpy as np
import random

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
        *****

    Returns
    -------
    cross_pos: 2d array
        Population of chromosomes after crossover.
    '''
    size_pop = population.shape[0]

    indices = np.where(
        np.random.uniform (0,1,size_pop) < prob_cross)[0]

    # if only one chromosome is selected, there is no crossover  
    if indices.shape[0] <= 1:
        return population

    # if a odd number of chromosomes is selected, don't include
    # the last one in the crossover
    if indices.shape[0] % 2 !=0:
        indices = indices[:-1]

    cross_pop = np.copy (population)

    indices = indices.reshape ( int( indices.shape[0]/2 ), 2)
    for i in indices:
        cross_pop[i[0]] = c*cross_pop[i[0]] + (1 - c)*cross_pop[i[1]]
        cross_pop[i[1]] = c*cross_pop[i[1]] + (1 - c)*cross_pop[i[0]]

    return cross_pop

def mutation (population, prob_mut, u_bound=1, l_bound=-1):
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
                            c = 0.5, l_bound=-1, u_bound=1):
    for i in range (partitions.shape[0]):
        cross_pop = crossover (partitions[i], prob_cross, c)
        partitions[i] = mutation (cross_pop, prob_mut, u_bound, l_bound)
