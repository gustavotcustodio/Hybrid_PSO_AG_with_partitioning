import numpy as np
import random

def crossover (population, p_crossover, c):
    size_pop = population.shape[0]

    indices = np.argwhere(
        np.random.uniform (0,1,size_pop) < p_crossover)[:,0]
        
    if indices.shape[0] <= 1:
        return population
        
    if indices.shape[0] % 2 !=0:
        indices = indices[:-1]

    cross_pop = np.copy(population)

    indices = indices.reshape( indices.shape[0]/2, 2)

    for i in indices:
        cross_pop[i[0]] = c * cross_pop[i[0]] + (1 - c) * cross_pop[i[1]]
        cross_pop[i[1]] = c * cross_pop[i[1]] + (1 - c) * cross_pop[i[0]]

    return cross_pop