import unittest
import numpy as np
import random
import genetic

class TestGA (unittest.TestCase):
    def test_crossover (self):
        population = np.array ([[0.1, 0.4, 0.4], [0.2, 0.3, 0.3], 
                                [0.5, 0.7, 0.8], [0.3, 0.9, 0.9],
                                [0.2, 0.9, 0.9]] )
        self.assertIsNone (genetic.crossover (population, 0, 0)[0])

        prob_cross = 1.0
        c = 0.2
        cross_pop, parents = genetic.crossover (population, prob_cross, c)

        self.assertTrue (cross_pop.shape == population.shape)
        np.testing.assert_array_equal (population[parents], population)
    
    def test_mutation (self):
        chromosomes = np.array (
            [[0.2, 0.3, 0.4], [0.2, 0.3, 0.3], [0.5, 0.7, 0.8]] )
        prob_mut = 0.0
        np.testing.assert_equal (
            genetic.mutation (chromosomes, prob_mut), chromosomes)

        array_to_compare = np.zeros((3, 3))
        prob_mut = 0.5

        np.testing.assert_allclose (
            genetic.mutation (chromosomes, prob_mut), 
            array_to_compare, atol=1.0
        )
        np.testing.assert_allclose (
            genetic.mutation (
                chromosomes, prob_mut, l_bound=-0.1, u_bound=0.1), 
            array_to_compare, atol=1.0)        

    def test_selection (self):
        np.random.seed (0)
        population = np.array ([[1,5,8,9,9,1], [0,1,7,8,7,7],
                                [7,1,2,2,3,5], [3,2,9,2,0,0]])
        n_to_select = 2
        fitness_vals = np.array ([sum(p**2) for p in population])
        pop_selected = genetic.roulette_selection (
                        population, n_to_select, fitness_vals)

        self.assertTrue (pop_selected.shape == (2,6))
        self.assertTrue (pop_selected[0] in population)
        self.assertTrue (pop_selected[1] in population)

    def test_run_single_ga_iter (self):
        population = np.array ([[-0.6, 0.5, 0.2], [0.1, 0.2, 0.5],
                                [0.8, -0.2, 0.8], [0.9, -0.9, 0.3]])
        fitness_func = lambda x: sum(x**2)

        result = genetic.run_single_ga_iter (
                    population, 1.0, 0.2, fitness_func, c = 0.2, 
                    l_bound = -1, u_bound = 1)
        array_to_compare = np.array (
                            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        np.testing.assert_allclose (result, array_to_compare, atol=1.0) 

if __name__ == '__main__':
    unittest.main()
