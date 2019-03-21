import unittest
import numpy as np
import random
import hybrid_pso_ga as hpsoga

class TestGA (unittest.TestCase):
    def test_split_particles (self):
        particles_1 = np.array ([[0,0,1,0],[0,1,0,0],[1,1,1,1],[1,1,0,1]])
        particles_2 = np.array ([[1,5,2,3,4,6],[1,4,6,7,8,3],
                                 [9,8,2,1,1,1],[3,3,2,4,9,9]])

        partition_1 = np.array ([[[0],[0]],[[1],[1]],[[0],[1]],[[1],[1]],
                                 [[1],[0]],[[1],[0]],[[0],[0]],[[1],[1]]])

        partition_2 = np.array ([[[1,5],[1,4]],[[9,8],[3,3]],[[2,3],[6,7]],
                                 [[2,1],[2,4]],[[4,6],[8,3]],[[1,1],[9,9]]])
        #n_solutions = 2 n_vars = 1
        np.testing.assert_equal (
            hpsoga.split_particles (particles_1, 2, 1), partition_1)                   
        #n_solutions = 2 n_vars = 2
        np.testing.assert_equal (
            hpsoga.split_particles (particles_2, 2, 2), partition_2)

    def test_split_and_crossover (self):
        population = np.array ([[ 0.5, 0.3, 0.2, 0.1, 0.2, 0.3, 0.9, 0.7],
                                [-0.2, 0.5, 0.7,-0.9, 0.8, 0.8, 0.8, 0.2],
                                [ 0.2,-0.8, 0.1, 0.9, 0.5, 0.1, 0.1,-0.2],
                                [ 0.7, 0.7, 0.7, 0.1 ,0.4,-0.4, 0.2, 0.0]]
        )
        prob_cross, c = 0.0, 0.3
        n_particles_part, n_vars_part = 4, 4 

        cross_pop = hpsoga.split_and_crossover (population, 
                n_particles_part,  n_vars_part, prob_cross, c)
        self.assertIsNone (cross_pop)

        prob_cross = 1.0
        cross_pop = hpsoga.split_and_crossover (population, 
                n_particles_part,  n_vars_part, prob_cross, c)
        #print (cross_pop.shape)
        self.assertTrue (cross_pop.shape == (8,8))

if __name__ == '__main__':
    unittest.main()
