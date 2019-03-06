import unittest
import numpy as np
import random
import hybrid_pso_ga

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
            hybrid_pso_ga.split_particles (particles_1, 2, 1), partition_1)                   
        #n_solutions = 2 n_vars = 2
        np.testing.assert_equal (
            hybrid_pso_ga.split_particles (particles_2, 2, 2), partition_2)


    def test_merge_particles (self):
        partition_1 = np.array ([[[0],[0]],[[1],[1]],[[0],[1]],[[1],[1]],
                                 [[1],[0]],[[1],[0]],[[0],[0]],[[1],[1]]])

        partition_2 = np.array ([[[1,5],[1,4]],[[9,8],[3,3]],[[2,3],[6,7]],
                                 [[2,1],[2,4]],[[4,6],[8,3]],[[1,1],[9,9]]])

        particles_1 = np.array ([[0,0,1,0],[0,1,0,0],[1,1,1,1],[1,1,0,1]])
        particles_2 = np.array ([[1,5,2,3,4,6],[1,4,6,7,8,3],
                                 [9,8,2,1,1,1],[3,3,2,4,9,9]])
        #n_subspaces = 4
        np.testing.assert_equal (
            hybrid_pso_ga.merge_particles (partition_1, 4), particles_1)                         
        #n_subspaces = 3
        np.testing.assert_equal (
            hybrid_pso_ga.merge_particles (partition_2, 3), particles_2)

if __name__ == '__main__':
    unittest.main()
