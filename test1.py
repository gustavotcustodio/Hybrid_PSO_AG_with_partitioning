import unittest
import numpy as np
import random

import pso

class TestPSO (unittest.TestCase):
    def test_generate_single_array (self):
        np.random.seed(0)
        array_test = np.random.uniform(0,1,size=(10,))

        np.testing.assert_allclose (
            pso.generate_single_array (10, 0.0, 1.0), array_test, atol=1.0
        )
 
    def test_generate_particles (self):
        particles  = pso.generate_particles(100, 10, 0, 1)
        self.assertTrue (particles.shape == (100,10))
        self.assertTrue ((particles <= 1).all() and (particles >= 0).all())

    def test_generate_velocities (self):
        velocities  = pso.generate_velocities(100, 10, 0, 1)
        self.assertTrue (velocities.shape == (100,10))
        self.assertTrue ((velocities <= 1).all() and (velocities >= -1).all())

    def test_evaluate_particle (self):
        particle = np.array ([1.0, 0.8, 0.2, 0.4])
        fitness = lambda p: np.sum (p ** 2)
        self.assertAlmostEqual (pso.evaluate_particle (fitness, particle), 1.84 )

    def test_copy_particle (self):
        particle = np.array ([0.2, 0.1, 0.9])
        np.testing.assert_array_equal (pso.copy_particle(particle), particle)

    def test_get_best_particle (self):
        particles = np.array (
            [[0.2, 0.3, 0.4], [0.2, 0.3, 0.3], [0.5, 0.7, 0.8]] )
        f_values = np.array ([0.4, 0.3, 0.8])

        np.testing.assert_array_equal (
            pso.get_best_particle (particles, f_values), particles[2]
        )

    def test_update_velocities (self):
        particles  = np.random.uniform ( 0,1,size=(10,8))
        best_parts = np.random.uniform ( 0,1,size=(10,8))
        global_best= np.random.uniform ( 0,1,size=(8,))
        velocities = np.random.uniform (-1,1,size=(10,8))

        consts = [random.random() for i in range(3)]
        # Asserts that the resulting velocities are between -3 and 3.
        array_test = np.zeros (shape=(10,8))
        np.testing.assert_allclose (
            pso.update_velocities (particles, best_parts, global_best, 
                                    velocities, consts), 
            array_test, atol=3.0)
        
    def test_update_positions (self):
        x = np.array([[ 0.6, 0.2, 0.2], [0.1, 0.5, 0.1]])
        v = np.array([[-0.3, 0.1, 0.8], [0.4,-0.5,-0.1]])

        array_test = np.array([[ 0.3, 0.3, 1.0], [0.5, 0.0, 0.0]])

        np.testing.assert_array_almost_equal (
            pso.update_positions (x, v), array_test)

    def test_update_best_solutions (self):
        positions  = np.array ([[0.6, 0.2, 0.4], [0.0, 0.1, 0.1], [0.0, 0.0, 0.0]])
        best_parts = np.array ([[0.2, 0.1, 0.3], [0.1, 0.1, 0.1], [0.3, 0.3, 0.3]])

        f_pos  = np.sum (positions **2, axis=1)
        f_best = np.sum (best_parts**2, axis=1)

        correct_best_parts = np.array (
            [[0.6, 0.2, 0.4], [0.1, 0.1, 0.1], [0.3, 0.3, 0.3]])
        correct_f_best = np.sum (correct_best_parts**2, axis=1)

        pso.update_best_solutions (positions, best_parts, f_pos, f_best)

        np.testing.assert_array_equal (best_parts, correct_best_parts)
        np.testing.assert_array_equal (f_best, correct_f_best)

    def test_update_global_best (self):
        positions = np.array (
            [[1.0, 0.5, 0.4, 0.1], [0.9, 0.3, 0.5, 0.0], [0.2, 0.2, 0.1, 0.2]])
        global_best = np.array ([0.4, 0.2, 0.3, 0.1])

        f_pos = np.sum (positions **2, axis=1)
        fg = np.sum (global_best **2)

        correct_global_best = np.array ([1.0, 0.5, 0.4, 0.1])
        correct_fg = np.sum (correct_global_best**2)

        global_best, fg = pso.update_global_best (positions, global_best, f_pos, fg)

        np.testing.assert_array_equal (global_best, correct_global_best)
        np.testing.assert_array_equal (fg, correct_fg)

    def test_split_particles (self):
        particles_1 = np.array ([[0,0,1,0],[0,1,0,0],[1,1,1,1],[1,1,0,1]])
        particles_2 = np.array ([[1,5,2,3,4,6],[1,4,6,7,8,3],
                                 [9,8,2,1,1,1],[3,3,2,4,9,9]])

        partition_1 = np.array ([[[0],[0]],[[1],[1]],[[0],[1]],[[1],[1]],
                                 [[1],[0]],[[1],[0]],[[0],[0]],[[1],[1]]])

        partition_2 = np.array ([[[1,5],[1,4]],[[9,8],[3,3]],[[2,3],[6,7]],
                                 [[2,1],[2,4]],[[4,6],[8,3]],[[1,1],[9,9]]])
        #n_subpops = 2 n_subspaces = 4
        np.testing.assert_equal (
            pso.split_particles (particles_1, 2, 4), partition_1)
                            
        #n_subpops = 3 n_subspaces = 3
        np.testing.assert_equal (
            pso.split_particles (particles_2, 2, 3), partition_2)
                                
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
            pso.merge_particles (partition_1, 4), particles_1)
                            
        #n_subspaces = 3
        np.testing.assert_equal (
            pso.merge_particles (partition_2, 3), particles_2)
    #def test_run_pso (self):
    #    n_particles = 5
    #    n_dims = 4
    #    n_subpops = 2
    #    n_subspaces = 2
    #    consts = np.array([0.7, 1.4, 1.4])
    #    fitness = lambda x: np.sum(x**2)
        
    #    _, evals = pso.run_pso (n_particles, n_dims, n_subpops, 
    #                            n_subspaces, consts, fitness)

    #    np.testing.assert_array_equal (evals, np.sort(evals) )


if __name__ == '__main__':
    unittest.main()
