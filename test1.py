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

    def test_evaluate_particles (self):
        particle = np.array ([[1.0, 0.8, 0.2, 0.4], [0.2, 0.5, 0.5, 0.5]])
        evals = np.array([1.84, 0.79])
        eval_func = lambda p: np.sum (p ** 2)
        np.testing.assert_almost_equal (
                        pso.evaluate_particles (eval_func, particle), evals)

    def test_copy_particle (self):
        particle = np.array ([0.2, 0.1, 0.9])
        np.testing.assert_array_equal (pso.copy_particle(particle), particle)

    def test_get_best_particle (self):
        particles = np.array (
            [[0.2, 0.3, 0.4], [0.2, 0.3, 0.3], [0.5, 0.7, 0.8]] )
        f_values = np.array ([0.4, 0.3, 0.8])

        best_particle, eval_best = pso.get_best_particle (
                                        particles, f_values, task='max')

        np.testing.assert_array_equal (best_particle, particles[2])
        self.assertEqual (eval_best, f_values[2])

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
        positions = np.array ([[0.6, 0.2, 0.4], [0.0, 0.1, 0.1], [0.0, 0.0, 0.0]])
        evals_parts  = np.sum (positions **2, axis=1)

        ################ Test for maximization ################
        best_parts = np.array ([[0.2, 0.1, 0.3], [0.1, 0.1, 0.1], [0.3, 0.3, 0.3]])
        evals_best = np.sum (best_parts**2, axis=1)

        correct_best_parts = np.array (
            [[0.6, 0.2, 0.4], [0.1, 0.1, 0.1], [0.3, 0.3, 0.3]])
        correct_evals_best = np.sum (correct_best_parts**2, axis=1)

        pso.update_best_solutions (
            positions, best_parts, evals_parts, evals_best, task='max')

        np.testing.assert_array_equal (best_parts, correct_best_parts)
        np.testing.assert_array_equal (evals_best, correct_evals_best)

        ################ Test for minimization ################
        best_parts = np.array ([[0.3, 0.2, 0.3], [0.2, 0.1, 0.1], [0.3, 0.6, 0.3]])
        evals_best = np.sum (best_parts**2, axis=1)

        correct_best_parts = np.array (
            [[0.3, 0.2, 0.3], [0.0, 0.1, 0.1], [0.0, 0.0, 0.0]])
        correct_evals_best = np.sum (correct_best_parts**2, axis=1)

        pso.update_best_solutions (positions, best_parts, evals_parts, evals_best)

        np.testing.assert_array_equal (best_parts, correct_best_parts)
        np.testing.assert_array_equal (evals_best, correct_evals_best)
        

    def test_update_global_best (self):
        particles = np.array (
            [[1.0, 0.5, 0.4, 0.1], [0.9, 0.3, 0.5, 0.0], [0.2, 0.2, 0.1, 0.2]])
        global_best = np.array ([0.4, 0.2, 0.3, 0.1])

        evals_parts = np.sum (particles **2, axis=1)
        eval_global = np.sum (global_best **2)

        ################ Test for maximization ################
        correct_global_best = np.array ([1.0, 0.5, 0.4, 0.1])
        correct_eval_global = np.sum (correct_global_best**2)

        global_best, eval_global = pso.update_global_best (
                particles, global_best, evals_parts, eval_global, task='max')

        np.testing.assert_array_equal (global_best, correct_global_best)
        np.testing.assert_array_equal (eval_global, correct_eval_global)

        ################ Test for minimization ################
        correct_global_best = np.array ([0.2, 0.2, 0.1, 0.2])
        correct_eval_global = np.sum (correct_global_best**2)

        global_best, eval_global = pso.update_global_best (
                            particles, global_best, evals_parts, eval_global)
        np.testing.assert_array_equal (global_best, correct_global_best)
        np.testing.assert_array_equal (eval_global, correct_eval_global)
        

    def test_run_pso(self):
        particles = np.array (
                    [[ 1.0, 0.6,-0.2], [0.1,-0.2,-0.7], [ 0.3, 0.2, 0.1]])
        particles_copy = np.copy (particles)

        eval_func = lambda p: np.sum (p ** 2)
        max_iter = 10

        consts = [0.7, 0.7, 0.9]

        new_particles, global_solutions, best_evals = pso.run_pso (
                eval_func, consts, max_iter, initial_particles = particles_copy)

        np.testing.assert_allclose (new_particles, np.zeros((3,3)), atol=1.0 )
        np.testing.assert_allclose (global_solutions, np.zeros((10,3)), atol=1.0 )

        self.assertListEqual (best_evals, sorted(best_evals, reverse=True))



if __name__ == '__main__':
    unittest.main()
