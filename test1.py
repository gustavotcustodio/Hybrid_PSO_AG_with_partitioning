import unittest
import numpy as np

import pso


class TestPSO (unittest.TestCase):
    def test_generate_single_array (self):
        array_test = np.array ([1.0, 0.0, 0.4, 0.7, 0.1838829])

        np.testing.assert_allclose (
            pso.generate_single_array (5, 0.0, 1.0), array_test, atol=1.0
        )        
 
    def test_generate_particles (self):
        particles  = pso.generate_particles(100, 10, 0, 1)
        condition1 = particles.shape == (100,10)
        condition2 = (particles <= 1).all() and (particles >= 0).all()

        self.assertTrue (condition1 and condition2)

    def test_generate_velocities (self):
        velocities  = pso.generate_velocities(100, 10, 0, 1)
        condition1 = velocities.shape == (100,10)
        condition2 = (velocities <= 1).all() and (velocities >= -1).all()

        self.assertTrue (condition1 and condition2)

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

    def test_update_position (self):
        x = np.array([ 0.6, 0.2,-0.2])
        v = np.array([-0.5, 0.4, 0.5])

        x_final = np.array([0.1, 0.6, 0.3])

        np.testing.assert_array_almost_equal (
            pso.update_position (x, v), x_final)

    #def test_update_velocities (self):
        

    def test_update_best (self):
        x = np.array([ 0.6, 0.2, 0.4]) #0.56
        p = np.array([ 0.4, 0.2, 0.1]) 
        g = np.array([ 0.3, 0.1, 0.1]) #0.11

        fx, fp, fg = np.sum(x**2), np.sum(p**2), np.sum(g**2)

        p, g, fp, fg = pso.update_best (x, p, g, fx, fp, fg)

        g_final = np.array([ 0.6, 0.2, 0.4])

        np.testing.assert_array_equal (g, g_final)


    def test_run_pso (self):
        n_particles = 5
        n_dims = 4
        n_subpops = 2
        n_subspaces = 2
        consts = np.array([0.7, 1.4, 1.4])
        fitness = lambda x: np.sum(x**2)
        
        _, evals = pso.run_pso (n_particles, n_dims, n_subpops, 
                                n_subspaces, consts, fitness)

        np.testing.assert_array_equal (evals, np.sort(evals) )


if __name__ == '__main__':
    unittest.main()
