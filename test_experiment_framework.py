import unittest
import numpy as np
import experiment_framework as ef

class TestExperimentsFramework(unittest.TestCase):
    def test_create_grid_params(self):
        dict_params = {'p1':[1, 2, 3], 'p2':['X'], 'p3':['a', 'b']}
        self.expected = [{'p1':1, 'p2':'X', 'p3':'a'},
                         {'p1':1, 'p2':'X', 'p3':'b'},
                         {'p1':2, 'p2':'X', 'p3':'a'},
                         {'p1':2, 'p2':'X', 'p3':'b'},
                         {'p1':3, 'p2':'X', 'p3':'a'},
                         {'p1':3, 'p2':'X', 'p3':'b'}
                        ]
        self.assertEqual(ef.create_grid_params(dict_params), self.expected)

    def test_run_pso_experiments(self):
        # 12 possible combinations of parameters.
        self.pso_params = {
            "pop_size": [5, 10, 20],
            "particle_size": [10, 20],
            "max_iters": [50],
            "consts": [[0.55, 1.49618, 1.49618], [0.72, 1.49618, 1.49618]]
        }
        self.pso_results = ef.run_pso_experiments(
                                self.pso_params, 'rosenbrock', n_runs=5)
        self.assertEqual(list(self.pso_results.columns), 
                        ['run','fitness','omega','c1','c2'])
        # 12 combinations of params, 50 iterations, 5 runs.
        # 12 x 50 x 5 = 3000
        self.assertEqual(self.pso_results.shape, (3000,5))

if __name__ == '__main__':
    unittest.main()
