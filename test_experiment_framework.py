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

    def test_cluster_run_pso_experiments(self):
        pso_params = {
            "pop_size": [18, 36],
            "particle_size": [18],
            "max_iters": [10],
            "consts": [[0.55, 1.49618, 1.49618]]}
        cluster_params = {
            "iris":{
                "n_clusters":[3, 6],
                "n_attrib": 4
            }}
        pso_results = ef.run_cluster_pso_experiments(
                pso_params, cluster_params['iris'], 'davies_bouldin',
                n_runs=2, dataset_name='iris')
        self.assertEqual(
                list(pso_results.columns), ['c1', 'c2', 'fitness',
                'max_iters', 'omega', 'pop_size', 'run', 'n_clusters'])
        # 4 combinations of params, 10 iterations, 2 runs.
        # 4 x 10 x 2 = 80
        self.assertEqual(pso_results.shape, (80, 8))

    def test_run_pso_experiments(self):
        # 12 possible combinations of parameters.
        pso_params = {
            "pop_size": [5, 10, 20],
            "particle_size": [10, 20],
            "max_iters": [50],
            "consts": [[0.55, 1.49618, 1.49618], [0.72, 1.49618, 1.49618]]
        }
        pso_results = ef.run_pso_experiments(pso_params, 'rosenbrock',
                                             n_runs=5)
        self.assertEqual(
                list(pso_results.columns), ['c1', 'c2', 'fitness',
                'max_iters', 'omega', 'pop_size', 'run', 'particle_size'])
        # 12 combinations of params, 50 iterations, 5 runs.
        # 12 x 50 x 5 = 3000
        self.assertEqual(pso_results.shape, (3000,8))

    def test_run_logapso_experiments(self):
        # 12 possible combinations of parameters.
        pso_params = {
            "pop_size": [4, 8],
            "particle_size": [10],
            "max_iters": [20],
            "consts": [[0.72, 1.49618, 1.49618]]}
        ga_params = {
            "prob_cross": [0.8],
            "prob_mut": [0.02, 0.05, 0.2],
            "pop_ga": [10],
            "c": [0.5],
            "n_gens": [15]}
        logapso_params = {
            "prob_run_ga": [0.1], 
            "step_size": [0.2]}

        attributes = ['c1', 'c2', 'fitness', 'max_iters', 'omega', 'pop_size',
                      'run', 'prob_run_ga', 'step_size', 'particle_size']
        pso_results = ef.run_logapso_experiments(
                pso_params, ga_params, logapso_params, 'rosenbrock', n_runs=2)

        self.assertEqual(list(pso_results.columns), attributes)


if __name__ == '__main__':
    unittest.main()
