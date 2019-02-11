import unittest
import numpy as np
import genetic
import random


class TestGA (unittest.TestCase):
    def test_crossover (self):
        population = np.array (
            [[0.2, 0.3, 0.4], [0.2, 0.3, 0.3], [0.5, 0.7, 0.8]] )
        p_crossover = 0.5
        genetic.crossover (population, p_crossover, 0.5)
        c = random.random()

        np.testing.assert_allclose (
            genetic.crossover (population, p_crossover, c), population, atol=1.0
        )


if __name__ == '__main__':
    unittest.main()
