import numpy as np
import math

def squareSum(x):
    return sum(np.square(x))	

def rosenbrock(x):
    x2 = np.array(x[1:])
    x1 = np.array(x[:len(x)-1]) 
    return sum(100 * np.square(x2-np.square(x1)) + np.square(x1-1))

def rastrigin(x):
    return sum(np.square(x) - 10 * np.cos(2 * math.pi * x) + 10)

def griewank(x):
    term1 = 1/4000 * sum(np.square(x - 100))
    term2 = np.prod (np.cos(
                (x - 100) / np.sqrt(range(1,len(x)+1)) 
            ))
    return  term1 - term2 + 1
