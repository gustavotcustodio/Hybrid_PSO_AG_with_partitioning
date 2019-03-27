import numpy as np
import math

def square_sum(x):
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

def ackley (x):
    n = len (x)
    term1 = -20 * math.exp (-0.2 * np.sqrt(sum(x**2)/n))
    term2 = -math.exp ( sum (np.cos(2*math.pi*x))/n )
    return term1 + term2 + 20 + math.e

def schwefel (x):
    abs_x = np.abs (x)
    return  x * np.sin (np.sqrt(abs_x)) 

def get_function (function_name):
    if function_name =='square_sum':
        return square_sum
    elif function_name == 'rosenbrock':
        return rosenbrock
    elif function_name == 'rastrigin':
        return rastrigin
    elif function_name == 'griewank':
        return griewank
    else:
        return None
    
    
