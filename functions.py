import numpy as np
import math
import random

def square_sum(x):
    return sum(np.square(x))	

def rosenbrock(x):
    x2 = np.array(x[1:])
    x1 = np.array(x[:len(x)-1]) 
    return sum(100 * np.square(x2-np.square(x1)) + np.square(x1-1))

def schwefel_222 (x):
    absx = np.abs (x)
    return sum (absx) + np.prod (absx)

def quartic_noise (x):
    indices = np.array (range (len(x)) )
    return sum (indices * x**4) + random.random()

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

def schwefel_226 (x): 
    # -500 to 500
    absx = np.abs (x)
    const = 418.982887272433799807913601398
    return const * len(x) - sum (x * np.sin (np.sqrt(absx)))

def u (a, k, m, x_i):
    if x_i > a: 
        return k * ( x_i - a)**m
    elif x_i < a:
        return k * (-x_i - a)**m
    else:
        return 0    

def penalty1 (x):
    # -50 to 50
    k, m, a = 100, 4, 10
    n = len (x)
    PI = math.pi
    y = 1.25 + x/4

    term1 = 10 * PI * math.sin (PI * y[0])** 2 / n
    term2 = sum ((y[:-1] - 1)** 2 * (1 + 10* math.sin (PI * y[1:])** 2))
    term3 = (y[-1] - 1) ** 2
    term4 = sum ([u (a, k, m, x_i) for x_i in x])
    return term1 + term2 + term3 + term4
    
def penalty2 (x):
    # -50 to 50
    k, m, a = 100, 4, 5
    PI = math.pi   

    term1 = 0.1 * math.sin (3*PI*x[0]) **2
    term2 = sum ((x - 1) **2 * (1 + np.sin (3*PI*x+1) **2))
    term3 = (x[-1] - 1) **2 * (1 + math.sin (2*PI*x[-1])) **2
    term4 = sum ([u (a, k, m, x_i) for x_i in x])
    return term1 + term2 + term3 + term4

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
    
    
