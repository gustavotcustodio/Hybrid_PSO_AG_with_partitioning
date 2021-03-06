import numpy as np
import math
import random
import data_loader
from scipy.spatial import distance_matrix
from scipy.spatial import distance
from sklearn.metrics import davies_bouldin_score


def davies_bouldin(inputs, labels):
    m = inputs.shape[1] # number of attributes
    def wrapper(particle):
        d = int(particle.shape[0]/m) # number of clusters
        clusters = np.reshape(particle, (d, m))
        distances = distance_matrix(inputs, clusters)
        labels = np.argmin(distances, axis=1)
        if len(np.unique(labels)) <= 1:
            return 9999999.0
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                return davies_bouldin_score(inputs, labels)
    return wrapper


def xie_beni(inputs, labels):
    n = inputs.shape[0] # number of inputs
    m = inputs.shape[1] # number of attributes
    def wrapper(particle):
        d = int(particle.shape[0]/m) # number of clusters 
        clusters = np.reshape(particle, (d, m)) #fit each cluster in a row
        distances = distance_matrix(inputs, clusters)
        # 10**(-100) avoids division by 0
        distances = np.where(distances!=0, distances, 10**(-100))
        # Shape of distance matrix:(n x d)
        u = distances**2 / np.sum(distances**2, axis=1)[:, np.newaxis]
        u = 1.0 / u
        u =(u.T / np.sum(u, axis=1)).T
        num = np.sum(u**2 * distances**2)     
        den = n * min(distance.pdist(clusters))**2
        return num / den
    return wrapper


def square_sum(x):
    # -100 to 100
    return sum(np.square(x))	


def rosenbrock(x):
    # -100 to 100
    x2 = np.array(x[1:])
    x1 = np.array(x[:-1]) 
    return sum(100*(x2-x1**2)**2 + (x1-1)**2)


def schwefel_222(x):
    # -10 to 10
    absx = np.abs(x)
    return sum(absx) + np.prod(absx)


def quartic_noise(x):
    # -1.28 to 1.28
    indices = np.array(range(len(x)))
    return sum(indices*x**4) + random.random()


def rastrigin(x):
    # -100 to 100
    return sum(np.square(x) - 10*np.cos(2*math.pi*x) + 10)


def griewank(x):
    # -600 to 600
    term1 = 1/4000 * sum(np.square(x - 100))
    term2 = np.prod(np.cos(
               (x - 100) / np.sqrt(range(1,len(x)+1)) 
            ))
    return  term1 - term2 + 1


def ackley(x):
    #-32 to 32
    n = len(x)
    term1 = -20 * math.exp(-0.2 * np.sqrt(sum(x**2)/n))
    term2 = -math.exp(sum(np.cos(2*math.pi*x))/n)
    return term1 + term2 + 20 + math.e


def schwefel_226(x): 
    # -500 to 500
    absx = np.abs(x)
    const = 418.982887272433799807913601398
    return const*len(x) - sum(x*np.sin(np.sqrt(absx)))


def u(a, k, m, x_i):
    if   x_i >  a: 
        return k * (x_i-a)**m
    elif x_i < -a:
        return k * (-x_i-a)**m
    else:
        return 0    


def penalty_1(x):
    # -50 to 50
    k, m, a = 100, 4, 10
    n = len(x)
    PI = math.pi
    y = 1.25 + x/4
    term1 = 10*PI/n * math.sin(PI*y[0])**2
    term2 = sum((y[:-1]-1)**2 * (1+10*np.sin(PI*y[1:])**2))
    term3 =(y[-1]-1)**2
    term4 = sum([u(a, k, m, x_i) for x_i in x])
    return term1 + term2 + term3 + term4
    

def penalty_2(x):
    # -50 to 50
    k, m, a = 100, 4, 5
    PI = math.pi   
    term1 = 0.1 * math.sin(3*PI*x[0])**2
    term2 = sum((x-1)**2 * (1 + np.sin(3*PI*x+1)**2))
    term3 =(x[-1]-1)**2 * (1 + math.sin(2*PI*x[-1]))**2
    term4 = sum([u(a, k, m, x_i) for x_i in x])
    return term1 + term2 + term3 + term4


def get_function(function_name):
    if function_name == 'square_sum':
        l_bound, u_bound = -100.0, 100.0
        task = 'min'
        return square_sum, l_bound, u_bound, task
    elif function_name == 'rosenbrock':
        l_bound, u_bound = -100.0, 100.0
        task = 'min'
        return rosenbrock, l_bound, u_bound, task
    elif function_name == 'schwefel_222':
        l_bound, u_bound = -10.0, 10.0
        task = 'min'
        return schwefel_222, l_bound, u_bound, task
    elif function_name == 'quartic_noise':
        l_bound, u_bound = -1.28, 1.28
        task = 'min'
        return quartic_noise, l_bound, u_bound, task
    elif function_name == 'rastrigin':
        l_bound, u_bound = -100.0, 100.0
        task = 'min'
        return rastrigin, l_bound, u_bound, task
    elif function_name == 'griewank':
        l_bound, u_bound = -600.0, 600.0
        task = 'min'
        return griewank, l_bound, u_bound, task
    elif function_name == 'ackley':
        l_bound, u_bound = -32.0, 32.0
        task = 'min'
        return ackley, l_bound, u_bound, task
    elif function_name == 'schwefel_226':
        l_bound, u_bound = -500.0, 500.0
        task = 'min'
        return schwefel_226, l_bound, u_bound, task
    elif function_name == 'penalty_1':
        l_bound, u_bound = -50.0, 50.0
        task = 'min'
        return penalty_1, l_bound, u_bound, task
    elif function_name == 'penalty_2':
        l_bound, u_bound = -50.0, 50.0
        task = 'min'
        return penalty_2, l_bound, u_bound, task
    else:
        return None
    

def get_cluster_index(function_name, dataset_name):
    """
    Get a cluster evaluation function for a specific dataset.

    Parameters
    ----------
    function_name: string
    dataset_name: string

    Returns
    -------
    eval_func: 1d function
        Index for evaluating a clustering final partition.
    task: 'min' or 'max'
        Type of problem (minimization or maximization)
    """
    X, y = data_loader.load_dataset(dataset_name)
    if function_name == 'davies_bouldin':
        return davies_bouldin(X, y), 'min'
    else:
        return xie_beni(X, y), 'min'

    
if __name__ == '__main__':
    X, y = data_loader.load_dataset('iris.data')
    X = data_loader.norm_plus_minus_1(X)

    DB = davies_bouldin(X, y)
    XB = xie_beni(X, y)

    particle = np.random.uniform(-1.0, 1.0,(int(2*X.shape[1])))

    print(DB(particle))
    print(XB(particle))