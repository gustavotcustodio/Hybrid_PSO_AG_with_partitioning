import numpy as np
import json

def read_json (file_name):
    with open(file_name) as json_file:
        params = json.load (json_file)
    return params


def run_experiments ():
    params = read_json ('parameters.json')

    '''Lists the experiments to be executed.
    The tuple determines the algorithm to be executed and the index of params.
    The algorithm and index (starting 0) are determined by the parameters.json file.
    Example: ('pso', 0)
        It will run the pso algorithm with the values of first positions in lists.'''
    experiments = [('hybrid', 0), ('pso', 0), ('hybrid', 1), ('pso', 1), 
                   ('hybrid', 2), ('pso', 2), ('hybrid', 3), ('pso', 3),
                   ('hybrid', 4), ('pso', 4)]

    #for i in range (len(params)):
        #params['pso']['pop_sizes'][0]


if __name__ == '__main__':
    try:
        run_experiments ()

    except:
        # File not found exception
        print ('Erro durante a execução dos experimentos')

        # File not found exception
        print ('Erro durante a execução dos experimentos')
