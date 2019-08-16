import pandas as pd
import os

path_to_read = os.path.join(os.path.dirname('__file__'), 'exp_tests')
path_to_save = os.path.join(os.path.dirname('__file__'), 'exp_results_print')

for f in os.listdir(path_to_read):
    df_results = pd.read_csv(os.path.join(path_to_read, f), delimiter=',')
    df_filter = df_results[df_results['max_iters']!=200]
    df_filter.to_csv(os.path.join(path_to_save, f), index=False)

