import pandas as pd
import os

def add_to_df(file_name, df):
    #print(file_name)    
    df_file = pd.read_csv(file_name)
    print(file_name)
    print(len(df_file))
    return df.append(df_file)

dot = os.path.dirname(__file__)

dir3 = '/home/gustavo/Dropbox/Doutorado/Codigos/HybridPsoGa/exp_results/'
dir2 = '/home/gustavo/Dropbox/Doutorado/Codigos/HybridPsoGa/exp_results_backup_2/'
dir1 = '/home/gustavo/Dropbox/Doutorado/Codigos/HybridPsoGa/exp_results_backup/'

list3 = os.listdir(dir3)
list2 = os.listdir(dir2)
list1 = os.listdir(dir1)

for l in list3:
    merge=False
    df = pd.DataFrame()
    if l in list2:
        df = add_to_df(os.path.join(dot, dir2+l), df)
        list2.remove(l)
        merge=True
    if l in list1:
        df = add_to_df(os.path.join(dot, dir1+l), df)
        list1.remove(l)
        merge=True
    if merge:
        df = add_to_df(os.path.join(dot, dir3+l), df)
        df.to_csv('exp_tests/'+l, index=False)
        print('==========================================')

for l in list2:
    merge=False
    df = pd.DataFrame()
    if l in list3:
        df = add_to_df(os.path.join(dot, dir3+l), df)
        list3.remove(l)
        merge=True
    if l in list1:
        df = add_to_df(os.path.join(dot, dir1+l), df)
        list1.remove(l)
        merge=True
    if merge:
        df = add_to_df(os.path.join(dot, dir2+l), df)
        df.to_csv('exp_tests/'+l, index=False)
        print('==========================================')
for l in list1:
    merge=False
    df = pd.DataFrame()
    if l in list2:
        df = add_to_df(os.path.join(dot, dir2+l), df)
        list2.remove(l)
        merge=True
    if l in list3:
        df = add_to_df(os.path.join(dot, dir3+l), df)
        list3.remove(l)
        merge=True
    if merge:
        df = add_to_df(os.path.join(dot, dir1+l), df)
        df.to_csv('exp_tests/'+l, index=False)
        print('==========================================')



#folder = pd.DataFrame('exp_results_2')