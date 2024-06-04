import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PARENT_DIR = os.path.dirname(os.getcwd())


def plot_cifar10_results():
    filename = os.path.join(PARENT_DIR, 'results', 'cifar10_hp_search.csv')
    df = pd.read_csv(filename)
    experiments = np.unique(df['experiment'].values).tolist()

    dfs = {}
    for experiment in experiments:
        dfs[experiment] = (df.loc[df['experiment'] == experiment])

    column_names = [
        'cohort_size', 'local_batch_size', 'local_learning_rate',
        'local_num_epochs'
    ]
    unique_vals = {}
    for column_name in column_names:
        unique_vals[column_name] = np.unique(dfs['live'][column_name]).tolist()

    accs = {}
    for name, df in dfs.items():
        accs[name] = {}
        for tup in product(*unique_vals.values()):
            filter_dic = dict(zip(column_names, tup))
            a = df.loc[(df[list(filter_dic)] == pd.Series(filter_dic)).all(
                axis=1)]['Central val | accuracy (avg)'].values.mean()
            accs[name][tup] = a

    x = np.array(list(accs['live'].values()))
    permutation = np.argsort(-x)
    mask = np.array(list(accs['live'].values()))[permutation] >= 0.6

    dic = {}
    c = dict(zip(accs.keys(), ['blue', 'red', 'green']))

    plt.rcParams.update({'font.size': 13})
    for name, d in accs.items():
        x = np.array(list(d.values()))[permutation][mask]
        dic[name] = x

        plt.plot(x, label=name, c=c[name])

    plt.xlabel('Random hyperparameter setting')
    plt.ylabel('Classification accuracy (%)')
    plt.legend()
    plt.show()

    print(
        np.mean(
            np.abs(
                np.array(dic['live']) - np.array(dic['simulated_dirichlet']))))
    print(
        np.mean(
            np.abs(np.array(dic['live']) -
                   np.array(dic['simulated_uniform']))))


def main():
    plot_cifar10_results()


if __name__ == '__main__':
    main()
