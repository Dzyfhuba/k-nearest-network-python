# =========================================================================
#
# Title     : K-Nearest Neighbor Algorithm
# Author    : Dzyfhuba
# GitHub    : https://github.com/Dzyfhuba/k-nearest-network-python.git
#
# =========================================================================

import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("TkAgg")

data_raw = pd.read_csv('dataset/creditApprovalUCI.csv')
data = data_raw[['A2', 'A3', 'A16']]
data_X, data_y = data.drop('A16', axis=1), data['A16']

fig, ax = plt.subplots()

ax.scatter(data[data['A16'] == 1]['A2'], data[data['A16'] == 1]['A3'], c='red', marker='+', label="+")
ax.scatter(data[data['A16'] == 0]['A2'], data[data['A16'] == 0]['A3'], c='blue', marker='o', label="-")
ax.set_xlabel('A2')
ax.set_ylabel('A3')

ax.set_title('Dataset Visualization')
ax.legend()

# show the plot
# plt.show()

# store the plot
plt.savefig('dataset/dataset_visualization.png')

# get train dan test data
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=21)

def delete_multiple_lines(n=1):
    """Delete the last line in the STDOUT."""
    for _ in range(n):
        sys.stdout.write("\x1b[1A")  # cursor up one line
        sys.stdout.write("\x1b[2K")  # delete the last line


def naive_euclidian_distance(point1, point2):
    differences = [point1[x] - point2[x] for x in range(len(point1))]
    differences_squared = [difference ** 2 for difference in differences]
    sum_of_squares = sum(differences_squared)
    return sum_of_squares ** 0.5

# start timer
start = time.time()

# knn classifier
def knn(x_train, y_train, x_test, actual=pd.DataFrame(), k=3, mode='test'):
    y_result = pd.DataFrame(columns=['y_pred'])
    score = 0


    if (x_train.shape == x_test.shape):
        raise Exception(f'The shape size is not same. x_train shape: {x_train.shape}, y_train: {y_train.shape}')
        return

    vertical_length_train, vertical_length_test = x_train.shape[0], x_test.shape[0]
    if (mode=='test'):
        for i in range(vertical_length_test):
            result_child = pd.DataFrame(columns=['distance', 'y'])
            for j in range(vertical_length_train):
                distance = naive_euclidian_distance(x_train.iloc[j], x_test.iloc[i])
                # add distance and y to result_child, dont use append, because is will deprecated
                result_child.loc[j] = [distance, y_train.iloc[j]]

                # print timer
                print(f'{i+1}/{vertical_length_test}')
                # clean print in terminal
                delete_multiple_lines(1)
            # get k lowest distance
            result_child = result_child.sort_values(by='distance')
            result_child = result_child.head(k)
            # get y_pred
            y_pred = result_child['y'].mode()[0]
            # add y_pred to result
            y_result.loc[i] = [y_pred]
            
        # get accuracy from actual
        if (actual.shape[0] == y_result.shape[0]):
            for i in range(vertical_length_test):
                if (actual.iloc[i] == y_result.iloc[i][0]):
                    score += 1
            score = score / vertical_length_test

        return y_result, score
    else:
        raise Exception(f'Mode {mode} is not supported')

# run knn with k = 1, 3, 5, 7
k_list = [1, 3, 5, 7]
for k in k_list:
    print(f'K: {k}')
    y_result, score = knn(X_train, y_train, X_test, y_test, k=k, mode='test')
    print(f'Score: {score}')
    print(f'Time: {time.time() - start}')

    # store result to csv
    y_result.to_csv(f'result/knn_k_{k}.csv', index=False)

    # plot result and store to png
    fig, ax = plt.subplots()
    ax.scatter(data[data['A16'] == 1]['A2'], data[data['A16'] == 1]['A3'], c='red', marker='+', label="+")
    ax.scatter(data[data['A16'] == 0]['A2'], data[data['A16'] == 0]['A3'], c='blue', marker='o', label="-")
    ax.set_xlabel('A2')
    ax.set_ylabel('A3')
    ax.set_title(f'Dataset Visualization with K = {k}')
    ax.legend()
    plt.savefig(f'result/knn_k_{k}.png')

    print('\n')
    os.system('play -nq -t alsa synth {} sine {}'.format(1, 1000))
    start = time.time()

# notify with double beep sound linux
os.system('play -nq -t alsa synth {} sine {}'.format(1, 1000))
