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
plt.show()