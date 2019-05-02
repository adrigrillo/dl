# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

# %% Load and plot the y
from utils import process_data

time_series: np.ndarray = loadmat('data/Xtrain.mat')['Xtrain'].squeeze()
data_points = time_series.shape[0]

plt.figure(1, figsize=[15, 5])
plt.plot(time_series)
plt.xlabel('x')
plt.xlabel('y')
plt.title('Time series data')
plt.show()

x, y = process_data(time_series, 3)
