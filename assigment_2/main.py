# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

# %% Load and plot the y
data: np.ndarray = loadmat('data/Xtrain.mat')['Xtrain']
data_points = data.shape[0]

plt.figure(1, figsize=[15, 5])
plt.plot(data)
plt.xlabel('x')
plt.xlabel('y')
plt.title('Time series data')
plt.show()
