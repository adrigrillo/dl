# %%
import keras
import numpy as np
from scipy.io import loadmat

from src.model.seq_to_seq import Seq2Seq
from src.utils.data_handling import process_data

# %% Load and plot the y

time_series: np.ndarray = loadmat('data/Xtrain.mat')['Xtrain']
data_points = time_series.shape[0]

# %%
x, y = process_data(time_series, window_size=3)

# %% encoder-decoder method
keras.backend.clear_session()
model = Seq2Seq()
model.fit(x, y, epochs=100, steps_per_epoch=100)
model.save_model('test')

# %%
prediction = model.predict(x[:800], 200)
