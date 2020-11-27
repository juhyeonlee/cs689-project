import numpy as np
import pickle
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt
from GaussianHMM import GaussianHMM
# from hmmlearn.hmm import GaussianHMM
# from sktime.transformers.summarise import PlateauFinder
# import pandas as pd
from load_data import load_data

datapath = './data/'
data = load_data(datapath)


rgzn_modes = GaussianHMM.RgznModes()
zeta = 3.
color_set = ['r', 'g', 'b']
axis_set = ["X", "Y", "Z"]

predicted_states_set = []
predicted_states_color_set = []
sub_id = data['sub_id']
for s, sub in enumerate(sub_id):

    train_set = data['sensor_data'][s]
    print(train_set.shape)
    K = 2
    model1 = GaussianHMM(K, train_set, rgzn_modes.INERTIAL)
    model1.learn(train_set, zeta=zeta)
    predicted_states = model1.decode(train_set[0])
    predicted_states_n = np.asarray(predicted_states)
    predicted_states_set.append(predicted_states_n)
    color = np.zeros(predicted_states_n.size, dtype='str')
    for i in range(K):
        color[np.where(predicted_states_n == float(i))] = color_set[i]
    predicted_states_color_set.append(color)
    fig = plt.figure(figsize=(9, 8))
    plt.plot(train_set[0])
    plt.scatter(np.arange((train_set.shape[0])), train_set[0], c=color,  marker=".")
    plt.show()
