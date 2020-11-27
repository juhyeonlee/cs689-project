import numpy as np
import matplotlib.pyplot as plt
from load_data import load_data
from utils import detect_zero_crossings

# data_path = './data/'
# data = load_data(data_path)
# np.savez_compressed('data', **data)
data = np.load('data.npz', allow_pickle=True)
sub_id = data['sub_id']
sensor_data= data['sensor_data']

for idx, sub in enumerate(sub_id):
    sI, sF = detect_zero_crossings(sensor_data[idx][:, 0], margin=np.min(sensor_data[idx][:, 0]))
    print(np.min(sensor_data[idx][:, 0]))


    for s in range(len(sI)):
        plt.figure()
        plt.plot(sensor_data[idx][sI[s]:sF[s], 0])

        plt.show()
    plt.close()
