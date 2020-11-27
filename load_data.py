import numpy as np


def load_data(datapath):
    sub_id = np.genfromtxt(datapath+'demographics.csv', delimiter=',', skip_header=1, usecols=(0,), dtype=str)
    sub_label_str = np.genfromtxt(datapath+'demographics.csv', delimiter=',', skip_header=1, usecols=(2,), dtype=str)
    updrs = np.genfromtxt(datapath+'demographics.csv', delimiter=',', skip_header=1, usecols=(10,), dtype=float)

    sub_label = [1 if ll == "PD" else 0 for ll in sub_label_str]
    sub_label = np.asarray(sub_label)
    print('# of subjects before excluding', len(sub_id))

    # exclude 2 subjects in PD cuz updrs is not available, and exclude JuC010 cuz sensor data is not available
    exclude_idx = np.logical_or(np.logical_and(sub_label == 1, np.isnan(updrs)), sub_id == 'Juc010')
    sub_id = sub_id[~exclude_idx]
    sub_label_str = sub_label_str[~exclude_idx]
    sub_label = sub_label[~exclude_idx]
    updrs = updrs[~exclude_idx]
    print('# of subjects after excluding', len(sub_id))

    # put 0 to control subjects' updrs score
    updrs[np.isnan(updrs)] = 0.
    # updrs_max = np.max(updrs)
    # print(updrs_max)
    # updrs /= updrs_max


    sensor_data = []
    time_stamp = []
    total_data = []
    cnt_n = 0.
    for sub in sub_id:
        # only loading normal walking data
        filename = datapath + sub + '_01.txt'
        time_data = np.genfromtxt(filename, usecols=(0,), dtype=float)
        time_stamp.append(time_data)
        force_data = np.genfromtxt(filename, usecols=(i for i in range(1, 17)), dtype=float)
        sensor_data.append(force_data)
        total_data.extend(force_data)
    sensor_avg = np.mean(total_data, axis=0)
    sensor_std = np.std(total_data, axis=0)
    print(sensor_avg, sensor_std)

    sensor_data_norm = []
    for dp in sensor_data:
        sensor_data_norm.append((dp - sensor_avg) / sensor_std)

    time_stamp = np.asarray(time_stamp)
    sensor_data = np.asarray(sensor_data)
    sensor_data_norm = np.asarray(sensor_data_norm)

    load_data = {'sub_id': sub_id, 'sub_label_str': sub_label_str,
                 'sub_label': sub_label, 'updrs': updrs,
                 'time_stamp': time_stamp, 'sensor_data': sensor_data_norm}

    return load_data
