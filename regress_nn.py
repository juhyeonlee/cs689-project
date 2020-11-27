import numpy as np
import torch
import torch.nn as nn
from cnn import NN
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, accuracy_score

data = np.load('data.npz', allow_pickle=True)
sub_id = data['sub_id']
sensor_data= data['sensor_data']
scores = data['updrs']
labels = data['sub_label']
# test only PD patients
test_idx = data['sub_label'].astype(np.bool)
print(np.sum(test_idx))

fold_length = 13
fold_num = int(np.sum(test_idx) / fold_length)
print(fold_length, fold_num)

win_length = 300

# scores_max = np.max(scores)
# print(scores_max)
# scores /= scores_max

seg_sensor_data = []
seg_scores = []
seg_labels = []
seg_subid = []
seg_sensor_next_data = []
for i, sub in enumerate(sub_id):
    print(len(sensor_data[i]) // win_length, len(sensor_data[i]))
    for w in range(len(sensor_data[i]) // win_length):
        seg_sensor_data.append(sensor_data[i][w*win_length:(w+1)*win_length, :])
        # seg_sensor_next_data.append(sensor_data[i][(w+1)*win_length:(w+1)*win_length+16, :])
        seg_scores.append(scores[i])
        seg_labels.append(labels[i])
        seg_subid.append(sub)


seg_sensor_data = np.array(seg_sensor_data)
seg_sensor_next_data = np.array(seg_sensor_next_data)
seg_scores = np.array(seg_scores)
seg_labels = np.array(seg_labels)
seg_subid = np.array(seg_subid)
print(seg_sensor_data.shape)
print(seg_sensor_next_data.shape)

# for i in range(fold_num):
#
#     test_mask = np.argwhere(test_idx)[i * fold_length: (i + 1) * fold_length].reshape(-1)
#     train_mask = np.delete(np.arange(len(test_idx)), test_mask)
#     print('checksum', np.sum(np.logical_and(test_mask, train_mask)))
#
#     train_data = seg_sensor_data[np.isin(seg_subid, sub_id[train_mask]), :, :]
#     train_scores = seg_scores[np.isin(seg_subid, sub_id[train_mask])]
#     train_labels = seg_labels[np.isin(seg_subid, sub_id[train_mask])]
#     # train_next_data = seg_sensor_next_data[np.isin(seg_subid, sub_id[train_mask]), :, :]
#     test_data = seg_sensor_data[np.isin(seg_subid, sub_id[test_mask]), :, :]
#     test_scores = seg_scores[np.isin(seg_subid, sub_id[test_mask])]
predictions = []
for i, sub in enumerate(sub_id):
    train_data = seg_sensor_data[~np.isin(seg_subid, sub), :, :]
    train_scores = seg_scores[~np.isin(seg_subid, sub)]
    train_labels = seg_labels[~np.isin(seg_subid, sub)]
    # train_next_data = seg_sensor_next_data[np.isin(seg_subid, sub_id[train_mask]), :, :]
    test_data = seg_sensor_data[np.isin(seg_subid, sub), :, :]
    test_scores = seg_scores[np.isin(seg_subid, sub)]
    test_labels = seg_labels[np.isin(seg_subid, sub)]


    nn = NN()

    nn.fit(np.transpose(train_data, (0, 2, 1)), train_scores, train_labels, step=1e-3, epochs=2)
    y_pred_train = nn.predict(np.transpose(train_data, (0, 2, 1)))
    y_pred_test = nn.predict(np.transpose(test_data, (0, 2, 1)))
    y_pred_train = np.argmax(y_pred_train, axis=1)
    y_pred_test = np.argmax(y_pred_test, axis=1)
    predictions.append(np.mean(y_pred_test))
    print(np.mean(np.abs(y_pred_train - train_labels)))

    # print(train_scores.shape, y_pred_train.shape)
    # print(np.mean(np.abs(train_scores - y_pred_train.reshape(-1))))
    # print(np.abs(test_scores - y_pred_test.reshape(-1)))
    print(np.mean(np.abs(test_labels - y_pred_test)))
    print(test_scores, y_pred_test)
    # print(np.mean(test_scores), np.mean(y_pred_test))
    print('!!!!!!!!!!!!!!!1')
predictions = np.array(predictions)
np.save('results3.npy', predictions)

r2 = r2_score(scores[test_idx], predictions)
rmse = np.sqrt(np.mean(np.square(predictions - scores[test_idx])))
nrmse = rmse / (scores[test_idx].max() - scores[test_idx].min()) * 100
print(r2, nrmse)
fig_reg = plt.figure(figsize=[6, 6])
ax1 = fig_reg.add_subplot(1, 1, 1)
max_score = max(scores[test_idx].max(), predictions.max())
sc = ax1.scatter(scores[test_idx], predictions) #, c=y_hat_std, cmap='viridis')
# ax1.scatter(y[dig==3], y_hat[dig==3], c='g')
# sc = ax1.scatter(y, y_hat, c=age_y, cmap='viridis')
fig_reg.colorbar(sc)
ax1.plot([0, max_score], [0, max_score], 'r')
# ax1.set_title('BARS - {}'.format(predict_mode))
ax1.set_xlabel('Clinician-Scored UPDRS')
ax1.set_ylabel('Estimated UPDRS')
ax1.axis('square')
plt.tight_layout()
plt.show()