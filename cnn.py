import numpy as np
import torch
import time
# import matplotlib.pyplot as plt

np.random.seed(1)
torch.manual_seed(1)

class NN:
    """A neural network model for timeseries forecasting

    Arguments:
        None (add what you need)
    """
    def __init__(self):
        self.conv1 = torch.nn.Conv1d(16, 16, kernel_size=100, stride=5, padding=1)
        self.conv2 = torch.nn.Conv1d(16, 16, kernel_size=30, stride=1)
        self.linear = torch.nn.Linear(192, 1)
        self.linear2 = torch.nn.Linear(192, 2)
        self.dropout = torch.nn.Dropout(p=0.1)
        self.dropout2 = torch.nn.Dropout(p=0.4)
        self.relu = torch.nn.ReLU()
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        
    def objective(self, X, y):
        """Compute objective function.

        Arguments:
            X (numpy ndarray, shape = (samples, 300)):
                Observed portion of timeseries for each data case.
            y  (numpy ndarray, shape = (samples,100)):
                Portion of time series to predict for each data case

        Returns:
            float: Mean squared error objective value. Mean is taken
            over all dimensions of all data cases.
        """
        inputs = torch.Tensor(X)
        targets = torch.Tensor(y)

        h = self.conv1(inputs)
        h = self.relu(h)
        h = self.conv2(h)
        h = self.relu(h)
        h = torch.flatten(h, start_dim=1)
        y_pred = self.linear(h)
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(targets, y_pred)
        return loss.item()

    def predict(self, X):
        """Forecast time series values.

        Arguments:
            X (numpy ndarray, shape = (samples, 300)):
                Observed portion of timeseries for each data case.

        Returns:
            y  (numpy ndarray, shape = (samples,100)):
                Predicted portion of time series for each data case.
        """
        inputs = torch.Tensor(X)
        h = self.conv1(inputs)
        h = self.relu(h)
        h = self.conv2(h)
        h = self.relu(h)
        h = torch.flatten(h, start_dim=1)
        # y_pred = self.linear(h)
        y_next = self.linear2(h)
        y_next = self.dropout(y_next)
        y_pred = self.logsoftmax(y_next)
        return y_pred.detach().numpy()
        
    def fit(self, X, y, y_binary, step=1e-3, epochs=400):
        """Train the model using the given training data.

        Arguments:
            X (numpy ndarray, shape = (samples, 300)):
                Observed portion of timeseries for each data case
            y  (numpy ndarray, shape = (samples,100)):
                Portion of time series to predict for each data case
            step (float):
                step size to use
            epochs (int):
                number of epochs of training
        """
        inputs = torch.Tensor(X)
        targets_binary = torch.LongTensor(y_binary)
        targets = torch.Tensor(y)
        loss_fn = torch.nn.MSELoss()
        cross_loss_fn = torch.nn.NLLLoss()
        indexes = np.arange(X.shape[0])
        batch_size = 512
        optimizer = torch.optim.Adam([{'params': self.conv1.parameters()},
                                      {'params': self.conv2.parameters()},
                                      {'params': self.linear.parameters()},
                                      {'params': self.linear2.parameters()}], lr=step, weight_decay=1e-4)
        for e in range(epochs):
            np.random.shuffle(indexes)
            for t in range(X.shape[0] // batch_size):
                optimizer.zero_grad()
                h = self.conv1(inputs[indexes[t * batch_size:(t + 1) * batch_size], :, :])
                h = self.relu(h)
                h = self.dropout2(h)
                h = self.conv2(h)
                h = self.relu(h)
                h = torch.flatten(h, start_dim=1)
                h = self.dropout(h)
                y_pred = self.linear(h)
                y_pred = self.dropout(y_pred)
                y_next = self.linear2(h)
                y_next = self.dropout(y_next)
                y_next = self.logsoftmax(y_next)
                loss = cross_loss_fn(y_next, targets_binary[indexes[t * batch_size:(t + 1) * batch_size]])
                #loss_fn(y_pred, targets[indexes[t * batch_size:(t + 1) * batch_size]]) + \

                # 0.5 * loss_fn(next_inputs[indexes[t * batch_size:(t + 1) * batch_size], :, :], y_next.reshape(batch_size, -1, 16))
                # print(targets[indexes[t * batch_size:(t + 1) * batch_size]], y_pred)
                print(loss.item())
                loss.backward()
                optimizer.step()

# def main():
#
#     start_t = time.time()
#     DATA_DIR = '../data'
#
#     data = np.load("../data/data_distribute.npz")
#
#     #Training Datat
#     X_tr=data['X_tr']
#     Y_tr=data['Y_tr']
#
#     #Validation Datat
#     X_val=data['X_val']
#     Y_val=data['Y_val']
#
#     #Test data
#     #Note: test outputs are not provided in the data set
#     X_te=data['X_te']
#
#     nn = NN()
#     #Try computing objective function
#     print("Obj:", nn.objective(X_tr, Y_tr))
#
#     #Try predicting
#     Y_tr_hat = nn.predict(X_tr)
#
#     #Try fitting
#     nn.fit(X_tr, Y_tr, step=1e-3, epochs=100)
#     print("Train Obj:", nn.objective(X_tr, Y_tr))
#     print("Val Obj:", nn.objective(X_val, Y_val))
#     end_t = time.time()
#     print(end_t - start_t)
#
#     # val_pred5 = nn.predict(X_val[:5])
#     # fig = plt.figure(figsize=(30, 6))
#     # for ii in range(5):
#     #     ax = fig.add_subplot(1, 5, ii+1)
#     #     ax.plot(Y_val[ii], label='Ground Truth')
#     #     ax.plot(val_pred5[ii], label='Prediction')
#     #     ax.set_xlabel('time')
#     #     ax.set_ylabel('ECG traces')
#     #     plt.legend()
#     # plt.show()
#     #
#     # Try saving predictions
#     pred = nn.predict(X_te)
#     np.save("predictions.npy", pred)
