import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam


def crf_train_loop(model, rolls, targets, n_epochs, learning_rate=0.01):
    optimizer = Adam(model.parameters(), lr=learning_rate,
                     weight_decay=1e-4)

    for epoch in range(n_epochs):
        batch_loss = []
        N = rolls.shape[0]
        model.zero_grad()
        for index, (roll, labels) in enumerate(zip(rolls, targets)):
            # Forward Pass
            model.update_log_likelihood()
            neg_log_likelihood = model.neg_log_likelihood(roll, labels)
            batch_loss.append(neg_log_likelihood)

            if index % 50 == 0:
                ll = torch.cat(batch_loss).mean()
                ll.backward()
                optimizer.step()
                print("Epoch {}: Batch {}/{} loss is {:.4f}".format(epoch, index // 50, N // 50, ll.data.numpy()))
                batch_loss = []


    return model

def detect_zero_crossings(v, margin=0.0):
    zc = [0, v.shape[0]-1]
    for i in range(v.shape[0]-1):
        if v[i] is not np.nan and v[i+1] is not np.nan:
            if (v[i] <= margin and v[i+1] > margin) or (v[i] > margin and v[i+1] <= margin): #np.sign(v[i]) != np.sign(v[i+1]):
                zc.append(i+1)
    zc = np.unique(zc)
    sI, sF = zc[:-1], zc[1:]
    return sI, sF