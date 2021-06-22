import torch
import torch.nn as nn
from torch.utils.data.dataloader import Dataset, DataLoader
from data import SequentialSEEDDataset, SequentialSEEDDataSetWithPrior
import numpy as np
import torch.nn.functional as F
import pandas as pd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

video_slices = [238, 233, 206, 238, 185, 195, 237, 216, 265, 237, 235, 233, 235, 238, 206]
data_path = "../hw2/data_hw2/"
input_size = 310
hidden_size = 256
num_layers = 1
seq_len = 256
batch_size = 32
lr = 1e-4
epoch = 1000
step = 30
window_size = 50
factor = 0.5
use_prior_knowledge = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data = np.load("../hw2/data_hw2/train_data.npy")
label = np.load("../hw2/data_hw2/train_label.npy")
test_data = np.load("../hw2/data_hw2/test_data.npy")
test_label = np.load("../hw2/data_hw2/test_label.npy")

label = label + 1
test_label = test_label + 1

train_val_split = int(0.8 * data.shape[0])
train_data = data
train_label = label


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bias=True,
                            batch_first=True,
                            bidirectional=False)
        self.linear = nn.Linear(hidden_size, 3)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = F.relu(lstm_out[:, -1, :])
        y_pred = self.linear(out)
        return y_pred

    def reset(self):
        self.lstm.reset_parameters()
        self.linear.reset_parameters()


if __name__ == '__main__':
    net = LSTM()
    if not use_prior_knowledge:
        trainset = SequentialSEEDDataset(train_data, train_label, step, seq_len)
        testset = SequentialSEEDDataset(test_data, test_label, step, seq_len)
    else:
        trainset = SequentialSEEDDataSetWithPrior(train_data, train_label)
        testset = SequentialSEEDDataSetWithPrior(test_data, test_label)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(net.parameters(), lr=lr)

    # training
    loss_history = []
    acc_history = []
    valacc_history = []
    for e in range(epoch):
        loss = 0
        for i, (feature, target) in enumerate(trainloader):
            optimizer.zero_grad()
            output = net(feature.to(device))
            err = criterion(output, target.to(device))
            err.backward()
            optimizer.step()
            loss += err.item()

        loss = loss / len(trainloader)
        loss_history.append(loss)

        # test
        with torch.no_grad():
            n_correct = 0
            n_total = 0
            for i, (feature, target) in enumerate(testloader):
                output = net(feature.to(device))
                pred = output.detach().cpu().argmax(dim=1, keepdim=False)
                n_correct += pred.eq(target.view_as(pred)).sum().item()
                n_total += len(pred)

            acc = n_correct / n_total
            acc_history.append(acc)

        print('Summary Epoch: %d, loss: %f, acc: %f' % (e, loss, acc))

    dict = {"loss": loss_history, "accuracy": acc_history}
    dataframe = pd.DataFrame(dict)
    dataframe.to_csv("result(with prior).csv")
