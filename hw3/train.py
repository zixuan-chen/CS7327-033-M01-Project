import random
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from data import SEEDDataSet
from util import plotLossAcc
from model import DANN
import numpy as np

model_root = "\\logs"
lr = 1e-4
batch_size_target_domain = 32
n_epoch = 50
_lambda = 0.5
alpha = 1
cuda = True
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)


def train_transfer(target_domain_id=2):
    domain = ['sub_0', 'sub_1', 'sub_2', 'sub_3', 'sub_4']
    print("Round %d, target_domain = %s" % (target_domain_id, domain[target_domain_id]))
    target_domain = [domain[target_domain_id]]
    source_domain = domain.copy()
    source_domain.pop(target_domain_id)

    dataset_target = SEEDDataSet(target_domain)
    dataset_source = SEEDDataSet(source_domain)

    batch_size_source_domain = int(len(dataset_source) / len(dataset_target) *
                                   batch_size_target_domain)

    target_loader = DataLoader(dataset=dataset_target,
                               batch_size=batch_size_target_domain,
                               shuffle=True)

    source_loader = DataLoader(dataset=dataset_source,
                               batch_size=batch_size_source_domain,
                               shuffle=True)

    net = DANN()

    optimizer = optim.Adam(net.parameters(), lr=lr)

    loss_class = torch.nn.CrossEntropyLoss()
    loss_domain = torch.nn.CrossEntropyLoss()

    net.to(device)

    loss_history = []
    acc_history = []
    acc_s_history = []
    for epoch in range(n_epoch):
        len_dataloader = min(len(source_loader), len(target_loader))
        source_loader_iter = iter(source_loader)
        target_loader_iter = iter(target_loader)
        loss = 0

        for i in range(len_dataloader):
            # p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            # _lambda = 2. / (1. + np.exp(-10 * p)) - 1

            # training model using source data
            s_feature, s_label = next(source_loader_iter)

            domain_label = torch.zeros(len(s_label)).long()

            net.zero_grad()

            s_feature.to(device)
            s_label.to(device)
            domain_label.to(device)

            class_output, domain_output = net(s_feature, _lambda)
            err_s_label = loss_class(class_output, s_label)
            err_s_domain = loss_domain(domain_output, domain_label)

            # training model using target data
            t_feature, _ = next(target_loader_iter)

            domain_label = torch.ones(len(t_feature)).long()

            t_feature.to(device)
            domain_label.to(device)

            _, domain_output = net(t_feature, _lambda)
            err_t_domain = loss_domain(domain_output, domain_label)
            err = err_s_label + alpha * (err_t_domain + err_s_domain)

            loss += err.item()
            err.backward()
            optimizer.step()

            # print('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f, _lambda =
            # %f' \ % (epoch, i, len_dataloader, err_s_label.cpu().data.numpy(), err_s_domain.cpu().data.numpy(),
            # err_t_domain.cpu().data.numpy(), _lambda))

            # torch.save(net, '.\\{0}\\DANN_epoch_{1}.pth'.format(model_root, epoch))

        loss = loss / len_dataloader
        loss_history.append(loss)
        # test model using target data
        n_total = 0
        n_correct = 0
        with torch.no_grad():
            for i, (feature, label) in enumerate(target_loader):
                feature.to(device)
                label.to(device)

                class_output, _ = net(feature, _lambda=0)

                pred = class_output.detach().argmax(dim=1, keepdim=False)
                n_correct += pred.eq(label.detach().view_as(pred)).cpu().sum()
                n_total += len(pred)

            acc = n_correct / n_total
            acc_history.append(acc)

        n_correct = 0
        n_total = 0
        with torch.no_grad():
            for i, (feature, label) in enumerate(source_loader):
                feature.to(device)
                label.to(device)

                class_output, _ = net(feature, _lambda=0)

                pred = class_output.detach().argmax(dim=1, keepdim=False)
                n_correct += pred.eq(label.detach().view_as(pred)).cpu().sum()
                n_total += len(pred)

            acc_s = n_correct / n_total
            acc_s_history.append(acc)
        print('Summary Epoch: %d, loss: %f, source accuracy: %f, target accuracy: %f'
              % (epoch, loss, acc_s, acc))
    return loss_history, acc_history


def train_normal(target_domain_id=2):
    domain = ['sub_0', 'sub_1', 'sub_2', 'sub_3', 'sub_4']
    print("Round %d, target_domain = %s" % (target_domain_id, domain[target_domain_id]))
    target_domain = [domain[target_domain_id]]
    source_domain = domain
    source_domain.pop(target_domain_id)

    dataset_target = SEEDDataSet(target_domain)
    dataset_source = SEEDDataSet(source_domain)

    batch_size_source_domain = int(len(dataset_source) / len(dataset_target) *
                                   batch_size_target_domain)

    target_loader = DataLoader(dataset=dataset_target,
                               batch_size=batch_size_target_domain,
                               shuffle=True)

    source_loader = DataLoader(dataset=dataset_source,
                               batch_size=batch_size_source_domain,
                               shuffle=True)

    net = nn.Sequential(nn.Linear(310, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, 3), )

    optimizer = optim.Adam(net.parameters(), lr=lr)

    loss_class = torch.nn.CrossEntropyLoss()
    loss_domain = torch.nn.CrossEntropyLoss()

    net.to(device)

    loss_history = []
    acc_history = []
    acc_s_history = []
    for epoch in range(n_epoch):

        loss = 0

        for i, (s_feature, s_label) in enumerate(source_loader):
            # training model using source data

            net.zero_grad()

            s_feature.to(device)
            s_label.to(device)

            class_output = net(s_feature)
            err = loss_class(class_output, s_label)
            loss += err.item()
            err.backward()
            optimizer.step()

            # print('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f, _lambda =
            # %f' \ % (epoch, i, len_dataloader, err_s_label.cpu().data.numpy(), err_s_domain.cpu().data.numpy(),
            # err_t_domain.cpu().data.numpy(), _lambda))

            # torch.save(net, '.\\{0}\\DANN_epoch_{1}.pth'.format(model_root, epoch))

        loss = loss / len(source_loader)
        loss_history.append(loss)
        # test model using target data
        n_total = 0
        n_correct = 0
        with torch.no_grad():
            for i, (feature, label) in enumerate(target_loader):
                feature.to(device)
                label.to(device)

                class_output = net(feature)

                pred = class_output.detach().argmax(dim=1, keepdim=False)
                n_correct += pred.eq(label.detach().view_as(pred)).cpu().sum()
                n_total += len(pred)

            acc = n_correct / n_total
            acc_history.append(acc)

        n_correct = 0
        n_total = 0
        with torch.no_grad():
            for i, (feature, label) in enumerate(source_loader):
                feature.to(device)
                label.to(device)

                class_output = net(feature)

                pred = class_output.detach().argmax(dim=1, keepdim=False)
                n_correct += pred.eq(label.detach().view_as(pred)).cpu().sum()
                n_total += len(pred)

            acc_s = n_correct / n_total
            acc_s_history.append(acc)
        print('Summary Epoch: %d, loss: %f, source accuracy: %f, target accuracy: %f'
              % (epoch, loss, acc_s, acc))
    return loss_history, acc_history


if __name__ == '__main__':
    loss_summary = []
    acc_summary = []
    loss_base_summary = []
    acc_base_summary = []
    for subject_id in range(5):
        loss, acc = train_transfer(subject_id)

        loss_base, acc_base = train_normal(subject_id)

        plotLossAcc(loss, acc, loss_base, acc_base,
                    name="loss_acc_target_source=sub_%d.png" % subject_id)

        loss_summary.append(loss)
        acc_summary.append(acc)
        loss_base_summary.append(loss_base)
        acc_base_summary.append(acc_base)

    # plot summary loss and accuracy
    loss_mean = np.mean(np.array(loss_summary), axis=0)
    acc_mean = np.mean(np.array(acc_summary), axis=0)
    loss_base_mean = np.mean(np.array(loss_base_summary), axis=0)
    acc_base_mean = np.mean(np.array(acc_base_summary), axis=0)

    plotLossAcc(loss_mean, acc_mean, loss_base_mean, acc_base_mean,
                name="loss_acc_summary.png")

    # record the best performance of DANN and DNN
    acc_max = np.max(np.array(acc_summary), axis=1)
    acc_base_max = np.max(np.array(acc_base_summary), axis=1)
    acc_mean_max = np.max(acc_mean)
    acc_base_mean_max = np.max(acc_base_mean)

    with open("result.txt", "w") as f:
        f.write("=====================Best Performance=======================\n")
        for i in range(5):
            f.write("target_source = sub_%d, DANN: %f, DNN: %f\n" % (i, acc_max[i], acc_base_max[i]))

        f.write("=====================Summary=================================\n")
        f.write("DANN: %f, DNN: %f\n" % (acc_mean_max, acc_base_mean_max))
