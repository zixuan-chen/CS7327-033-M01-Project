import torch
import numpy as np
from matplotlib import pyplot as plt

path = "two-spiral traing data(update).txt"


def plot_data(data, ax, lw=0.5, clr0='b', clr1='r'):
    ax.set_facecolor('darkgray')
    idx1 = np.where(data[:, 2] == 1)[0]
    ax.scatter(data[idx1, 0], data[idx1, 1], c=clr1, linewidths=lw)
    idx0 = np.where(data[:, 2] == 0)[0]
    ax.scatter(data[idx0, 0], data[idx0, 1], c=clr0, linewidths=lw)


def plot_decision_boundary(net, ax, lw=0.1):
    ax.set_facecolor('darkgray')
    indexs = np.linspace(-6.5, 6.5, num=100).astype(float)
    X, Y = np.meshgrid(indexs, indexs)
    input_ = np.dstack((X, Y)).reshape(-1, 2)
    input_ = torch.tensor(input_, dtype=torch.float, requires_grad=False)
    out_ = net(input_).detach().numpy()

    x, y = X.flatten(), Y.flatten()
    idx0 = np.where(out_ > 0)[0]
    ax.scatter(x[idx0], y[idx0], c='w', linewidths=lw)
    idx1 = np.where(out_ < 0)[0]
    ax.scatter(x[idx1], y[idx1], c='k', linewidths=lw)


def load_data_with_partition(partition=False):
    """
    devide the origin problem into n*n sub-problems
    """
    lines = open(path).readlines()
    n = len(lines)
    data = np.zeros((n, 3), dtype=np.float)
    for i in range(n):
        data[i] = np.array([float(j) for j in lines[i].split()])

    if not partition:
        return torch.tensor(data, dtype=torch.float)

    idx0 = np.where(data[:, 2] == 0)[0]
    idx1 = np.where(data[:, 2] == 1)[0]

    np.random.shuffle(idx0)
    np.random.shuffle(idx1)

    # step0, step1 = len(idx0) // npart, len(idx1) // npart
    #
    # data_part = []
    # for i in range(0, len(idx0)-step0, step0):
    #     for j in range(0, len(idx1)-step1, step1):
    #         idx = np.append(idx0[i: i + step0], idx1[j: j + step1])
    #         sub_data = data[idx]
    #         np.random.shuffle(sub_data)
    #         data_part.append(sub_data)
    mid0, mid1 = len(idx0) // 2, len(idx1) // 2

    data_part0 = data[idx0[0: mid0]]
    data_part1 = data[idx0[mid0 + 1:]]
    data_part2 = data[idx1[0: mid1]]
    data_part3 = data[idx1[mid1 + 1:]]
    data_part = [data_part0, data_part1, data_part2, data_part3]

    sub_data0 = np.append(data_part0, data_part2, axis=0)
    sub_data1 = np.append(data_part0, data_part3, axis=0)
    sub_data2 = np.append(data_part1, data_part2, axis=0)
    sub_data3 = np.append(data_part1, data_part3, axis=0)
    sub_data = [sub_data0, sub_data1, sub_data2, sub_data3]

    return data_part, sub_data


if __name__ == "__main__":
    # data_part, sub_data = load_data_with_partition(True, 2)
    # fig1 = plt.figure()
    # fig2 = plt.figure()
    # for i in range(len(data_part)):
    #     ax = fig1.add_subplot(2, 2, i + 1)
    #     plot_data(data_part[i], ax)
    #
    # for i in range(len(sub_data)):
    #     ax = fig2.add_subplot(2, 2, i + 1)
    #     plot_data(sub_data[i], ax)

    plt.show()
