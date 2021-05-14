from matplotlib import pyplot as plt
import os

save_dir = "imgs"


def plotLossAcc(loss, acc, loss_base, acc_base, name=None):
    n = max(len(loss), len(loss_base))
    x = list(range(1, n + 1))
    fig = plt.figure()
    ax0 = fig.add_subplot(1, 2, 1)
    ax0.plot(x, loss, label="ADNN", color='red')
    ax0.plot(x, loss_base, label="baseline", color='blue')
    ax0.set_title("Loss")
    ax0.legend()
    ax1 = fig.add_subplot(1, 2, 2)
    ax1.plot(x, acc, label="ADNN", color='red')
    ax1.plot(x, acc_base, label='DNN', color='blue')
    ax1.set_title("Accuracy")
    ax1.legend()
    if name is not None:
        plt.savefig(os.path.join(save_dir, name))
    plt.show()
