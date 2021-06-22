from matplotlib import pyplot as plt
import os


def plotLossAcc(loss, acc, loss_base, acc_base, name=None):
    save_dir = "imgs"
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


def plotLossAccSingle(loss, acc, valacc=None, save=False, smooth=False, **kwargs):
    assert len(loss) == len(acc)

    x = list(range(1, len(loss) + 1))
    fig = plt.figure()
    ax0 = fig.add_subplot(1, 2, 1)
    ax0.plot(x, loss, label="loss", color='blue')
    ax0.set_title("Loss")
    ax0.legend()
    ax1 = fig.add_subplot(1, 2, 2)
    if smooth:
        smooth_acc = smooth1d(acc, kwargs['window_size'], kwargs['factor'])
        ax1.plot(x, acc, label="test_acc", color='pink')
        ax1.plot(x, smooth_acc, label="test_acc(smoothed)", color="red", linewidth=2)
    else:
        ax1.plot(x, acc, label="test_acc", color='red')
    if valacc:
        assert len(loss) == len(valacc)
        ax1.plot(x, valacc, label="validation", color='purple')
    ax1.set_title("Accuracy")
    ax1.legend()
    if save:
        plt.savefig(os.path.join(kwargs['save_dir'], kwargs['save_title']))
    plt.show()


def smooth1d(x, window_size, factor=0.5):
    assert len(x) > window_size, "Error in smoothing, sliding window too large!"
    length = len(x)
    cur_size = 0
    left = 0
    right = window_size // 2
    cur_sum = 0
    smooth_x = []
    for i in range(left, right):
        cur_sum += x[i]
        cur_size += 1

    for i in range(length):
        if right < length:
            cur_sum += x[right]
            cur_size += 1
            right += 1
        if cur_size > window_size or right == length:
            cur_sum -= x[left]
            cur_size -= 1
            left += 1

        new_xi = factor * cur_sum / cur_size + (1 - factor) * x[i]
        smooth_x.append(new_xi)

    return smooth_x


if __name__ == "__main__":
    x = list(range(10))
    print(smooth1d(x, 5, 1))