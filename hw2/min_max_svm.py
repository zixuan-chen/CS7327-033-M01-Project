import sklearn
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.preprocessing import StandardScaler

dim = 8
data = np.load('data_hw2/train_data.npy')
label = np.load("data_hw2/train_label.npy")
c = np.arange(len(data))
np.random.shuffle(c)
data, label = data[c], label[c]
n = data.shape[0]
split = int(0.8 * n)
train_data = data[:split]
train_label = label[:split]
val_data = data[split:]
val_label = label[split:]
test_data = np.load("data_hw2/test_data.npy")
test_label = np.load("data_hw2/test_label.npy")


class Min_Max_SVM:
    def __init__(self):
        clf = make_pipeline(StandardScaler(),
                            SVC(kernel='linear'))
        self.clf = [clf] * dim

    def decision_function(self, x):
        outputs = np.zeros((dim, len(x)))
        for i in range(dim):
            outputs[i] = self.clf[i].decision_function(x)
        min1 = np.min(outputs[[0, 2, 4, 6], :], axis=0)
        min2 = np.min(outputs[[1, 3, 5, 7], :], axis=0)
        max_ = np.max(np.vstack((min1, min2)), axis=0)
        return max_

    def predict(self, x):
        return self.decision_function(x) > 0


def train_single(net, category, pos_index, neg_index):
    # prepare training data
    indexs = np.append(pos_index, neg_index)
    y = np.zeros(len(indexs))
    y[:len(pos_index)] = 1
    X = train_data[indexs]
    c = np.arange(len(indexs))
    np.random.shuffle(c)
    X, y = X[c], y[c]
    # fit
    net.fit(X, y)
    # validate
    val_label_new = np.where(val_label == category, 1, 0)
    acc = net.score(val_data, val_label_new)
    print("validation accuracy: %f" % acc)


def train(category, method='random'):
    net = Min_Max_SVM()
    index_positive = np.argwhere(train_label == category).reshape(-1)
    index_negative = np.argwhere(train_label != category).reshape(-1)

    # split into 2-class subproblems
    n_pos, n_nega = len(index_positive), len(index_negative)
    pindx = [index_positive[:n_pos // 2], index_positive[n_pos // 2:]]
    if method == 'random':
        nindx = [index_negative[: n_nega // 4], index_negative[n_nega // 4: 2 * n_nega // 4],
                 index_negative[2 * n_nega // 4: 3 * n_nega // 4], index_negative[3 * n_nega // 4:]]
    else:  # with prior knowledge
        negacls = [-1, 0, 1]
        negacls.remove(category)
        index_class0 = np.argwhere(train_label == negacls[0]).reshape(-1)
        index_class1 = np.argwhere(train_label == negacls[1]).reshape(-1)
        n_nega0, n_nega1 = len(index_class0), len(index_class1)
        nindx = [index_class0[: n_nega0 // 2], index_class0[n_nega0 // 2:],
                 index_class1[: n_nega1 // 2], index_class1[n_nega1 // 2:]]

    for i in range(2):
        for j in range(4):
            s = i * 4 + j
            print("training the %dth SVM" % s)
            train_single(net.clf[s], category, pindx[i], nindx[j])

    print("finish training!")
    return net


if __name__ == '__main__':
    # random task decomposition
    method = 'prior'
    net1 = train(-1, method)
    net2 = train(0, method)
    net3 = train(1, method)
    decision1 = net1.decision_function(test_data)
    decision2 = net2.decision_function(test_data)
    decision3 = net3.decision_function(test_data)
    test_pred = np.vstack((decision1, decision2, decision3))
    test_pred = np.argmax(test_pred, axis=0)
    test_pred = np.where(test_pred == 0, -1, test_pred)
    test_pred = np.where(test_pred == 1, 0, test_pred)
    test_pred = np.where(test_pred == 2, 1, test_pred)
    accuracy = np.sum(test_pred == test_label) / len(test_label)
    print("final accuracy: %f", accuracy)