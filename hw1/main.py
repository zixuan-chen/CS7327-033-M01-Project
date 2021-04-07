from model import *
from utils import *


def train_mlqp(net, data, epochs=200, bs=1, lr=0.0002):
    n = len(data)
    remainder = n % bs
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    criterion = nn.BCEWithLogitsLoss(reduce='mean')
    targets = data[:, -1].unsqueeze(dim=1)
    loss_history = []
    acc_history = []
    for epoch in range(epochs):
        running_loss = 0.0
        for i in range(0, n, bs):
            input = data[i:i + bs, :-1]
            target = targets[i:i + bs]
            optimizer.zero_grad()
            output = net(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # if i % 10 == 0:
            #     with torch.no_grad():
            #         input = data[:, :-1]
            #         target = data[:, -1].type(torch.long)
            #         output = net(input)
            #         output = output.argmax(dim=1)
            #         accuracy = torch.sum((output == target).type(torch.int)) / n
            #     print('[%d, %5d] running loss: %.3f, loss: %.3f, test accuracy: %.3f' %
            #           (epoch + 1, i + 1, running_loss / (i+1), loss.item(), accuracy))
        if remainder:
            input = data[-remainder:, :-1]
            target = targets[-remainder:, -1]
            optimizer.zero_grad()
            output = net(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            input = data[:, :-1]
            target = targets
            output = net(input)
            loss = criterion(output, target)
            output = output > 0
            accuracy = torch.sum((output == target).type(torch.int)) / n

        print('epoch: %d loss: %.3f, accuracy: %.3f' % (epoch + 1, loss.item(), accuracy))
        loss_history.append(loss.item())
        acc_history.append(accuracy)
    x = list(range(epochs))
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(x, loss_history, label="loss")
    axs[0].legend()
    axs[1].plot(x, acc_history, label='accuracy')
    axs[1].legend()
    plt.show()

    print("Finished Training")
    return net


def train_min_max(net, epochs=100, bs=1, lr=0.0002):
    # train
    _, sub_data = load_data_with_partition(True)

    for i in range(4):
        print("Train MLQP on sub-problems %d:" % i)
        np.random.shuffle(sub_data[i])
        data = torch.tensor(sub_data[i], dtype=torch.float)
        train_mlqp(net.net[i], data, epochs=epochs, bs=bs, lr=lr)

    # test
    print("Test min-max modular network")
    data = load_data_with_partition(False)
    n = len(data)
    targets = data[:, -1].unsqueeze(dim=1)
    with torch.no_grad():
        input = data[:, :-1]
        target = targets
        output = net(input)
        output = output > 0
        accuracy = torch.sum((output == target).type(torch.int)) / n
        print("accuracy: %.3f" % accuracy)


if __name__ == '__main__':
    import time

    data = load_data_with_partition(False)

    # =================== train MLQP networks===========================
    net = nn.Sequential(Quadratic(2, 512), nn.Sigmoid(), Quadratic(512, 1))
    start = time.time()
    train_mlqp(net, data, epochs=200, bs=1, lr=0.001)
    end = time.time()
    print("running time: %.2f" % (end - start))
    fig, ax = plt.subplots()
    plot_decision_boundary(net, ax, 0.1)
    plot_data(data, ax, lw=0.5, clr0='lightskyblue', clr1='lightpink')
    plt.show()

    # =================== train min-max net =============================
    min_max_net1 = MinMaxNet2x2()
    start = time.time()
    train_min_max(min_max_net1, epochs=100, bs=1, lr=0.0002)
    end = time.time()
    print("running time: %.2f" % (end - start))

    fig, ax = plt.subplots()
    plot_decision_boundary(min_max_net1, ax, 0.1)
    plot_data(data, ax, lw=0.5, clr0='lightskyblue', clr1='lightpink')
    plt.show()
