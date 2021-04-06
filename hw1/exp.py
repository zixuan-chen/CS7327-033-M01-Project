from mlqp import *
import numpy as np

path = "two-spiral traing data(update).txt"
cuda0 = torch.device('cuda:0')


def train_online(epochs=1, bs=1):
    net = MLQP()
    lines = open(path).readlines()
    n = len(lines)
    data = torch.zeros(size=(n, 3), dtype=torch.float)
    labels = torch.zeros(size=(n, 2), dtype=torch.int)

    for i in range(n):
        data[i] = torch.tensor([float(j) for j in lines[i].split()])
        cls = int(data[i, -1])
        labels[i, cls] = 1

    remainder = n % bs
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0002, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss = 0.0
        for i in range(0, n, bs):
            input = data[i:i + bs, :-1]
            target = data[i:i + bs, -1].type(torch.long)
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
            target = data[-remainder:, -1].type(torch.long)
            optimizer.zero_grad()
            output = net(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            input = data[:, :-1]
            target = data[:, -1].type(torch.long)
            output = net(input)
            loss = criterion(output, target)
            output = output.argmax(dim=1)

            accuracy = torch.sum((output == target).type(torch.int)) / n
        print('epoch: %d loss: %.3f, accuracy: %.3f' % (epoch + 1, loss.item(), accuracy))
    print("Finished Training")


def plot_decision_boundaries():
    pass


if __name__ == '__main__':
    train_online(200, 1)
