import torch
import torch.nn as nn
from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, _lambda):
        ctx._lambda = _lambda
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx._lambda
        return output, None


class DANN(nn.Module):

    def __init__(self):
        super(DANN, self).__init__()
        self.feature_extractor = nn.Sequential(nn.Linear(310, 128),
                                               nn.ReLU(),
                                               nn.Linear(128, 128))

        self.label_predictor = nn.Sequential(nn.Linear(128, 64),
                                             nn.ReLU(),
                                             nn.Linear(64, 64),
                                             nn.ReLU(),
                                             nn.Linear(64, 3))

        self.domain_predictor = nn.Sequential(nn.Linear(128, 64),
                                              nn.ReLU(),
                                              nn.Linear(64, 64),
                                              nn.ReLU(),
                                              nn.Linear(64, 2))

    def forward(self, x, _lambda):
        x = self.feature_extractor(x)
        reverse_x = ReverseLayerF.apply(x, _lambda)
        class_out = self.label_predictor(x)
        domain_out = self.domain_predictor(reverse_x)
        return class_out, domain_out





if __name__ == '__main__':
    import pickle

    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)
        print(data.keys())
