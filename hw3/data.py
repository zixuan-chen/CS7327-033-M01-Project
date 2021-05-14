import torch.utils.data as data
import numpy as np
import pickle
import torch
from sklearn.preprocessing import minmax_scale

path = "data.pkl"


class SEEDDataSet(data.Dataset):

    def __init__(self, category: list, transform=None):

        with open(path, 'rb') as f:
            data = pickle.load(f)

        feature_list = []
        target_list = []
        for c in category:
            if c not in data.keys():
                print("Error: %s is not a valid category name, automatically ignored" % c)
            else:
                feature_list.append(data[c]['data'])
                target_list.append(data[c]['label'])

        feature = np.concatenate(feature_list, axis=0)
        target = np.concatenate(target_list, axis=0)

        lf = feature.shape[0]
        lt = target.shape[0]

        assert lf == lt
        minmax_scale(feature, feature_range=(-1, 1), axis=1, copy=False)
        # normalization

        self.feature = torch.tensor(feature, dtype=torch.float)
        self.target = torch.tensor(target, dtype=torch.long) + 1

    def __getitem__(self, item):
        return self.feature[item], self.target[item]

    def __len__(self):
        return self.feature.shape[0]


class SEEDDatasetWithDomain(data.Dataset):
    def __init__(self, source_category: list, target_category: list, transform=None):

        with open(path, 'rb') as f:
            data = pickle.load(f)

        feature_list = []
        label_list = []
        domain_list = []
        for c in source_category+target_category:
            if c not in data.keys():
                print("Error: %s is not a valid category name, automatically ignored" % c)
            else:
                feature_list.append(data[c]['data'])
                label_list.append(data[c]['label'])
                if c in source_category:
                    domain_list.append(np.zeros(len(data[c]['label'])))
                else:
                    domain_list.append(np.ones(len(data[c]['label'])))

        feature = np.concatenate(feature_list, axis=0)
        label = np.concatenate(label_list, axis=0)
        domain = np.concatenate(domain_list, axis=0)

        # minmax_scale(feature, feature_range=(-1, 1), axis=1, copy=False)
        # normalization

        self.feature = torch.tensor(feature, dtype=torch.float)
        self.label = torch.tensor(label, dtype=torch.long) + 1
        self.domain = torch.tensor(domain, dtype=torch.long)

    def __getitem__(self, item):
        return self.feature[item], self.label[item], self.domain[item]

    def __len__(self):
        return self.feature.shape[0]
