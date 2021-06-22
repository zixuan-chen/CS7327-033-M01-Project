import torch
from torch.utils.data.dataloader import Dataset

video_slices = [238, 233, 206, 238, 185, 195, 237, 216, 265, 237, 235, 233, 235, 238, 206]
len_per_subject = 3397

class SequentialSEEDDataset(Dataset):
    def __init__(self, data, label, step, seq_len):
        assert data.shape[0] == label.shape[0]
        self.data = torch.tensor(data, dtype=torch.float)
        self.label = torch.tensor(label, dtype=torch.long)
        self.seq_len = seq_len
        # base = 0
        self.index = list(range(0, data.shape[0] - seq_len, step))
        # for num in sub_sample_num:
        #     self.index = self.index + list(range(base, base + num - 1, step))
        #     base += num

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        idx = self.index[item]
        return self.data[idx: idx + self.seq_len], self.label[idx + self.seq_len]


class SequentialSEEDDataSetWithPrior(Dataset):
    def __init__(self, data, label):
        assert data.shape[0] == label.shape[0]
        self.data = torch.tensor(data, dtype=torch.float)
        self.label = torch.tensor(label, dtype=torch.long)
        self.seq_len = max(video_slices)
        self.num_subject = self.data.shape[0] // len_per_subject
        self.feature = torch.zeros((self.num_subject * len(video_slices), self.seq_len, 310), dtype=torch.float)
        self.target = torch.zeros(self.num_subject * len(video_slices), dtype=torch.long)

        for i in range(self.num_subject):
            base = i * len_per_subject
            for j in range(len(video_slices)):
                batch = i * len(video_slices) + j
                num = video_slices[j]
                self.feature[batch, 0:num, :] = self.data[base: base+num]
                self.target[batch] = self.label[base]
                base += num

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, item):
        return self.feature[item], self.target[item]
