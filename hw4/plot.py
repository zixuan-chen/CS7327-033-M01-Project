import numpy as np
import random
import os
import torch
import pandas as pd
from hw3.util import plotLossAccSingle

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False


if __name__ == '__main__':
    df = pd.read_csv("result(with prior).csv")
    plotLossAccSingle(loss=df['loss'][:350],
                      acc=df['accuracy'][:350],
                      valacc=None,
                      save=True,
                      save_title="LSTMresult(with prior).png",
                      save_dir="./imgs",
                      smooth=True,
                      window_size=50,
                      factor=0.9)