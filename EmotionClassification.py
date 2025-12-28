import pandas as pd
import itertools

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

import torchtext.data
from torchtext.vocab import build_vocab_from_iterator

# from lightning.pytorch import LightningModule, Trainer
# from torchmetrics import Accuracy


# # Load dataset as dataframe
# def load_df(path: str) -> pd.DataFrame:
#     with path as p:
#         with p.open('/mnt/c/Users/Saffron/Documents/Ontario Tech Class Notes/Thesis/AI_Powered_Game/Datasets/emotions.csv') as f:
#             df = pd.read_csv(f)
#     return df
train_iter, test_iter = torchtext.datasets.emotions(
    root='/mnt/c/Users/Saffron/Documents/Ontario Tech Class Notes/Thesis/AI_Powered_Game/Datasets/emotions.csv', 
    split=('train', 'test')
)

data = list(iter(train_iter))
len(data)

# # Tokenize sequences
# tokenizer = torchtext.data.get_tokenizer('basic_english')

# def iterate_tokens(df):
#     for text in df:
#         yield tokenizer(str(text))

