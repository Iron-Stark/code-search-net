import json
import pandas as pd
from pathlib import Path
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer


class CodeSearchNetDataset(Dataset):

    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.model = SentenceTransformer('bert-base-nli-mean-tokens').to('cuda')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return {'code': self.dataframe.iloc[idx]['code'],
                'docstring': self.dataframe.iloc[idx]['docstring'],
                'code_emb': self.model.encode(self.dataframe.iloc[idx]['code']),
                'docstring_emb': self.model.encode(self.dataframe.iloc[idx]['docstring'])}


def jsonl_list_to_dataframe(files):
    """Load a list of jsonl.gz files into a pandas DataFrame."""
    return pd.concat([pd.read_json(f,
                                   orient='records',
                                   compression='gzip',
                                   lines=True)
                      for f in files], sort=False)


def get_dataset(data_type):
    """
    :parameter
        data_type (str): Either of train, valid and test
    :return: Dataset for given type
    """
    return CodeSearchNetDataset(pd.read_csv(f'generated_resources/{data_type}_data.csv'))


def get_dataloaders(data_type, batch_size, shuffle):
    """
    :parameter
        data_type (str): Either of train, valid and test
        batch_size (int): Batch size for the dataloader
        shuffle (bool): Whether you want to shuffle the entries or not
    :return:
        Dataloader for given type
    """

    return DataLoader(get_dataset(data_type), batch_size=batch_size, shuffle=True)