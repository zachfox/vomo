import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class VomoCytDataset(Dataset):
    def __init__(self, data, labels, max_sequence_length=None, pad_value=0):
        """
        Args:
            data (list): List of time series sequences.
            labels (list): List of corresponding labels.
            max_sequence_length (int): Maximum length of the sequences after padding.
            pad_value (float): Value to pad the sequences with.
        """
        self.data = data
        self.labels = labels
        self.max_sequence_length = max_sequence_length
        self.pad_value = pad_value

        # If max_sequence_length is not provided, determine it based on the maximum length in the dataset
        if max_sequence_length is None:
            self.max_sequence_length = max(len(seq) for seq in data)

        # Pad the sequences
        self.data = [self.pad_sequence(seq) for seq in self.data]

    def pad_sequence(self, sequence):
        """
        Pad a sequence to the specified length.
        """
        if len(sequence) >= self.max_sequence_length:
            return sequence[:self.max_sequence_length]
        else:
            return sequence + [self.pad_value] * (self.max_sequence_length - len(sequence))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # sample = {'data': torch.Tensor(self.data[idx]), 'label': torch.tensor(self.labels[idx], dtype=torch.long)}
        sample = torch.Tensor(self.data[idx]).unsqueeze(0), torch.tensor(self.labels[idx], dtype=torch.long)
        return sample

def load_data(data_path):
    class_0 = list(pd.read_csv(os.path.join(data_path,'vomocytosed.csv')).to_numpy())
    _labels = [0]*len(class_0)
    class_1 = list(pd.read_csv(os.path.join(data_path,'internalized.csv')).to_numpy())
    _labels += [1]*len(class_1)
    class_2 = list(pd.read_csv(os.path.join(data_path,'not_internalized.csv')).to_numpy())
    _labels += [2]*len(class_2)

    _data = np.array(class_0+class_1+class_2)
    _labels = np.array(_labels)
    return _data, _labels

def get_dataloaders(data_path, config):
    print('generating dataloaders...')
    _data, _labels = load_data(data_path)  
    if config['sub_mean']:
        _data = _data-np.atleast_2d(np.mean(_data, axis=1)).T
    inds = np.arange(len(_data))
    np.random.shuffle(inds)
    train_stop = int(len(_data)*config['train_frac'])
    train_inds = inds[:train_stop]
    val_inds = inds[train_stop:]
    print('number of vomocytoses in training: {0}'.format(np.sum(_labels[train_inds]==0)))
    print(len(list(_data[train_inds,:])))
    train_dataset = VomoCytDataset(list(_data[train_inds,:]), list(_labels[train_inds]))
    val_dataset = VomoCytDataset(list(_data[val_inds,:]), list(_labels[val_inds]))

    train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, config['batch_size'], shuffle=False)
    
    print('...done')
    return train_loader, val_loader


if __name__=='__main__':
    config = {'train_frac':.5, 'batch_size':4}
    get_dataloaders('data/', config) 
