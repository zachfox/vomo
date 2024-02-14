import torch
from models.cnn import CNN
from utils.data_utils import get_dataloaders
from utils.train_utils import Trainer
import wandb
wandb.login()

config = {'sub_mean': False,
          'batch_size': 4,
          'train_frac': .7,
          'criterion': 'cross_entropy',
          'optimizer': 'adam',
          'lr': 1e-5,
          'n_epochs': 200  }

logger = wandb.init(project="vomocytosis", config=config)

input_size = 100
num_classes = 3

# Create an instance of the cnn
model = CNN(input_size, num_classes)

# get the dataloaders
train_loader, val_loader = get_dataloaders('data', config)

# train the model
trainer = Trainer(model, train_loader, val_loader, logger, config)

for epoch in range(config['n_epochs']):
    trainer.train_epoch()
    trainer.evaluate()

