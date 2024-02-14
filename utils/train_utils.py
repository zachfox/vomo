import torch
import torch.nn as nn
import torch.optim as optim

import wandb

class Trainer:
    def __init__(self, model, train_loader, val_loader, logger, config, device='cpu'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        # parse config for settings
        if config['criterion'] == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        if config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr = config['lr'])
        self.device = device
        self.logger = logger

    def train_epoch(self):
        # Set the model to training mode
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0

        # Iterate over the training dataset
        for inputs, targets in self.train_loader:
            # Move inputs and targets to the specified device (e.g., GPU)
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Zero the gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)

            # Compute the loss
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Update the parameters
            self.optimizer.step()

            # Accumulate the total loss
            total_loss += loss.item()

            # Calculate the number of correct predictions
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == targets).sum().item()

        # Calculate average loss and accuracy for the epoch
        avg_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / len(self.train_loader.dataset)

        # Logging 
        self.logger.log({'train acc':accuracy, 'train loss':avg_loss}, commit=False)

        return avg_loss, accuracy

    def evaluate(self):
        # Set the model to evaluation mode
        self.model.eval()

        total_loss = 0.0
        total_correct = 0

        # Disable gradient computation during validation
        with torch.no_grad():
            # Iterate over the validation dataset
            for inputs, targets in self.val_loader:
                # Move inputs and targets to the specified device (e.g., GPU)
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)

                # Compute the loss
                loss = self.criterion(outputs, targets)

                # Accumulate the total loss
                total_loss += loss.item()

                # Calculate the number of correct predictions
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == targets).sum().item()

        # Calculate average loss and accuracy for the validation set
        avg_loss = total_loss / len(self.val_loader)
        accuracy = total_correct / len(self.val_loader.dataset)
        self.logger.log({'val acc':accuracy, 'val loss':avg_loss})
        return avg_loss, accuracy


