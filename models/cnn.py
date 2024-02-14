import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN, self).__init__()
        # First 1D convolutional layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        # Second 1D convolutional layer
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Third 1D convolutional layer
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # Fully connected layer
        self.fc = nn.Linear(64 * 12, num_classes)

    def forward(self, x):
        # Input shape: (batch_size, 1, input_size)
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool1d(x, 2, 2)
        x = x.view(-1, 64 * x.size(2))  # Flatten the input for the fully connected layer
        x = self.fc(x)
        return x

