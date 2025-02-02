import torch
import torch.nn as nn
import torch.optim as optim

# Define a 1D CNN Model for EEG Classification
class EEG1DCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(EEG1DCNN, self).__init__()
        
        # 1D Convolutional Layers with BatchNorm & Dropout
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.5)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.5)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)


        self.fc = nn.Linear(256, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)  

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)

        x = self.global_avg_pool(x).squeeze(-1)
        x = self.fc(x)
        
        return x



class EEGClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(EEGClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
