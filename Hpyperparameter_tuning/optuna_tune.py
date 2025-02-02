import os
from collections import defaultdict
from copy import deepcopy as dc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from xgboost import XGBClassifier
from models_optuna import EEG1DCNN, EEGClassifier


file_path = 'EDA/df_imputed.csv'

df = pd.read_csv(file_path)
df = df.drop(columns=["ID","Unnamed: 122", "eeg.date", "specific.disorder"], errors="ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Remove the target variable for clustering
data_reshaped = df.drop(columns=["main.disorder"])

# Compute Spearman correlation matrix
corr, _ = spearmanr(data_reshaped, axis=0)
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)

distance_matrix = 1 - np.abs(corr)
dist_linkage = hierarchy.ward(squareform(distance_matrix))

threshold = 0.05 * np.max(dist_linkage[:, 2]) 
clusters = fcluster(dist_linkage, threshold, criterion="distance")


cluster_map = defaultdict(list)
for i, feature in enumerate(data_reshaped.columns):
    cluster_map[clusters[i]].append(feature)


selected_features = [features[0] for features in cluster_map.values()]
df_selected = df[selected_features + ["main.disorder"]]  
with open("training_df_columns.txt", "w") as f:
    for col in df_selected.columns:
        f.write(col + "\n")

target_column = "main.disorder"
features = df[selected_features]
target = df[target_column]
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)
X_train, X_val, y_train, y_val = train_test_split(features, target_encoded, test_size=0.2, random_state=42)

class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score

# Define the objective function
def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    hidden_dim1 = trial.suggest_int("hidden_dim1", 128, 1024, step=128)
    hidden_dim2 = trial.suggest_int("hidden_dim2", 128, 512, step=64)
    dropout_rate = trial.suggest_uniform("dropout", 0.2, 0.6)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)

    # Create DataLoaders with suggested batch_size
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define Model with suggested hidden dimensions and dropout
    class EEGClassifier(nn.Module):
        def __init__(self, input_dim, num_classes):
            super(EEGClassifier, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim1)
            self.bn1 = nn.BatchNorm1d(hidden_dim1)
            self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
            self.bn2 = nn.BatchNorm1d(hidden_dim2)
            self.fc3 = nn.Linear(hidden_dim2, num_classes)
            self.dropout = nn.Dropout(dropout_rate)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.bn1(self.fc1(x)))
            x = self.dropout(x)
            x = self.relu(self.bn2(self.fc2(x)))
            x = self.dropout(x)
            x = self.fc3(x)
            return x

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = X_train.shape[1]
    num_classes = len(label_encoder.classes_)

    model = EEGClassifier(input_dim, num_classes).to(device)

    # Compute class weights
    class_weights = compute_class_weight(class_weight="balanced", 
                                         classes=np.unique(y_train), 
                                         y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training loop
    num_epochs = 2000  # Keep small for Optuna
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # Evaluate on validation set
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y_batch.numpy())

    accuracy = accuracy_score(y_true, y_pred)
    return accuracy  # Higher is better

# Run Optuna study
study = optuna.create_study(direction="maximize")  # Maximize accuracy
study.optimize(objective, n_trials=100, n_jobs=-1)  # Run 30 trials

# Print best parameters
print("Best hyperparameters:", study.best_params)
"""Best hyperparameters: {'lr': 0.0010196581994490728, 'batch_size': 64, 'hidden_dim1': 512, 'hidden_dim2': 128, 'dropout': 0.23099732616032995, 'weight_decay': 7.439210733192944e-05}