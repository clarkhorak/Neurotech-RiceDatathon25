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

# Convert labels to one-hot encoding
from sklearn.preprocessing import label_binarize

file_path = 'Train_and_Validate_EEG.csv'

df = pd.read_csv(file_path)
df = df.drop(columns=["ID", "Unnamed: 122", "eeg.date", "specific.disorder"], errors="ignore")
df = df.dropna()

# Normalize numerical columns
minmax_scaler = MinMaxScaler()
numerical_cols = df.select_dtypes(include=['number']).columns
df[numerical_cols] = minmax_scaler.fit_transform(df[numerical_cols])

# Encode 'sex' column
df['sex'] = df['sex'].map({'M': 1, 'F': 0})

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

with open("./training_df_columns.txt", "w") as f:
    for col in df_selected.columns:
        f.write(col + "\n")

print(f"Reduced from {data_reshaped.shape[1]} to {len(selected_features)} features.")

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

# Create DataLoaders
batch_size = 64
train_dataset = EEGDataset(X_train, y_train)
val_dataset = EEGDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Compute class weights
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

input_dim = X_train.shape[1]
num_classes = len(label_encoder.classes_)
model = EEGClassifier(input_dim, num_classes).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=7.439210733192944e-05)

num_epochs = 5000
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = correct / total

    # Validation phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_acc = correct / total

    if epoch % 500 == 0:
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

model.eval()

correct = 0
total = 0
class_correct = defaultdict(int)
class_total = defaultdict(int)
all_labels = []
all_preds = []

# Run inference on validation set
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for i in range(len(labels)):
            label = labels[i].item()
            pred = predicted[i].item()
            class_total[label] += 1
            if label == pred:
                class_correct[label] += 1

            all_labels.append(label)
            all_preds.append(pred)

per_class_accuracy = {label_encoder.inverse_transform([cls])[0]: class_correct[cls] / class_total[cls]
                      for cls in class_total if class_total[cls] > 0}

accuracy_df = pd.DataFrame.from_dict(per_class_accuracy, orient='index', columns=["Accuracy"])
accuracy_df = accuracy_df.sort_values(by="Accuracy", ascending=False)

model.eval()

# Initialize storage lists
all_labels = []
all_probs = []

# Run inference on validation data
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

y_true_binarized = label_binarize(all_labels, classes=np.unique(all_labels))

# AUC
auc_scores = roc_auc_score(y_true_binarized, all_probs, average=None)
auc_df = pd.DataFrame({"Class": label_encoder.classes_, "AUC": auc_scores})

print(auc_df)
print(auc_df['AUC'].mean())

torch.save(model, "tuned_model.pth")
