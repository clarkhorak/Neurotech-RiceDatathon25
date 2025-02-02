import os
from collections import defaultdict
from copy import deepcopy as dc

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess the dataset
file_path = 'Train_and_Validate_EEG.csv'

df = pd.read_csv(file_path)
df = df.drop(columns=["ID","Unnamed: 122", "eeg.date", "specific.disorder"], errors="ignore")
df = df.dropna()

# Normalize numerical columns

minmax_scaler = MinMaxScaler()
numerical_cols = df.select_dtypes(include=['number']).columns

df[numerical_cols] = minmax_scaler.fit_transform(df[numerical_cols])
joblib.dump(minmax_scaler, 'minmax_scaler.pkl') 
# Encode 'sex' column
df['sex'] = df['sex'].map({'M': 1, 'F': 0})

# Remove the target variable for clustering
data_reshaped = df.drop(columns=["main.disorder"])

# Compute Spearman correlation matrix
corr, _ = spearmanr(data_reshaped, axis=0)
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)

# Convert correlation matrix to distance matrix and perform hierarchical clustering
distance_matrix = 1 - np.abs(corr)
dist_linkage = hierarchy.ward(squareform(distance_matrix))

# Loop over different thresholds for feature selection
for threshold_v in [0.01, 0.05, 0.075, 0.1]:
    # Define the threshold for clustering
    threshold = threshold_v * np.max(dist_linkage[:, 2])
    clusters = fcluster(dist_linkage, threshold, criterion="distance")

    # Map features to clusters
    cluster_map = defaultdict(list)
    for i, feature in enumerate(data_reshaped.columns):
        cluster_map[clusters[i]].append(feature)

    # Select one feature per cluster
    selected_features = [features[0] for features in cluster_map.values()]
    print(f"Reduced from {data_reshaped.shape[1]} to {len(selected_features)} features.")

    # Prepare dataset with selected features
    df_selected = df[selected_features + ["main.disorder"]]

    # Split data into train and validation sets
    target_column = "main.disorder"
    features = df[selected_features]
    target = df[target_column]
    label_encoder = LabelEncoder()
    target_encoded = label_encoder.fit_transform(target)
    X_train, X_val, y_train, y_val = train_test_split(features, target_encoded, test_size=0.2, random_state=42)

    # Initialize and train the XGBoost classifier
    model = XGBClassifier(
        n_estimators=200, 
        learning_rate=0.05, 
        max_depth=6, 
        scale_pos_weight=1, 
        use_label_encoder=False, 
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train)

    # Evaluate model performance
    y_val_pred = model.predict(X_val)
    y_val_probs = model.predict_proba(X_val)

    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy for threshold {threshold_v}: {val_acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_val, y_val_pred, target_names=label_encoder.classes_))

    # Compute AUC for each class
    y_val_binarized = label_binarize(y_val, classes=np.unique(y_train))
    auc_scores = roc_auc_score(y_val_binarized, y_val_probs, average=None)
    auc_df = pd.DataFrame({"Class": label_encoder.classes_, "AUC": auc_scores})

    # Save AUC scores for this threshold
    os.makedirs("results", exist_ok=True)
    auc_df.to_csv(f"results/XGBoost_auc_df_{threshold_v}.csv", index=False)
    print(auc_df)
