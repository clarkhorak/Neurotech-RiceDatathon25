import optuna
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score

# file_path = 'Train_and_Validate_EEG.csv'
from scipy.stats import pearsonr, spearmanr
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
import os
import numpy as np
from collections import defaultdict
from copy import deepcopy as dc
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

file_path = 'Train_and_Validate_EEG.csv'
df = pd.read_csv(file_path)
df = df.drop(columns=["ID", "Unnamed: 122", "eeg.date", "specific.disorder"], errors="ignore")
df = df.dropna()

# Normalize numerical columns
minmax_scaler = MinMaxScaler()
numerical_cols = df.select_dtypes(include=['number']).columns
df[numerical_cols] = minmax_scaler.fit_transform(df[numerical_cols])
joblib.dump(minmax_scaler, 'minmax_scaler.pkl')
df['sex'] = df['sex'].map({'M': 1, 'F': 0})

def feature_selection(threshold_v, df_features):
    corr, _ = spearmanr(df_features, axis=0)
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))

    threshold = threshold_v * np.max(dist_linkage[:, 2])
    clusters = fcluster(dist_linkage, threshold, criterion="distance")

    cluster_map = defaultdict(list)
    for i, feature in enumerate(df_features.columns):
        cluster_map[clusters[i]].append(feature)

    selected_features = [features[0] for features in cluster_map.values()]

    return selected_features

selected_features = feature_selection(0.01, df)

target_column = "main.disorder"
features = df[selected_features].drop(columns=[target_column])
target = df[target_column]
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)
num_classes = len(label_encoder.classes_)

unique_classes = np.unique(target_encoded)
sampling_strategy = {label: 600 for label in unique_classes}
smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)

X_resampled, y_resampled = smote.fit_resample(features, target_encoded)


# Define objective function for Optuna
def objective(trial):
    # Define the hyperparameters to tune
    n_estimators = trial.suggest_int("n_estimators", 50, 500, step=50)
    max_depth = trial.suggest_int("max_depth", 5, 50, step=5)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])

    # Initialize the model with sampled hyperparameters
    model = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )

    # Evaluate using cross-validation
    score = cross_val_score(model, X_resampled, y_resampled, cv=5, scoring="accuracy").mean()
    
    return score  # Optuna tries to maximize this value

# Run Optuna optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Best hyperparameters
best_params = study.best_params
print("Best Hyperparameters:", best_params)

# Train model with best hyperparameters
best_model = ExtraTreesClassifier(**best_params, random_state=42)
best_model.fit(X_resampled, y_resampled)

# Save the best model
import joblib
joblib.dump(best_model, "best_extratrees_model.pkl")

