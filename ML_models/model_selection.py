import os
from collections import defaultdict
from copy import deepcopy as dc

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    make_scorer,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    MinMaxScaler
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from xgboost import XGBClassifier
from tqdm import tqdm

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
joblib.dump(list(X_resampled.columns), "selected_features.pkl")
models = {
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Naïve Bayes': GaussianNB(),
    'Extra Trees': ExtraTreesClassifier(random_state=42),
    'MLP (Neural Network)': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
}

# Define metrics
metrics = {
    'Accuracy': 'accuracy',
    'Precision': make_scorer(precision_score, average='weighted'),
    'Recall': make_scorer(recall_score, average='weighted'),
    'F1 Score': make_scorer(f1_score, average='weighted')
}


kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

def evaluate_model(model_name, model):
    results = {}
    for metric_name, metric in metrics.items():
        scores = cross_val_score(model, X_resampled, y_resampled, cv=kfold, scoring=metric, n_jobs=-1)
        results[metric_name] = {
            'Mean Score': np.mean(scores),
            'Standard Deviation': np.std(scores),
            'Scores': scores.tolist()
        }
    return model_name, results


BEST_MODEL_CRITERION = 'Accuracy'  


parallel_results = Parallel(n_jobs=-1)(
    delayed(evaluate_model)(name, model) for name, model in tqdm(models.items(), desc="Evaluating Models")
)

# Convert results to a Pandas DataFrame
data_records = []
best_model_name = None
best_model_score = -np.inf
best_model = None

for model_name, results in tqdm(parallel_results, desc="Processing Results"):
    for metric_name, values in results.items():
        data_records.append({
            'Model': model_name,
            'Metric': metric_name,
            'Mean Score': values['Mean Score'],
            'Standard Deviation': values['Standard Deviation'],
            'All Scores': values['Scores']  
        })

        # Track the best model based on selected metric
        if metric_name == BEST_MODEL_CRITERION and values['Mean Score'] > best_model_score:
            best_model_score = values['Mean Score']
            best_model_name = model_name
            best_model = models[model_name]


df_results = pd.DataFrame(data_records)
df_results.to_csv("model_evaluation_results.csv", index=False)

print(f"Best model based on {BEST_MODEL_CRITERION}: {best_model_name} with score {best_model_score:.4f}")



best_model.fit(X_resampled, y_resampled)

# Save the best model
joblib.dump(best_model, f"{best_model_name.replace(' ', '_')}_best_model.pkl")

