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
import joblib
import json
import matplotlib.pyplot as plt
file_path = 'Train_and_Validate_EEG.csv'
df = pd.read_csv(file_path)
df = df.drop(columns=["ID", "Unnamed: 122", "eeg.date", "specific.disorder"], errors="ignore")
df = df.dropna()

# Normalize numerical columns
minmax_scaler = MinMaxScaler()

df['sex'] = df['sex'].map({'M': 1, 'F': 0})
numerical_cols = df.select_dtypes(include=['number']).columns
df[numerical_cols] = minmax_scaler.fit_transform(df[numerical_cols])

joblib.dump(minmax_scaler, 'minmax_scaler.pkl')


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

joblib.dump(label_encoder, 'label_encoder.pkl')
unique_classes = np.unique(target_encoded)
sampling_strategy = {label: 600 for label in unique_classes}
smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)

X_resampled, y_resampled = smote.fit_resample(features, target_encoded)

# Load the trained ExtraTrees model
best_model = joblib.load("best_extratrees_model.pkl")

# Convert data to DataFrame format if needed
X_resampled_df = pd.DataFrame(X_resampled, columns=features.columns)

import joblib
import lime.lime_tabular
import numpy as np
from multiprocessing import Pool

# Initialize LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_resampled_df.values,
    feature_names=X_resampled_df.columns,
    class_names=["Class_" + str(i) for i in np.unique(y_resampled)],
    mode="classification"
)

# Function to generate LIME explanations for a single instance
def explain_instance(i):
    sample_instance = X_resampled_df.iloc[i].values.reshape(1, -1)
    exp = explainer.explain_instance(
        data_row=sample_instance[0],
        predict_fn=best_model.predict_proba
    )
    return exp

# Use multiprocessing to compute LIME explanations in parallel
num_cores = cpu_count()
with Pool(num_cores) as p:
    lime_explanations = p.map(explain_instance, range(len(X_resampled_df)))



# Generate the LIME explanation figure
fig = lime_explanations[0].as_pyplot_figure()

# Save the figure as a PNG file
fig.savefig("lime_explanation.png", dpi=300, bbox_inches="tight")