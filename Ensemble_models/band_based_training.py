import os
from itertools import combinations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from xgboost import XGBClassifier

# Function to sort while keeping AB and COH together
def band_sort(feature_list, band):
    """Sorts AB and COH features together within each frequency band"""
    ab_features = sorted([col for col in feature_list if col.startswith("AB") and band in col])
    coh_features = sorted([col for col in feature_list if col.startswith("COH") and band in col])
    return ab_features + coh_features  # Ensure AB and COH appear together

def train(bands, df_sorted):
    target_col = 'main.disorder'  # Target variable: main disorder

    sorted_features = []
    feature_list = [col for col in df_sorted.columns if col.startswith("AB") or col.startswith("COH")]
    for band in bands:
        ab_features = sorted([col for col in feature_list if col.startswith("AB") and band in col])
        coh_features = sorted([col for col in feature_list if col.startswith("COH") and band in col])

        # Keep AB and COH features together for each band
        sorted_features.extend(ab_features + coh_features)

    selected_features = demographic_features + sorted_features
    features = df_sorted[selected_features]
    y = df_sorted[target_col]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(features.values, y, test_size=0.2, random_state=42, stratify=y)

    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)

    model = XGBClassifier(n_estimators=100, max_depth=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, X_test, y_test

if __name__ == "__main__":
    file_path = 'Train_and_Validate_EEG.csv'
    df = pd.read_csv(file_path)
    df = df.drop(columns=["Unnamed: 122", "eeg.date", "specific.disorder"], errors="ignore")
    df = df.dropna()

    # Initialize MinMaxScaler
    minmax_scaler = MinMaxScaler()
    joblib.dump(minmax_scaler, "bandmodel_minmax_scaler.pkl")
    numerical_cols = df.select_dtypes(include=['number']).columns
    df[numerical_cols] = minmax_scaler.fit_transform(df[numerical_cols])
    df['sex'] = df['sex'].map({'M': 1, 'F': 0})
    feature_columns = [col for col in df.columns if col.startswith("AB") or col.startswith("COH")]

    # Define frequency band order (Delta first, then others)
    freq_order = ["A.delta", "B.theta", "C.alpha", "D.beta", "E.highbeta", "F.gamma"]

    # Sort features for each band
    sorted_features = []
    for band in freq_order:
        sorted_features.extend(band_sort(feature_columns, band))

    # Define demographic features
    demographic_features = ["sex", "age", "education", "IQ"]

    # Reorder DataFrame
    df_sorted = df[demographic_features + sorted_features + ['main.disorder']]  # Keep target variable at the end

    with open("training_df_columns.txt", "w") as f:
        for col in df_sorted.columns:
            f.write(col + "\n")

    bands = ["delta", "theta", "alpha", "beta", "highbeta", "gamma"]

    all_combinations = []
    for k in range(1, 7):
        all_combinations.extend(combinations(bands, k))

    os.makedirs("Final_models", exist_ok=True)
    for bands in tqdm(all_combinations, desc="Training Models", unit="model"):
        model, X_test, y_test = train(list(bands), df_sorted)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        fn = "_".join(bands)
        model_filename = f"Final_models/{fn}.pkl"
        joblib.dump(model, model_filename)
