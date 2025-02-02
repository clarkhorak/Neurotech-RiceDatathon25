import joblib
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, label_binarize
from xgboost import XGBClassifier



def create_pca_pipeline(X, y, n_components=0.95):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = Pipeline([
        ('pca', PCA(n_components=n_components)),
        ('classifier', XGBClassifier(n_estimators=100, max_depth=100, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)

    # Compute explained variance
    pca = pipeline.named_steps['pca']
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)

    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

    auc_scores = roc_auc_score(y_test_bin, y_pred_proba, average=None)
    macro_auc = roc_auc_score(y_test_bin, y_pred_proba, average="macro")

    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    cv_scores = cross_val_score(pipeline, X, y, cv=5)

    return {
        'pipeline': pipeline,
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance_ratio': cumulative_variance_ratio,
        'classification_report': report,
        'confusion_matrix': conf_matrix,
        'cv_scores': cv_scores,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'auc_scores': auc_scores,
        'macro_auc': macro_auc
    }

if __name__ == "__main__":
    file_path = 'Train_and_Validate_EEG.csv'
    df = pd.read_csv(file_path)
    df = df.drop(columns=["ID","Unnamed: 122", "eeg.date", "specific.disorder"], errors="ignore")

    df = df.dropna()

    # Initialize MinMaxScaler
    minmax_scaler = MinMaxScaler()
    numerical_cols = df.select_dtypes(include=['number']).columns

    df[numerical_cols] = minmax_scaler.fit_transform(df[numerical_cols])
    df['sex'] = df['sex'].map({'M': 1, 'F': 0})
    target_col = 'main.disorder'
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)


        
    # Run PCA pipeline
    for n_c in [0.90, 0.95, 0.99]:
        results = create_pca_pipeline(X, y, n_components=n_c)

        
        print("Classification Report:\n", results['classification_report'])
        print("\nCross-validation scores:", results['cv_scores'])
        print("Mean CV score:", results['cv_scores'].mean())
        print("\nMacro-Average AUC:", results['macro_auc'])

        auc_df = pd.DataFrame({"Class": label_encoder.classes_, "AUC Score": results['auc_scores']})

        # Plot Explained Variance
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(results['cumulative_variance_ratio']) + 1), results['cumulative_variance_ratio'], 'bo-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('Explained Variance Ratio vs. Number of Components')
        plt.grid(True)
        plt.show()
