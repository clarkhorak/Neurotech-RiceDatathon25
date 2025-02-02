import pandas as pd
import joblib
from sklearn.impute import KNNImputer

file_path = 'Train_and_Validate_EEG.csv'
df = pd.read_csv(file_path)

# Save the dropped columns
dropped_columns = df[["Unnamed: 122", "eeg.date", "specific.disorder", "main.disorder"]]

df = df.drop(columns=["Unnamed: 122", "eeg.date", "specific.disorder", "main.disorder"], errors="ignore")

knn_imputer = KNNImputer(n_neighbors=5)

df['sex'] = df['sex'].map({'M': 1, 'F': 0})

df_imputed = pd.DataFrame(knn_imputer.fit_transform(df), columns=df.columns)

# Reset index of dropped columns before concatenation
dropped_columns.reset_index(drop=True, inplace=True)
# Concatenate the dropped columns back to the imputed DataFrame
df_imputed = pd.concat([df_imputed, dropped_columns], axis=1)

df_imputed.to_csv('df_imputed.csv', index=False)

# Imputer for test_set
joblib.dump(knn_imputer, "knn_imputer.pkl")
