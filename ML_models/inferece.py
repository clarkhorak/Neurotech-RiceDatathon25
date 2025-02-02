import joblib
import json
import pandas as pd


knn_imputer = joblib.load("knn_imputer.pkl")
minmax_scaler = joblib.load('minmax_scaler.pkl')

label_encoder = joblib.load('label_encoder.pkl')
model = joblib.load('best_extratrees_model.pkl')
with open('columns_for_imputation.json', 'r') as file:
    columns_for_imputation = json.load(file)

test_df = pd.read_csv('Test_Set_EEG.csv')
test_df = test_df.drop(columns=["ID","Unnamed: 120","eeg.date",], errors="ignore")
test_df['sex'] = test_df['sex'].map({'M': 1, 'F': 0})


test_df_imputed = pd.DataFrame(knn_imputer.transform(test_df), columns=test_df.columns)
numerical_cols = test_df_imputed.select_dtypes(include=['number']).columns
test_df_imputed[numerical_cols] = minmax_scaler.transform(test_df_imputed[numerical_cols])

valid_columns_for_imputation = [col for col in columns_for_imputation if col in test_df_imputed.columns]
inference_input = test_df_imputed[valid_columns_for_imputation]
inference_pred = model.predict(inference_input)
original_labels = label_encoder.inverse_transform(inference_pred)

submission_df = pd.read_csv('Test_Set_EEG.csv')
submission_df['main.disorder'] = original_labels
submission_df.to_csv('Submission.csv', index=False)