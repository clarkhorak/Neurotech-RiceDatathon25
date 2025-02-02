# Neurotech-RiceDatathon25
Rice University Datathon 2025 - Neurotech Track

## Description

This project is part of the Neurotech track for the Rice University Datathon 2025. The goal is to analyze EEG data and build machine learning models to classify different neurological disorders.
The Submission.csv file contains the the model predictions for the test set.

## Directory Structure

- **EDA/**: Contains exploratory data analysis scripts and notebooks.
  - `eda.ipynb`: Jupyter notebook for EDA.
  - `Missing_data_imputation.py`: Script for handling missing data.

- **Ensemble_models/**: Contains scripts for training and inference using ensemble models.
  - `band_based_inference.py`: Script for inference using band-based models.
  - `band_based_training.py`: Script for training band-based models.
  - `df_sorted_columns.json`: JSON file with sorted column names.

- **Explainability/**: Contains scripts for model explainability.
  - `shap_explainability.py`: Script for generating SHAP explanations for model predictions.
  - `lime_explainability.py`: Script for generating LIME explanations for model predictions.

- **Feature_Selection/**: Contains scripts for feature selection.
  - `hier_nn.py`: Hierarchical feature selection using neural networks.
  - `hier_xg.py`: Hierarchical feature selection using XGBoost.
  - `models_for_hier.py`: Models used for hierarchical feature selection.
  - `results/`: Directory containing evaluation results.
    - `eval.ipynb`: Jupyter notebook for evaluating feature selection results.

- **Hyperparameter_tuning/**: Contains scripts for hyperparameter tuning.
  - `optuna_ml.py`: Script for hyperparameter tuning using Optuna.
  - `optuna_tune.py`: Script for hyperparameter tuning using Optuna.

- **ML_models/**: Contains scripts for training and inference using machine learning models.
  - `inference.py`: Script for model inference.
  - `model.py`: Script for model training.

- **NN_model/**: Contains scripts for training neural network models.
  - `NN_train.py`: Script for training neural network models.

- **PCA_model/**: Contains scripts for training PCA models.
  - `pca_train.py`: Script for training PCA models.

## Replicating Results

To replicate the results of this project, follow these steps:

1. **Data Preprocessing**:
    - Run the `Missing_data_imputation.py` script to handle missing data.

2. **Feature Selection**:
    - Run the `hier_nn.py` or `hier_xg.py` scripts in the [Feature_Selection](Feature_Selection) directory to perform hierarchical feature selection and select the best threshold.

3. **Model Training**:
    - For machine learning models, run the `model_selection.py` script in the [ML_models](ML_models) directory, to get the best model based on cross-validation.
    - For neural network models, run the `NN_train.py` script in the [NN_model](NN_model) directory.
    - For PCA models, run the `pca_train.py` script in the [PCA_model](PCA_model) directory.

4. **Hyperparameter Tuning**:
    - Run the `optuna_ml.py` or `optuna_tune.py` scripts in the `Hyperparameter_tuning` directory to perform hyperparameter tuning using Optuna.

5. **Model Inference**:
    - Run the `inference.py` script in the [ML_models](ML_models) directory for model inference.

*For testing diffeent models, hyperparameter tuning and feature selection, we dropped the nan values instead of imputing them. But for the final submission model , we imputed the nan values and used the best model for inference.*

## License

This project is licensed under the MIT License.
