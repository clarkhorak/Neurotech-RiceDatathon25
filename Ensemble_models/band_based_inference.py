import numpy as np
import joblib
import pandas as pd
import json
from glob import glob
from sklearn.metrics import roc_auc_score
from itertools import combinations

def predict_on_test_set(model_dir, test_data, bands_list, ensemble_method='average'):
    """
    Make predictions on a new test set using the ensemble of models
    
    Parameters:
    - model_dir: directory containing saved models
    - test_data: DataFrame arranged similarly to df_sorted
    - bands_list: list of band combinations used in training
    - ensemble_method: 'average', 'weighted', or 'max_confidence'
    
    Returns:
    - ensemble_predictions: final predictions from the ensemble
    - individual_predictions: dictionary of predictions from each model
    """
    individual_predictions = {}
    all_probas = []
    
    for bands in bands_list:
        # Load the model
        fn = "_".join(bands)
        model_filename = f"{model_dir}/{fn}.pkl"
        model = joblib.load(model_filename)
        
        # Get features for this band combination
        demographic_features = [col for col in test_data.columns if not (col.startswith("AB") or col.startswith("COH"))]
        feature_list = [col for col in test_data.columns if col.startswith("AB") or col.startswith("COH")]
        
        sorted_features = []
        for band in bands:
            ab_features = sorted([col for col in feature_list if col.startswith("AB") and band in col])
            coh_features = sorted([col for col in feature_list if col.startswith("COH") and band in col])
            sorted_features.extend(ab_features + coh_features)
            
        selected_features = demographic_features + sorted_features
        X_test = test_data[selected_features]
        
        # Get predictions
        pred_proba = model.predict_proba(X_test)
        individual_predictions[fn] = pred_proba
        all_probas.append(pred_proba)
    
    # Convert to numpy array for easier manipulation
    predictions_array = np.array(all_probas)
    n_classes = predictions_array.shape[2]
    
    # Apply ensemble method
    if ensemble_method == 'average':
        ensemble_predictions = np.mean(predictions_array, axis=0)
    
    elif ensemble_method == 'max_confidence':
        confidences = predictions_array.max(axis=2)
        best_model_idx = confidences.argmax(axis=0)
        ensemble_predictions = np.zeros((len(test_data), n_classes))
        for i in range(len(test_data)):
            ensemble_predictions[i] = predictions_array[best_model_idx[i], i]
    
    # Note: Weighted method isn't available for new test data without true labels
    
    return ensemble_predictions, individual_predictions

def get_class_predictions(probabilities):
    """Convert probabilities to class predictions"""
    return np.argmax(probabilities, axis=1)

# Example usage:
def make_test_predictions(model_dir, test_data):
    """
    Make predictions on test data using different ensemble methods
    
    Returns:
    - Dictionary containing predictions for each ensemble method
    """
    # Generate band combinations (matching your training)
    bands = ["delta", "theta", "alpha", "beta", "highbeta", "gamma"]
    all_combinations = []
    for k in range(2, 7):
        all_combinations.extend(combinations(bands, k))
    
    results = {}
    
    # Get predictions using different ensemble methods
    for method in ['average', 'max_confidence']:
        ensemble_probs, individual_probs = predict_on_test_set(
            model_dir, 
            test_data, 
            all_combinations,
            ensemble_method=method
        )
        
        results[method] = {
            'probabilities': ensemble_probs,
            'predictions': get_class_predictions(ensemble_probs),
            'individual_model_probabilities': individual_probs
        }
    
    return results

if __name__ == "__main__":
    test_data = pd.read_csv("Test_Set_EEG.csv")
    with open("df_sorted_columns.json", "r") as f:
        saved_columns = json.load(f)
        
    minmax_scaler = joblib.load("minmax_scaler.pkl")
    
    predictions = make_test_predictions(
        "SIngle_models/models", 
        test_data
    )
    
    average_predictions = predictions['average']['predictions']
    average_probabilities = predictions['average']['probabilities']

    # Get predictions from max confidence ensemble:
    max_conf_predictions = predictions['max_confidence']['predictions']
    max_conf_probabilities = predictions['max_confidence']['probabilities']

    # If you want predictions from a specific model:
    model_name = "delta_theta_alpha"  # example band combination
    individual_model_probs = predictions['average']['individual_model_probabilities'][model_name]
    
    test_data['predictions'] = average_predictions
    print(test_data)