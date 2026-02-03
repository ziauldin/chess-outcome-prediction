"""
Prediction module for chess game outcomes.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle
from pathlib import Path


def load_model(model_path):
    """
    Load a trained LightGBM model from disk.
    
    Args:
        model_path: Path to saved model file (.pkl)
        
    Returns:
        Loaded LightGBM Booster model
    """
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def load_metadata(metadata_path):
    """
    Load model metadata (encoders, feature columns, etc.)
    
    Args:
        metadata_path: Path to metadata file
        
    Returns:
        Metadata dictionary
    """
    with open(metadata_path, 'rb') as f:
        return pickle.load(f)


def predict_outcomes(model, X_test):
    """
    Predict outcomes for test games.
    
    Args:
        model: Trained LightGBM model
        X_test: Test features (DataFrame)
        
    Returns:
        Tuple of (predicted_classes, predicted_probabilities)
    """
    # Get probabilities for each class
    y_pred_proba = model.predict(X_test)
    
    # Get predicted class (0, 1, or 2)
    y_pred_class = np.argmax(y_pred_proba, axis=1)
    
    # Convert class labels back to game results (0.0, 0.5, 1.0)
    label_to_result = {0: 0.0, 1: 0.5, 2: 1.0}
    y_pred_result = np.array([label_to_result[c] for c in y_pred_class])
    
    return y_pred_result, y_pred_proba


def make_predictions(test_data_dir, model_path, metadata_path):
    """
    Generate predictions for test data.
    
    Args:
        test_data_dir: Directory containing test tournament JSON files
        model_path: Path to trained model
        metadata_path: Path to metadata file
        
    Returns:
        DataFrame with predictions
    """
    from features import load_tournament_data
    
    print("\n" + "="*70)
    print("GENERATING PREDICTIONS")
    print("="*70)
    
    # Load model and metadata
    print(f"\nLoading model from {model_path}...")
    model = load_model(model_path)
    metadata = load_metadata(metadata_path)
    
    encoders = metadata['encoders']
    time_control_columns = metadata['time_control_columns']
    feature_columns = metadata['feature_columns']
    
    # Load test data
    df_test = load_tournament_data(test_data_dir)
    print(f"Loaded {len(df_test)} test games")
    
    # Encode features using saved encoders
    print("\nEncoding test features...")
    X_test = pd.DataFrame()
    
    # Time control one-hot
    tc_dummies = pd.get_dummies(df_test['time_control'], prefix='tc')
    for col in time_control_columns:
        X_test[col] = tc_dummies[col] if col in tc_dummies.columns else 0
    
    # Numeric features
    X_test['tours'] = df_test['tours'].values
    X_test['round_index'] = df_test['round_index'].values
    
    # ELO rating features (Phase 2)
    X_test['white_elo'] = df_test['white_elo'].values
    X_test['black_elo'] = df_test['black_elo'].values
    X_test['elo_diff'] = df_test['elo_diff'].values
    X_test['elo_avg'] = df_test['elo_avg'].values
    
    # Performance features (Phase 2)
    X_test['white_prev_performance'] = df_test['white_prev_performance'].values
    X_test['black_prev_performance'] = df_test['black_prev_performance'].values
    X_test['white_perf_diff'] = df_test['white_perf_diff'].values
    X_test['black_perf_diff'] = df_test['black_perf_diff'].values
    
    # Head-to-head features (Phase 3)
    X_test['h2h_games_played'] = df_test['h2h_games_played'].values
    X_test['h2h_white_win_rate'] = df_test['h2h_white_win_rate'].values
    X_test['h2h_last_result'] = df_test['h2h_last_result'].values
    
    # Recent form features (Phase 3)
    X_test['white_form_last5'] = df_test['white_form_last5'].values
    X_test['black_form_last5'] = df_test['black_form_last5'].values
    X_test['white_form_last10'] = df_test['white_form_last10'].values
    X_test['black_form_last10'] = df_test['black_form_last10'].values
    X_test['white_win_streak'] = df_test['white_win_streak'].values
    X_test['black_win_streak'] = df_test['black_win_streak'].values
    
    # Encoded player names
    X_test['white_encoded'] = encoders['white'].transform(df_test['white'])
    X_test['black_encoded'] = encoders['black'].transform(df_test['black'])
    
    # Ensure column order matches training
    X_test = X_test[feature_columns]
    
    print(f"Test features shape: {X_test.shape}")
    
    # Make predictions
    print("\nGenerating predictions...")
    predictions, probabilities = predict_outcomes(model, X_test)
    
    # Add predictions to dataframe
    df_test['predicted_result'] = predictions
    df_test['prob_black_wins'] = probabilities[:, 0]
    df_test['prob_draw'] = probabilities[:, 1]
    df_test['prob_white_wins'] = probabilities[:, 2]
    
    print(f"✓ Generated {len(predictions)} predictions")
    print(f"\nPrediction distribution:")
    print(f"  Black wins (0.0): {(predictions == 0.0).sum()} ({(predictions == 0.0).mean()*100:.1f}%)")
    print(f"  Draws (0.5):      {(predictions == 0.5).sum()} ({(predictions == 0.5).mean()*100:.1f}%)")
    print(f"  White wins (1.0): {(predictions == 1.0).sum()} ({(predictions == 1.0).mean()*100:.1f}%)")
    
    return df_test
