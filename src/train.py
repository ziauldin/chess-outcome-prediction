"""
Training module for chess game prediction models.

=================================================================
PHASE 4: HYPERPARAMETER TUNING (Feb 2026)
=================================================================

Now includes hyperparameter tuning for LightGBM.
Automatically finds best parameters before training final model.

CHANGES FROM PHASE 3:
  - Added import for tuning module
  - train_model() now tunes hyperparameters first
  - Uses tuned parameters for final model
  - Returns tuning results in metrics dictionary

EXPECTED IMPROVEMENT:
  Phase 3: 52.02% accuracy
  Phase 4: 53.5-54.5% accuracy
"""

import json
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix, classification_report
import lightgbm as lgb
from src.hyperparameter_tuning import tune_lightgbm_quick


RANDOM_SEED = 42


def train_model(X_train, y_train, X_val, y_val, tune_hyperparameters=True):
    """
    Train LightGBM model with optional hyperparameter tuning.
    
    PHASE 4 UPDATE: Now includes hyperparameter tuning by default
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        tune_hyperparameters: If True, run hyperparameter tuning (default: True)
    
    Returns:
        model: Trained LightGBM model
        tuning_results: Results from hyperparameter tuning (if performed)
    """
    
    tuning_results = None
    
    if tune_hyperparameters:
        print("\n[4.1/8] PHASE 4: Tuning hyperparameters...")
        model, best_params, best_cv_score, tuning_results = tune_lightgbm_quick(
            X_train, y_train, X_val, y_val,
            n_iter=50,  # Test 50 random combinations
            cv=5        # 5-fold cross-validation
        )
    else:
        # Use baseline parameters (Phase 3)
        print("\n[4/8] Training LightGBM model (baseline parameters)...")
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Model parameters
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': ['multi_logloss', 'multi_error'],
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': RANDOM_SEED,
            'n_jobs': -1
        }
        
        # Train
        callbacks = [
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=callbacks
        )
    
    return model, tuning_results


def evaluate_model(model, X_train, y_train, X_val, y_val, tuning_results=None):
    """
    Evaluate model performance.
    
    PHASE 4 UPDATE: Includes tuning results in output
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        tuning_results: Optional tuning results dictionary
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("\n[5/8] Evaluating model...")
    
    # Get predictions (class labels directly)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Get probabilities (for logloss)
    y_train_proba = model.predict_proba(X_train)
    y_val_proba = model.predict_proba(X_val)
    
    # Ensure proba is 2D (handle edge case)
    if y_train_proba.ndim == 1:
        y_train_proba = y_train_proba.reshape(-1, 1)
    if y_val_proba.ndim == 1:
        y_val_proba = y_val_proba.reshape(-1, 1)
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')
    
    train_logloss = log_loss(y_train, y_train_proba)
    val_logloss = log_loss(y_val, y_val_proba)
    
    print("\nMETRICS:")
    print(f"  Training Accuracy:   {train_acc:.4f}")
    print(f"  Validation Accuracy: {val_acc:.4f}")
    print(f"  Training F1 (weighted): {train_f1:.4f}")
    print(f"  Validation F1 (weighted): {val_f1:.4f}")
    print(f"  Training Log Loss:   {train_logloss:.4f}")
    print(f"  Validation Log Loss: {val_logloss:.4f}")
    
    if tuning_results:
        print(f"  Tuning Improvement:  {tuning_results['improvement_pp']:+.2f}pp")
    
    # Feature importance
    feature_importance = sorted(
        zip(model.feature_name_, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )
    
    print(f"\n  Top 5 Features:")
    for i, (feature, importance) in enumerate(feature_importance[:5], 1):
        print(f"    {i}. {feature}: {importance:.0f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_val_pred)
    cm_list = cm.tolist()
    
    metrics = {
        'train_accuracy': float(train_acc),
        'val_accuracy': float(val_acc),
        'train_f1': float(train_f1),
        'val_f1': float(val_f1),
        'train_logloss': float(train_logloss),
        'val_logloss': float(val_logloss),
        'confusion_matrix': cm_list,
        'feature_importance': [(name, float(imp)) for name, imp in feature_importance],
        'tuning_results': tuning_results
    }
    
    return metrics


def main():
    """Main training pipeline - standalone execution."""
    from features import load_tournament_data, encode_features, prepare_labels
    
    print("="*70)
    print("BASELINE CHESS OUTCOME PREDICTOR")
    print("Raw Features Only - No History, No Aggregations")
    print("="*70)
    
    # Load data
    df = load_tournament_data('data/train')
    
    print(f"\nDataset info:")
    print(f"  Total games: {len(df)}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Unique players: {len(set(df['white']) | set(df['black']))}")
    
    # Time-based split (80/20)
    print("\n" + "="*70)
    print("Creating Time-Based Train/Validation Split")
    print("="*70)
    
    df = df.sort_values('date').reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    
    df_train = df.iloc[:split_idx].copy()
    df_val = df.iloc[split_idx:].copy()
    
    print(f"\nTrain: {len(df_train)} games ({df_train['date'].min()} to {df_train['date'].max()})")
    print(f"Val:   {len(df_val)} games ({df_val['date'].min()} to {df_val['date'].max()})")
    
    # Verify no data leakage
    assert df_train['date'].max() <= df_val['date'].min(), "Data leakage detected!"
    print("✓ No data leakage - validation games are strictly after training games")
    
    # Encode features
    X_train, X_val, _, encoders, time_control_columns = encode_features(df_train, df_val)
    y_train = prepare_labels(df_train)
    y_val = prepare_labels(df_val)
    
    print(f"\nClass distribution (train):")
    print(f"  Black wins (0): {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.1f}%)")
    print(f"  Draws (1):      {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")
    print(f"  White wins (2): {(y_train == 2).sum()} ({(y_train == 2).mean()*100:.1f}%)")
    
    # Train model
    model, tuning_results = train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate
    metrics = evaluate_model(model, X_train, y_train, X_val, y_val, tuning_results)
    
    # Save artifacts
    print("\n" + "="*70)
    print("Saving Model Artifacts")
    print("="*70)
    
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Create model filename with algorithm name and accuracies
    train_acc = metrics['train_accuracy']
    val_acc = metrics['val_accuracy']
    model_filename = f"lightgbm_{train_acc:.4f}_{val_acc:.4f}.pkl"
    model_path = models_dir / model_filename
    
    # Save model using pickle
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Saved model to {model_path}")
    
    # Save encoders and metadata with matching filename
    metadata_filename = f"lightgbm_{train_acc:.4f}_{val_acc:.4f}_metadata.pkl"
    metadata_path = models_dir / metadata_filename
    
    metadata = {
        'encoders': encoders,
        'time_control_columns': time_control_columns,
        'feature_columns': X_train.columns.tolist(),
        'label_map': {0: 0.0, 1: 0.5, 2: 1.0},
        'metrics': metrics,
        'random_seed': RANDOM_SEED
    }
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✓ Saved metadata to {metadata_path}")
    
    print("\n" + "="*70)
    print("BASELINE MODEL TRAINING COMPLETE")
    print("="*70)
    print(f"\n📊 Final Validation Accuracy: {metrics['val_accuracy']:.4f}")
    print(f"📊 Final Validation F1:       {metrics['val_f1']:.4f}")
    print("\nModel files saved in models/")


if __name__ == '__main__':
    main()
