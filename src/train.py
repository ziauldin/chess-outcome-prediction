"""
Training module for chess game prediction models.

Baseline chess game outcome predictor using only raw features.

Features used (NO player history, NO rolling stats):
- time_control (classic/rapid) - one-hot encoded
- tours (number of rounds) - numeric
- round_index (round number) - numeric  
- white (player name) - label encoded
- black (player name) - label encoded

No historical aggregations, no head-to-head, no form features.
"""

import json
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix, classification_report
import lightgbm as lgb


RANDOM_SEED = 42


def train_model(X_train, y_train, X_val, y_val):
    """
    Train LightGBM baseline model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        
    Returns:
        Trained LightGBM model
    """
    print("\n" + "="*70)
    print("Training Baseline Model (LightGBM)")
    print("="*70)
    
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
    
    return model


def evaluate_model(model, X_train, y_train, X_val, y_val):
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("\n" + "="*70)
    print("BASELINE MODEL EVALUATION")
    print("="*70)
    
    # Predictions
    y_train_pred_proba = model.predict(X_train)
    y_val_pred_proba = model.predict(X_val)
    
    y_train_pred = np.argmax(y_train_pred_proba, axis=1)
    y_val_pred = np.argmax(y_val_pred_proba, axis=1)
    
    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    
    train_f1 = f1_score(y_train, y_train_pred, average='macro')
    val_f1 = f1_score(y_val, y_val_pred, average='macro')
    
    train_logloss = log_loss(y_train, y_train_pred_proba)
    val_logloss = log_loss(y_val, y_val_pred_proba)
    
    print("\nMETRICS:")
    print(f"  Training Accuracy:   {train_acc:.4f}")
    print(f"  Validation Accuracy: {val_acc:.4f}")
    print(f"  Training F1 (macro): {train_f1:.4f}")
    print(f"  Validation F1 (macro): {val_f1:.4f}")
    print(f"  Training Log Loss:   {train_logloss:.4f}")
    print(f"  Validation Log Loss: {val_logloss:.4f}")
    
    print("\nVALIDATION CLASSIFICATION REPORT:")
    print(classification_report(
        y_val, y_val_pred,
        target_names=['Black Wins (0.0)', 'Draw (0.5)', 'White Wins (1.0)'],
        digits=4
    ))
    
    print("\nCONFUSION MATRIX (Validation):")
    cm = confusion_matrix(y_val, y_val_pred)
    print("                Predicted")
    print("                Black  Draw  White")
    print(f"Actual Black  {cm[0][0]:6d} {cm[0][1]:5d} {cm[0][2]:6d}")
    print(f"       Draw   {cm[1][0]:6d} {cm[1][1]:5d} {cm[1][2]:6d}")
    print(f"       White  {cm[2][0]:6d} {cm[2][1]:5d} {cm[2][2]:6d}")
    
    print("\nFEATURE IMPORTANCE (Top 10):")
    feature_importance = pd.DataFrame({
        'feature': model.feature_name(),
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    print(feature_importance.head(10).to_string(index=False))
    
    return {
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'train_f1': train_f1,
        'val_f1': val_f1,
        'train_logloss': train_logloss,
        'val_logloss': val_logloss,
        'confusion_matrix': cm.tolist(),
        'feature_importance': feature_importance.to_dict('records')
    }


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
    model = train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate
    metrics = evaluate_model(model, X_train, y_train, X_val, y_val)
    
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
