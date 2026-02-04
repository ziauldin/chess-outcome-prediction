"""
Main pipeline script for chess game prediction.

=================================================================
PHASE 4: HYPERPARAMETER TUNING FOR LIGHTGBM (Feb 2026)
=================================================================

PIPELINE OVERVIEW:
Orchestrates end-to-end ML workflow with hyperparameter tuning.
Automatically finds best LightGBM parameters before training.

EXECUTION STEPS:
1. Load tournament data (174K games, 2133 players)
2. Engineer features (Phase 1 + 2 + 3 = 23 total features)
3. Time-based train/val split (80/20, prevents leakage)
4. TUNE LightGBM hyperparameters (NEW in Phase 4!)
5. Train LightGBM with best parameters
6. Evaluate & analyze results
7. Generate predictions for test data
8. Update test JSON files with predictions

=================================================================
PHASE 4 IMPROVEMENTS
=================================================================

HYPERPARAMETER TUNING:
  Method: RandomizedSearchCV (efficient, not exhaustive)
  Parameters: 5 most impactful for LightGBM
  Iterations: 50 random combinations
  CV Folds: 5-fold cross-validation
  Time: ~45 minutes

PARAMETERS TUNED:
  1. learning_rate (0.01-0.15): Convergence speed
  2. max_depth (5-15): Model complexity
  3. num_leaves (20-100): Model capacity
  4. min_child_samples (5-30): Regularization
  5. subsample (0.5-1.0): Overfitting prevention

BASELINE (Phase 3) → TARGET (Phase 4):
  Accuracy: 52.02% → 53.5-54.5% (+1.5-2.5pp)
  F1 Score: 0.5110 → 0.5250-0.5400

=================================================================
PHASE PROGRESSION & ACCURACY:
=================================================================

PHASE 1 (Baseline - Jan 2026):
  Features: 6 | Accuracy: ~46-47%
  Components: Tournament context + player identity

PHASE 2 (ELO + Performance - Feb 2026):
  Features: 14 | Accuracy: 50.20%
  Components: ELO ratings + historical performance

PHASE 3 (Derived Features - Feb 2026):
  Features: 23 | Accuracy: 52.02%
  Components: Head-to-head + recent form + momentum

PHASE 4 (Hyperparameter Tuning - Feb 2026 - CURRENT):
  Parameters: 50 random combinations tested
  Accuracy: 53.5-54.5% (TARGET) ✨
  Components: Tuned LightGBM + all Phase 3 features
"""

import sys
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.features import load_tournament_data, encode_features, prepare_labels
from src.train import train_model, evaluate_model
from src.predict import make_predictions
from src.update_test_jsons import update_test_jsons


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def run_pipeline():
    """
    Execute the complete ML pipeline with hyperparameter tuning.
    """
    print("="*70)
    print("CHESS GAME PREDICTION PIPELINE - PHASE 4")
    print("HYPERPARAMETER TUNING FOR LIGHTGBM")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Load data
    print("\n[1/8] Loading data...")
    df = load_tournament_data('data/train')
    print(f"  Total games: {len(df)}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Unique players: {len(set(df['white']) | set(df['black']))}")
    
    # 2. Create time-based train/validation split
    print("\n[2/8] Creating train/validation split...")
    df = df.sort_values('date').reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    
    df_train = df.iloc[:split_idx].copy()
    df_val = df.iloc[split_idx:].copy()
    
    print(f"  Train: {len(df_train)} games ({df_train['date'].min().date()} to {df_train['date'].max().date()})")
    print(f"  Val:   {len(df_val)} games ({df_val['date'].min().date()} to {df_val['date'].max().date()})")
    
    # Verify no data leakage
    assert df_train['date'].max() <= df_val['date'].min(), "Data leakage detected!"
    print("  ✓ No data leakage")
    
    # 3. Feature engineering
    print("\n[3/8] Engineering features...")
    X_train, X_val, _, encoders, tc_columns = encode_features(df_train, df_val)
    y_train = prepare_labels(df_train)
    y_val = prepare_labels(df_val)
    
    print(f"  Class distribution (train):")
    print(f"    Black wins (0): {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.1f}%)")
    print(f"    Draws (1):      {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")
    print(f"    White wins (2): {(y_train == 2).sum()} ({(y_train == 2).mean()*100:.1f}%)")
    
    # 4. Train model with hyperparameter tuning
    print("\n[4/8] Training model with hyperparameter tuning...")
    model, tuning_results = train_model(X_train, y_train, X_val, y_val, tune_hyperparameters=True)
    
    # 5. Evaluate model
    print("\n[5/8] Evaluating model...")
    metrics = evaluate_model(model, X_train, y_train, X_val, y_val, tuning_results)
    
    # 6. Save artifacts
    print("\n[6/8] Saving model artifacts...")
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Create model filename with algorithm name and accuracies
    train_acc = metrics['train_accuracy']
    val_acc = metrics['val_accuracy']
    model_filename = f"lightgbm_tuned_{train_acc:.4f}_{val_acc:.4f}.pkl"
    model_path = models_dir / model_filename
    
    # Save model using pickle
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  ✓ Saved model to {model_path}")
    
    # Save metadata with matching filename
    metadata_filename = f"lightgbm_tuned_{train_acc:.4f}_{val_acc:.4f}_metadata.pkl"
    metadata_path = models_dir / metadata_filename
    
    metadata = {
        'encoders': encoders,
        'time_control_columns': tc_columns,
        'feature_columns': X_train.columns.tolist(),
        'label_map': {0: 0.0, 1: 0.5, 2: 1.0},
        'metrics': metrics,
        'random_seed': RANDOM_SEED,
        'train_date_range': (str(df_train['date'].min()), str(df_train['date'].max())),
        'val_date_range': (str(df_val['date'].min()), str(df_val['date'].max())),
        'phase': 4,
        'tuning_results': tuning_results if tuning_results else None
    }
    
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"  ✓ Saved metadata to {metadata_path}")
    
    # Save metrics to JSON
    outputs_dir = Path('outputs')
    outputs_dir.mkdir(exist_ok=True)
    
    metrics_output = {
        'phase': 4,
        'model': 'lightgbm_tuned',
        'validation_accuracy': float(metrics['val_accuracy']),
        'validation_f1': float(metrics['val_f1']),
        'validation_logloss': float(metrics['val_logloss']),
        'train_accuracy': float(metrics['train_accuracy']),
        'train_f1': float(metrics['train_f1']),
        'train_logloss': float(metrics['train_logloss']),
        'confusion_matrix': metrics['confusion_matrix'],
        'feature_importance': metrics['feature_importance'][:10],
        'tuning_results': {
            'best_params': tuning_results['best_params'] if tuning_results else None,
            'improvement_pp': float(tuning_results['improvement_pp']) if tuning_results else None,
            'n_iterations': tuning_results['n_iterations'] if tuning_results else None
        } if tuning_results else None
    }
    
    metrics_json_path = outputs_dir / 'model_metrics.json'
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_output, f, indent=2)
    print(f"  ✓ Saved metrics to {metrics_json_path}")
    
    # 7. Generate predictions and update test JSONs
    print("\n[7/8] Generating predictions for test data...")
    test_dir = 'data/test'
    df_predictions = make_predictions(test_dir, str(model_path), str(metadata_path))
    
    # Update test JSON files
    # 8. Update test JSON files
    print("\n[8/8] Updating test JSON files...")
    update_test_jsons(df_predictions, test_dir)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE - PHASE 4")
    print("="*70)
    print(f"\n📊 FINAL RESULTS:")
    print(f"  Train Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"  Val Accuracy:   {metrics['val_accuracy']:.4f} ✨")
    print(f"  Val F1:         {metrics['val_f1']:.4f}")
    
    if tuning_results:
        print(f"\n🎯 TUNING IMPROVEMENT: {tuning_results['improvement_pp']:+.2f}pp")
        print(f"   From Phase 3: 52.02% → {metrics['val_accuracy']:.2%}")
    
    print(f"\n✓ Model files saved in {models_dir}/")
    print(f"✓ Metrics saved in {outputs_dir}/")
    print(f"✓ Test predictions saved in {test_dir}/")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    run_pipeline()
