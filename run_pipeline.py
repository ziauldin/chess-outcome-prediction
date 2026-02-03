"""
Main pipeline script for chess game prediction.

PIPELINE OVERVIEW:
This script orchestrates the complete end-to-end ML workflow for predicting
chess game outcomes (Black wins / Draw / White wins).

EXECUTION STEPS:
1. Load and preprocess tournament data from JSON files
2. Engineer features (Phase 1 baseline + Phase 2 ELO/performance features)
3. Train LightGBM multiclass classifier
4. Evaluate model performance on validation set
5. Generate predictions for test tournaments
6. Update test JSON files with predicted results

PHASE 1 (Baseline Model):
- Basic features: time_control, tours, round_index, player names
- Simple label encoding for categorical variables
- Achieved ~46-47% validation accuracy

PHASE 2 (Enhanced Features - Current):
- Added ELO rating features (white_elo, black_elo, elo_diff, elo_avg)
- Added performance tracking (prev_performance, perf_diff for both players)
- Improved feature importance insights
- Achieved ~50% validation accuracy

MODEL: LightGBM (Gradient Boosting Decision Trees)
TASK: 3-class classification (0.0=Black wins, 0.5=Draw, 1.0=White wins)
VALIDATION: Time-based 80/20 split (prevents data leakage)
"""

import sys
import json
import pickle
import numpy as np
from pathlib import Path

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
    Execute the complete ML pipeline.
    """
    print("="*70)
    print("CHESS GAME PREDICTION PIPELINE")
    print("="*70)
    
    # 1. Load data
    print("\n[1/7] Loading data...")
    df = load_tournament_data('data/train')
    print(f"  Total games: {len(df)}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Unique players: {len(set(df['white']) | set(df['black']))}")
    
    # 2. Create time-based train/validation split
    print("\n[2/7] Creating train/validation split...")
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
    print("\n[3/7] Engineering features...")
    X_train, X_val, _, encoders, tc_columns = encode_features(df_train, df_val)
    y_train = prepare_labels(df_train)
    y_val = prepare_labels(df_val)
    
    print(f"  Class distribution (train):")
    print(f"    Black wins (0): {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.1f}%)")
    print(f"    Draws (1):      {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")
    print(f"    White wins (2): {(y_train == 2).sum()} ({(y_train == 2).mean()*100:.1f}%)")
    
    # 4. Train model
    print("\n[4/7] Training model...")
    model = train_model(X_train, y_train, X_val, y_val)
    
    # 5. Evaluate model
    print("\n[5/7] Evaluating model...")
    metrics = evaluate_model(model, X_train, y_train, X_val, y_val)
    
    # 6. Save artifacts
    print("\n[6/7] Saving model artifacts...")
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
    print(f"  ✓ Saved model to {model_path}")
    
    # Save metadata with matching filename
    metadata_filename = f"lightgbm_{train_acc:.4f}_{val_acc:.4f}_metadata.pkl"
    metadata_path = models_dir / metadata_filename
    
    metadata = {
        'encoders': encoders,
        'time_control_columns': tc_columns,
        'feature_columns': X_train.columns.tolist(),
        'label_map': {0: 0.0, 1: 0.5, 2: 1.0},
        'metrics': metrics,
        'random_seed': RANDOM_SEED,
        'train_date_range': (str(df_train['date'].min()), str(df_train['date'].max())),
        'val_date_range': (str(df_val['date'].min()), str(df_val['date'].max()))
    }
    
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"  ✓ Saved metadata to {metadata_path}")
    
    # Save metrics to JSON
    outputs_dir = Path('outputs')
    outputs_dir.mkdir(exist_ok=True)
    
    metrics_output = {
        'model': 'raw_baseline',
        'validation_accuracy': float(metrics['val_accuracy']),
        'validation_f1': float(metrics['val_f1']),
        'validation_logloss': float(metrics['val_logloss']),
        'train_accuracy': float(metrics['train_accuracy']),
        'train_f1': float(metrics['train_f1']),
        'train_logloss': float(metrics['train_logloss']),
        'confusion_matrix': metrics['confusion_matrix'],
        'feature_importance': metrics['feature_importance'][:10]
    }
    
    metrics_json_path = outputs_dir / 'model_metrics.json'
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_output, f, indent=2)
    print(f"  ✓ Saved metrics to {metrics_json_path}")
    
    # 7. Generate predictions and update test JSONs
    print("\n[7/7] Generating predictions for test data...")
    test_dir = 'data/test'
    df_predictions = make_predictions(test_dir, str(model_path), str(metadata_path))
    
    # Update test JSON files
    update_test_jsons(df_predictions, test_dir)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"\n📊 Final Validation Accuracy: {metrics['val_accuracy']:.4f}")
    print(f"📊 Final Validation F1:       {metrics['val_f1']:.4f}")
    print(f"\n✓ Model files saved in {models_dir}/")
    print(f"✓ Metrics saved in {outputs_dir}/")
    print(f"✓ Test predictions saved in {test_dir}/")


if __name__ == '__main__':
    run_pipeline()
