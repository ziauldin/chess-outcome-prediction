"""
Main pipeline script for chess game prediction.

=================================================================
PHASE 3: DERIVED FEATURES - HEAD-TO-HEAD & RECENT FORM (Feb 2026)
=================================================================

PIPELINE OVERVIEW:
Orchestrates end-to-end ML workflow predicting chess game outcomes.
Implements all 3 phases of feature engineering and model development.

EXECUTION STEPS:
1. Load tournament data (174K games, 2133 players)
2. Engineer features (Phase 1 + 2 + 3 = 23 total features)
3. Time-based train/val split (80/20, prevents leakage)
4. Train LightGBM classifier
5. Evaluate & analyze feature importance
6. Generate predictions for test data
7. Update test JSON files with predictions

=================================================================
PHASE PROGRESSION & ACCURACY:
=================================================================

PHASE 1 (Baseline - Jan 2026):
  Features: 6 | Accuracy: ~46-47%
  Components: Tournament context + player identity

PHASE 2 (ELO + Performance - Feb 2026):
  Features: 14 | Accuracy: 50.20%
  Components: ELO ratings + historical performance tracking

PHASE 3 (Derived Features - Feb 2026 - CURRENT):
  Features: 23 | Accuracy: 52.02% ✅ +1.82 pp improvement!
  Components: Head-to-head matchups + recent form/momentum

=================================================================
PHASE 3: NEW FEATURES (9 TOTAL)
=================================================================

HEAD-TO-HEAD (3):
  - h2h_games_played: Previous matchups between players
  - h2h_white_win_rate: Win rate in this specific matchup
  - h2h_last_result: Outcome of most recent game

RECENT FORM (6):
  - white/black_form_last5: Last 5 games performance
  - white/black_form_last10: Last 10 games performance (TOP FEATURE!)
  - white/black_win_streak: Current win/loss streak

IMPACT: form_last10 ranks #3 overall in feature importance!

=================================================================
TOP 5 FEATURES (Phase 3)
=================================================================
1. black_prev_performance (51,506)
2. white_prev_performance (50,308)
3. white_form_last10 (35,576) ← PHASE 3 NEW!
4. black_form_last10 (30,817) ← PHASE 3 NEW!
5. tc_classic (27,276)

KEY INSIGHT: Recent momentum is HIGHLY predictive!
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
