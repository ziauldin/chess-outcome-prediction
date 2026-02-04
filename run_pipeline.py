"""
Main pipeline script for chess game prediction.

=================================================================
PHASE 5: RANDOM FOREST WITH ADVANCED TUNING (Feb 2026)
=================================================================

PIPELINE OVERVIEW:
Implements Random Forest classifier with Bayesian optimization
to maximize accuracy beyond LightGBM's performance.

EXECUTION STEPS:
1. Load tournament data (174K games, 2133 players)
2. Engineer features (Phase 1 + 2 + 3 = 23 total features)
3. Time-based train/val split (80/20, prevents leakage)
4. TUNE Random Forest with Optuna (100 trials, 9 hyperparameters)
5. Train Random Forest with optimized parameters
6. Evaluate & compare vs LightGBM baseline
7. Generate predictions for test data
8. Update test JSON files with predictions

=================================================================
PHASE 5: WHY RANDOM FOREST?
=================================================================

ADVANTAGES OVER GRADIENT BOOSTING:
  ✓ Parallel tree building (faster training)
  ✓ More robust to outliers & noise
  ✓ Natural regularization through averaging
  ✓ Different feature importance patterns
  ✓ Less prone to overfitting on complex data

HYPERPARAMETER TUNING:
  Method: Bayesian Optimization (Optuna + TPE sampler)
  Parameters: 9 key Random Forest hyperparameters
  Trials: 100 intelligent parameter combinations
  Time: ~20-30 minutes

PARAMETERS OPTIMIZED:
  1. n_estimators (200-800): Number of trees
  2. max_depth (15-40): Tree depth control
  3. min_samples_split (2-15): Split threshold
  4. min_samples_leaf (1-8): Leaf size minimum
  5. max_features (sqrt/log2): Feature sampling
  6. class_weight (balanced/balanced_subsample/None): Handle imbalance
  7. min_impurity_decrease (0.0-0.001): Split quality
  8. max_samples (0.7-1.0): Bootstrap sampling rate
  9. bootstrap (True): Ensemble diversity

COMPARISON (LightGBM Phase 4 vs Random Forest Phase 5):
  LightGBM: 52.11% validation accuracy
  Target:   52.5-53.5% validation accuracy (+0.4-1.4pp)

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

PHASE 4 (Hyperparameter Tuning - Feb 2026):
  Parameters: 50 Bayesian trials for LightGBM
  Accuracy: 52.11% (+0.09pp from Phase 3)
  Components: Optuna-tuned LightGBM

PHASE 5 (Random Forest - Feb 2026 - CURRENT):
  Parameters: 100 Bayesian trials for Random Forest
  Accuracy: 52.5-53.5% (TARGET) ✨
  Components: Optuna-tuned Random Forest + all Phase 3 features
"""

import sys
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.features import load_tournament_data, encode_features, prepare_labels
from src.models.random_forest import tune_random_forest_optuna, train_random_forest_baseline
from src.predict import make_predictions
from src.update_test_jsons import update_test_jsons


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def run_pipeline():
    """
    Execute the complete ML pipeline with Random Forest optimization.
    """
    print("="*70)
    print("CHESS GAME PREDICTION PIPELINE - PHASE 5")
    print("RANDOM FOREST WITH BAYESIAN OPTIMIZATION")
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
    
    # 4. Tune Random Forest hyperparameters
    print("\n[4/8] Tuning Random Forest hyperparameters...")
    best_params, best_val_acc = tune_random_forest_optuna(
        X_train.values, y_train, X_val.values, y_val, n_trials=100
    )
    
    # Train baseline for comparison
    print("\n[4b/8] Training baseline Random Forest...")
    baseline_model = train_random_forest_baseline(X_train.values, y_train)
    baseline_val_preds = baseline_model.predict(X_val.values)
    baseline_val_acc = accuracy_score(y_val, baseline_val_preds)
    print(f"  Baseline validation accuracy: {baseline_val_acc:.4f}")
    
    # Train optimized model
    print("\n[4c/8] Training optimized Random Forest...")
    model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    model.fit(X_train.values, y_train)
    
    # Store tuning results
    tuning_results = {
        'best_params': best_params,
        'best_val_accuracy': float(best_val_acc),
        'baseline_val_accuracy': float(baseline_val_acc),
        'improvement_pp': float((best_val_acc - baseline_val_acc) * 100),
        'n_trials': 100
    }
    
    # 5. Evaluate model
    print("\n[5/8] Evaluating model...")
    
    # Training metrics
    train_preds = model.predict(X_train.values)
    train_proba = model.predict_proba(X_train.values)
    train_acc = accuracy_score(y_train, train_preds)
    train_f1 = f1_score(y_train, train_preds, average='weighted')
    train_logloss = log_loss(y_train, train_proba)
    
    # Validation metrics
    val_preds = model.predict(X_val.values)
    val_proba = model.predict_proba(X_val.values)
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds, average='weighted')
    val_logloss = log_loss(y_val, val_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_val, val_preds)
    
    # Feature importance
    feature_importance = sorted(
        zip(X_train.columns, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )
    
    metrics = {
        'train_accuracy': train_acc,
        'train_f1': train_f1,
        'train_logloss': train_logloss,
        'val_accuracy': val_acc,
        'val_f1': val_f1,
        'val_logloss': val_logloss,
        'confusion_matrix': cm.tolist(),
        'feature_importance': [(feat, float(imp)) for feat, imp in feature_importance]
    }
    
    print(f"\n  Train Accuracy: {train_acc:.4f}")
    print(f"  Val Accuracy:   {val_acc:.4f}")
    print(f"  Val F1:         {val_f1:.4f}")
    print(f"  Val Log Loss:   {val_logloss:.4f}")
    
    print("\n  Top 10 Features:")
    for i, (feat, imp) in enumerate(feature_importance[:10], 1):
        print(f"    {i:2d}. {feat:30s} {imp:.4f}")
    
    # 6. Save artifacts
    print("\n[6/8] Saving model artifacts...")
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Create model filename with algorithm name and accuracies
    train_acc = metrics['train_accuracy']
    val_acc = metrics['val_accuracy']
    model_filename = f"random_forest_tuned_{train_acc:.4f}_{val_acc:.4f}.pkl"
    model_path = models_dir / model_filename
    
    # Save model using pickle
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  ✓ Saved model to {model_path}")
    
    # Save metadata with matching filename
    metadata_filename = f"random_forest_tuned_{train_acc:.4f}_{val_acc:.4f}_metadata.pkl"
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
        'phase': 5,
        'model_type': 'random_forest',
        'tuning_results': tuning_results
    }
    
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"  ✓ Saved metadata to {metadata_path}")
    
    # Save metrics to JSON
    outputs_dir = Path('outputs')
    outputs_dir.mkdir(exist_ok=True)
    
    metrics_output = {
        'phase': 5,
        'model': 'random_forest_tuned',
        'validation_accuracy': float(metrics['val_accuracy']),
        'validation_f1': float(metrics['val_f1']),
        'validation_logloss': float(metrics['val_logloss']),
        'train_accuracy': float(metrics['train_accuracy']),
        'train_f1': float(metrics['train_f1']),
        'train_logloss': float(metrics['train_logloss']),
        'confusion_matrix': metrics['confusion_matrix'],
        'feature_importance': metrics['feature_importance'][:10],
        'tuning_results': tuning_results,
        'comparison': {
            'phase_4_lightgbm': 0.5211,
            'phase_5_random_forest': float(metrics['val_accuracy']),
            'improvement_vs_lightgbm_pp': float((metrics['val_accuracy'] - 0.5211) * 100)
        }
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
    print("PIPELINE COMPLETE - PHASE 5")
    print("="*70)
    print(f"\n📊 FINAL RESULTS:")
    print(f"  Train Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"  Val Accuracy:   {metrics['val_accuracy']:.4f} ✨")
    print(f"  Val F1:         {metrics['val_f1']:.4f}")
    
    print(f"\n🎯 OPTIMIZATION IMPROVEMENT: {tuning_results['improvement_pp']:+.2f}pp")
    print(f"   Baseline RF: {tuning_results['baseline_val_accuracy']:.4f} → Tuned RF: {tuning_results['best_val_accuracy']:.4f}")
    
    # Compare with Phase 4 LightGBM
    lightgbm_acc = 0.5211
    improvement_vs_lgbm = (metrics['val_accuracy'] - lightgbm_acc) * 100
    print(f"\n📈 PHASE 5 vs PHASE 4 COMPARISON:")
    print(f"   LightGBM (Phase 4): {lightgbm_acc:.4f}")
    print(f"   Random Forest (Phase 5): {metrics['val_accuracy']:.4f}")
    print(f"   Improvement: {improvement_vs_lgbm:+.2f}pp")
    
    print(f"\n✓ Model files saved in {models_dir}/")
    print(f"✓ Metrics saved in {outputs_dir}/")
    print(f"✓ Test predictions saved in {test_dir}/")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    run_pipeline()
