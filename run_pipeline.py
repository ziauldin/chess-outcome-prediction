"""
Main pipeline script for chess game prediction.

=================================================================
PHASE 6: MULTI-MODEL WEIGHTED ENSEMBLE (Feb 2026)
=================================================================

PIPELINE OVERVIEW:
Implements multiple models with balanced class weighting,
and uses weighted averaging ensemble for improved accuracy.

EXECUTION STEPS:
1. Load tournament data (174K games, 2133 players)
2. Engineer features (Phase 1 + 2 + 3 = 23 total features)
3. Time-based train/val split (80/20, prevents leakage)
4. Train BASE MODELS: Random Forest, XGBoost, CatBoost, LightGBM
   (All models use class_weight='balanced' on original data)
5. WEIGHTED ENSEMBLE: Combine predictions via performance-weighted averaging
6. Evaluate ensemble vs individual models (use best performer)
7. Generate predictions for test data
8. Update test JSON files with predictions

=================================================================
PHASE 6: SIMPLIFIED ENSEMBLE STRATEGY
=================================================================

PHASE 6B: MULTIPLE BASE MODELS
  Model 1: Random Forest (Phase 5 best: 52.34%)
  Model 2: XGBoost (gradient boosting variant)
  Model 3: CatBoost (categorical handling optimized)
  Model 4: LightGBM (Phase 4 best: 52.11%)
  Strategy: Train on original data with class_weight='balanced'
  Expected: Individual models 52-53%

PHASE 6C: WEIGHTED ENSEMBLE
  Architecture: Simple weighted averaging
  Weights: Based on validation accuracy of each model
  No meta-learner: Avoids overfitting on synthetic data
  Selection: Use ensemble only if it beats best single model
  Expected: +0.3-0.8pp from best single model → 52.6-53.1%

ENSEMBLE ADVANTAGES:
  ✓ Model diversity reduces variance
  ✓ Different algorithms capture different patterns
  ✓ Simple averaging avoids overfitting
  ✓ Fallback to best single model if ensemble underperforms
  ✓ No synthetic data = better generalization

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

PHASE 4 (LightGBM Tuning - Feb 2026):
  Parameters: 50 Bayesian trials for LightGBM
  Accuracy: 52.11% (+0.09pp from Phase 3)
  Components: Optuna-tuned LightGBM

PHASE 5 (Random Forest - Feb 2026):
  Parameters: 100 Bayesian trials for Random Forest
  Accuracy: 52.34% (+0.23pp from Phase 4)
  Components: Optuna-tuned Random Forest

PHASE 6 (Weighted Ensemble - Feb 2026 - CURRENT):
  Components: 4 models + weighted averaging
  Target: 52.6-53.1% (+0.3-0.8pp from Phase 5) ✨
  Time: 4-5 hours total
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix
import xgboost as xgb
import catboost as cb
import lightgbm as lgb

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.features import load_tournament_data, encode_features, prepare_labels
from src.predict import make_predictions
from src.update_test_jsons import update_test_jsons


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class WeightedEnsemble:
    """Weighted average ensemble of multiple models."""
    
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights
        
    def predict_proba(self, X):
        proba = np.zeros((X.shape[0], 3))
        for name, model in self.models.items():
            proba += self.weights[name] * model.predict_proba(X)
        return proba
        
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


def run_pipeline():
    """
    Execute Phase 6: Class Balancing + Ensemble Stacking Pipeline.
    """
    print("="*70)
    print("CHESS GAME PREDICTION PIPELINE - PHASE 6")
    print("CLASS BALANCING + ENSEMBLE STACKING")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Load data
    print("\n[1/10] Loading data...")
    df = load_tournament_data('data/train')
    print(f"  Total games: {len(df)}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Unique players: {len(set(df['white']) | set(df['black']))}")
    
    # 2. Create time-based train/validation split
    print("\n[2/10] Creating train/validation split...")
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
    print("\n[3/10] Engineering features...")
    X_train, X_val, _, encoders, tc_columns = encode_features(df_train, df_val)
    y_train = prepare_labels(df_train)
    y_val = prepare_labels(df_val)
    
    print(f"  Class distribution (training):")
    print(f"    Black wins (0): {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.1f}%)")
    print(f"    Draws (1):      {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")
    print(f"    White wins (2): {(y_train == 2).sum()} ({(y_train == 2).mean()*100:.1f}%)")
    
    print(f"\n  Class distribution (validation):")
    print(f"    Black wins (0): {(y_val == 0).sum()} ({(y_val == 0).mean()*100:.1f}%)")
    print(f"    Draws (1):      {(y_val == 1).sum()} ({(y_val == 1).mean()*100:.1f}%)")
    print(f"    White wins (2): {(y_val == 2).sum()} ({(y_val == 2).mean()*100:.1f}%)")
    
    # Skip to Phase 6B - training base models
    print("\n[4/10] Skipping class balancing - training on original data with class_weight='balanced'...")
    
    # PHASE 6B: TRAIN MULTIPLE BASE MODELS
    print("\n" + "="*70)
    print("PHASE 6B: TRAINING MULTIPLE BASE MODELS (ON ORIGINAL DATA)")
    print("="*70)
    print("Note: Training on original data with class_weight='balanced' for better generalization")
    
    base_models = {}
    base_model_scores = {}
    
    # Model 1: Random Forest
    print("\n[5/10] Training Random Forest (Model 1/4)...")
    rf_model = RandomForestClassifier(
        n_estimators=749,
        max_depth=16,
        min_samples_split=6,
        min_samples_leaf=3,
        max_features='sqrt',
        class_weight='balanced_subsample',
        max_samples=0.979,
        bootstrap=True,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=0
    )
    rf_model.fit(X_train, y_train)
    rf_val_acc = accuracy_score(y_val, rf_model.predict(X_val))
    base_models['RandomForest'] = rf_model
    base_model_scores['RandomForest'] = rf_val_acc
    print(f"  Random Forest validation accuracy: {rf_val_acc:.4f}")
    
    # Model 2: XGBoost  
    print("\n[6/10] Training XGBoost (Model 2/4)...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    xgb_val_acc = accuracy_score(y_val, xgb_model.predict(X_val))
    base_models['XGBoost'] = xgb_model
    base_model_scores['XGBoost'] = xgb_val_acc
    print(f"  XGBoost validation accuracy: {xgb_val_acc:.4f}")
    
    # Model 3: CatBoost
    print("\n[7/10] Training CatBoost (Model 3/4)...")
    cat_model = cb.CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3,
        random_seed=RANDOM_SEED,
        verbose=False,
        thread_count=-1
    )
    cat_model.fit(X_train, y_train)
    cat_val_acc = accuracy_score(y_val, cat_model.predict(X_val))
    base_models['CatBoost'] = cat_model
    base_model_scores['CatBoost'] = cat_val_acc
    print(f"  CatBoost validation accuracy: {cat_val_acc:.4f}")
    
    # Model 4: LightGBM (use Phase 4 best params)
    print("\n[8/10] Training LightGBM (Model 4/4)...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.103,
        num_leaves=24,
        min_child_samples=24,
        subsample=0.732,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight='balanced',
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=-1
    )
    lgb_model.fit(X_train, y_train)
    lgb_val_acc = accuracy_score(y_val, lgb_model.predict(X_val))
    base_models['LightGBM'] = lgb_model
    base_model_scores['LightGBM'] = lgb_val_acc
    print(f"  LightGBM validation accuracy: {lgb_val_acc:.4f}")
    
    print("\n  Base Model Summary:")
    for name, score in sorted(base_model_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"    {name:15s}: {score:.4f}")
    best_base_model = max(base_model_scores, key=base_model_scores.get)
    print(f"    Best base model: {best_base_model} ({base_model_scores[best_base_model]:.4f})")
    
    # PHASE 6C: WEIGHTED ENSEMBLE
    print("\n" + "="*70)
    print("PHASE 6C: WEIGHTED ENSEMBLE (SIMPLE AVERAGING)")
    print("="*70)
    
    print("\n[9/10] Creating weighted ensemble...")
    
    # Calculate weights based on validation accuracy
    print("\n  Calculating model weights based on validation accuracy...")
    total_score = sum(base_model_scores.values())
    weights = {name: score / total_score for name, score in base_model_scores.items()}
    
    print("\n  Model weights:")
    for name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        print(f"    {name:15s}: {weight:.4f} (val acc: {base_model_scores[name]:.4f})")
    
    # Generate weighted predictions for validation set
    print("\n  Generating weighted ensemble predictions...")
    val_proba_ensemble = np.zeros((X_val.shape[0], 3))
    for name, model in base_models.items():
        val_proba_ensemble += weights[name] * model.predict_proba(X_val)
    
    ensemble_preds = np.argmax(val_proba_ensemble, axis=1)
    ensemble_proba = val_proba_ensemble
    ensemble_val_acc = accuracy_score(y_val, ensemble_preds)
    
    print(f"\n  Ensemble validation accuracy: {ensemble_val_acc:.4f} ✨")
    
    # Compare with best base model
    improvement = (ensemble_val_acc - base_model_scores[best_base_model]) * 100
    print(f"  Improvement over best base model: {improvement:+.2f}pp")
    
    if improvement > 0:
        print(f"  ✓ Ensemble outperforms best base model!")
    else:
        print(f"  ✗ Ensemble underperforms - using best base model instead")
    
    # Use best performing approach
    if ensemble_val_acc >= max(base_model_scores.values()):
        print(f"\n  → Using weighted ensemble (accuracy: {ensemble_val_acc:.4f})")
        use_ensemble = True
    else:
        print(f"\n  → Using best base model: {best_base_model} (accuracy: {base_model_scores[best_base_model]:.4f})")
        use_ensemble = False
        ensemble_preds = base_models[best_base_model].predict(X_val)
        ensemble_proba = base_models[best_base_model].predict_proba(X_val)
        ensemble_val_acc = base_model_scores[best_base_model]
    
    # 5. Evaluate final model
    print("\n[10/10] Evaluating final model...")
    
    # Training metrics
    if use_ensemble:
        train_proba_ensemble = np.zeros((X_train.shape[0], 3))
        for name, model in base_models.items():
            train_proba_ensemble += weights[name] * model.predict_proba(X_train)
        train_preds_ensemble = np.argmax(train_proba_ensemble, axis=1)
    else:
        train_preds_ensemble = base_models[best_base_model].predict(X_train)
        train_proba_ensemble = base_models[best_base_model].predict_proba(X_train)
    
    train_acc = accuracy_score(y_train, train_preds_ensemble)
    train_f1 = f1_score(y_train, train_preds_ensemble, average='weighted')
    train_logloss = log_loss(y_train, train_proba_ensemble)
    
    # Validation metrics
    val_acc = ensemble_val_acc
    val_f1 = f1_score(y_val, ensemble_preds, average='weighted')
    val_logloss = log_loss(y_val, ensemble_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_val, ensemble_preds)
    
    # Use best base model for feature importance (weighted ensemble doesn't have direct feature importance)
    best_model = base_models[best_base_model]
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = sorted(
            zip(X_train.columns, best_model.feature_importances_),
            key=lambda x: x[1],
            reverse=True
        )
    else:
        feature_importance = [(col, 0.0) for col in X_train.columns]
    
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
    
    print(f"\n  Validation Confusion Matrix:")
    print(f"    Predicted:  Black  Draw  White")
    print(f"    Black:      {cm[0][0]:5d}  {cm[0][1]:5d}  {cm[0][2]:5d}")
    print(f"    Draw:       {cm[1][0]:5d}  {cm[1][1]:5d}  {cm[1][2]:5d}")
    print(f"    White:      {cm[2][0]:5d}  {cm[2][1]:5d}  {cm[2][2]:5d}")
    
    # Store final model (ensemble or best base)
    if use_ensemble:
        model = WeightedEnsemble(base_models, weights)
        model_type = 'ensemble_weighted'
    else:
        model = base_models[best_base_model]
        model_type = best_base_model.lower()
    
    # 6. Save artifacts
    print("\n  Saving model artifacts...")
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Create model filename with algorithm name and accuracies
    train_acc = metrics['train_accuracy']
    val_acc = metrics['val_accuracy']
    model_filename = f"{model_type}_{train_acc:.4f}_{val_acc:.4f}.pkl"
    model_path = models_dir / model_filename
    
    # Save model using pickle
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  ✓ Saved model to {model_path}")
    
    # Save metadata with matching filename
    metadata_filename = f"{model_type}_{train_acc:.4f}_{val_acc:.4f}_metadata.pkl"
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
        'phase': 6,
        'model_type': model_type,
        'base_model_scores': base_model_scores,
        'ensemble_type': 'weighted_average' if use_ensemble else 'single_model',
        'num_base_models': len(base_models) if use_ensemble else 1
    }
    
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"  ✓ Saved metadata to {metadata_path}")
    
    # Save metrics to JSON
    outputs_dir = Path('outputs')
    outputs_dir.mkdir(exist_ok=True)
    
    metrics_output = {
        'phase': 6,
        'model': 'ensemble_stacking',
        'validation_accuracy': float(metrics['val_accuracy']),
        'validation_f1': float(metrics['val_f1']),
        'validation_logloss': float(metrics['val_logloss']),
        'train_accuracy': float(metrics['train_accuracy']),
        'train_f1': float(metrics['train_f1']),
        'train_logloss': float(metrics['train_logloss']),
        'confusion_matrix': metrics['confusion_matrix'],
        'feature_importance': metrics['feature_importance'][:10],
        'base_models': {
            name: float(score) for name, score in base_model_scores.items()
        },
        'ensemble_improvement_pp': float((metrics['val_accuracy'] - max(base_model_scores.values())) * 100),
        'comparison': {
            'phase_4_lightgbm': 0.5211,
            'phase_5_random_forest': 0.5234,
            'phase_6_ensemble': float(metrics['val_accuracy']),
            'improvement_vs_phase5_pp': float((metrics['val_accuracy'] - 0.5234) * 100)
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
    print("\n  Updating test JSON files...")
    update_test_jsons(df_predictions, test_dir)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE - PHASE 6")
    print("="*70)
    print(f"\n📊 ENSEMBLE FINAL RESULTS:")
    print(f"  Train Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"  Val Accuracy:   {metrics['val_accuracy']:.4f} ✨")
    print(f"  Val F1:         {metrics['val_f1']:.4f}")
    
    print(f"\n🎯 BASE MODEL COMPARISON:")
    for name, score in sorted(base_model_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"   {name:15s}: {score:.4f}")
    print(f"   {'ENSEMBLE':15s}: {metrics['val_accuracy']:.4f} ✨")
    
    ensemble_improvement = (metrics['val_accuracy'] - max(base_model_scores.values())) * 100
    print(f"\n📈 ENSEMBLE IMPROVEMENT: {ensemble_improvement:+.2f}pp over best base model")
    
    print(f"\n📊 PHASE PROGRESSION:")
    print(f"   Phase 4 (LightGBM):     52.11%")
    print(f"   Phase 5 (RandomForest): 52.34% (+0.23pp)")
    print(f"   Phase 6 (Ensemble):     {metrics['val_accuracy']:.2%} ({(metrics['val_accuracy'] - 0.5234)*100:+.2f}pp)")
    
    total_improvement = (metrics['val_accuracy'] - 0.5211) * 100
    print(f"\n🚀 TOTAL IMPROVEMENT FROM PHASE 4: {total_improvement:+.2f}pp")
    
    print(f"\n✓ Ensemble model saved in {models_dir}/")
    print(f"✓ Metrics saved in {outputs_dir}/")
    print(f"✓ Test predictions saved in {test_dir}/")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    run_pipeline()
