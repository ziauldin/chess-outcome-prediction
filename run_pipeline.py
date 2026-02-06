"""
Main pipeline script for chess game prediction.

=================================================================
PHASE 7: DATA CLEANING + TEMPORAL FEATURES (Feb 2026)
=================================================================

PIPELINE OVERVIEW:
Implements comprehensive data cleaning and advanced temporal features
to improve model accuracy through better data quality and richer features.

EXECUTION STEPS:
1. Load tournament data (174K games, 2133 players)
2. PHASE 7A: Data cleaning and validation
   - Remove duplicates, invalid ELO, missing values
   - Validate results and dates
   - Feature correlation analysis
3. PHASE 7B: Temporal feature engineering
   - Rating momentum (3mo, 6mo trends)
   - Win/loss/draw streaks
   - Rest days and activity levels
   - Tournament pressure features
4. Time-based train/val split (80/20, prevents leakage)
5. Train BASE MODELS: Random Forest, XGBoost, CatBoost, LightGBM
6. WEIGHTED ENSEMBLE: Combine predictions
7. Generate predictions for test data
8. Update test JSON files with predictions

=================================================================
PHASE 7: TWO-PART STRATEGY
=================================================================

PHASE 7A: DATA CLEANING & VALIDATION
  Operations:
    - Remove duplicate games
    - Remove invalid ELO ratings (< 1200 or > 2900)
    - Impute missing ELO with median
    - Validate results (only 0.0, 0.5, 1.0)
    - Remove future dates
    - Feature correlation analysis
  Expected: +0.7-1.5pp -> 53.2-54.0%

PHASE 7B: TEMPORAL FEATURES
  New Features (~25 additional):
    - Rating momentum (20 games, 40 games)
    - Rating volatility (std dev)
    - Peak rating distance
    - Win/loss/draw streaks
    - Days since last game
    - Games in last 30 days
    - Tournament progress
    - Expected score (ELO-based)
    - Underdog/favorite factors
  Expected: +2-3pp from 7A -> 55.2-57.0%

COMBINED PHASE 7:
  [OK] Better data quality = better patterns
  [OK] Temporal features = psychological factors
  [OK] Momentum + streaks = confidence/pressure
  [OK] Total expected: 55.2-57.0% (+2.7-4.5pp from Phase 6)

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

PHASE 6 (Weighted Ensemble - Feb 2026):
  Components: 4 models + weighted averaging
  Accuracy: 52.53% (+0.19pp from Phase 5)

PHASE 7 (Clean Data + Temporal - Feb 2026 - CURRENT):
  Components: Data cleaning + 25 temporal features + ensemble
  Target: 55.2-57.0% (+2.7-4.5pp from Phase 6) ***
  Time: 5-7 hours total
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
from src.data_cleaning import check_data_quality, clean_data, analyze_feature_correlation, remove_correlated_features
from src.temporal_features import add_temporal_features


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
    Execute Phase 7: Data Cleaning + Temporal Features Pipeline.
    """
    print("="*70)
    print("CHESS GAME PREDICTION PIPELINE - PHASE 7")
    print("DATA CLEANING + TEMPORAL FEATURES")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Load data
    print("\n[1/12] Loading data...")
    df = load_tournament_data('data/train')
    print(f"  Total games: {len(df)}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Unique players: {len(set(df['white']) | set(df['black']))}")
    
    # 2. PHASE 7A: Data Quality Assessment
    print("\n[2/12] Assessing data quality...")
    quality_report = check_data_quality(df, verbose=True)
    
    # 3. PHASE 7A: Data Cleaning
    print("\n[3/12] Cleaning data...")
    df, cleaning_stats = clean_data(df, verbose=True)
    
    # 4. PHASE 7B: Add Temporal Features (before split to track history)
    print("\n[4/12] Adding temporal features...")
    features_before = df.shape[1]
    df = add_temporal_features(df, verbose=True)
    added_features = df.shape[1] - features_before
    
    # 5. Create time-based train/validation split
    print("\n[5/12] Creating train/validation split...")
    df = df.sort_values('date').reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    
    df_train = df.iloc[:split_idx].copy()
    df_val = df.iloc[split_idx:].copy()
    
    print(f"  Train: {len(df_train)} games ({df_train['date'].min().date()} to {df_train['date'].max().date()})")
    print(f"  Val:   {len(df_val)} games ({df_val['date'].min().date()} to {df_val['date'].max().date()})")
    
    # Verify no data leakage
    assert df_train['date'].max() <= df_val['date'].min(), "Data leakage detected!"
    print("  [OK] No data leakage")
    
    # 6. Feature engineering
    print("\n[6/12] Engineering features...")
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
    
    print(f"\n  Total features: {X_train.shape[1]} (Phase 3: 23 + Phase 7B: ~25 temporal)")
    
    # 7. Feature Correlation Analysis
    print("\n[7/12] Analyzing feature correlations...")
    correlated_pairs = analyze_feature_correlation(X_train, threshold=0.95, verbose=True)
    
    # Optionally remove highly correlated features
    if len(correlated_pairs) > 0:
        print("\n  Removing highly correlated features...")
        X_train, X_val, removed_features = remove_correlated_features(X_train, X_val, correlated_pairs, verbose=True)
    else:
        removed_features = []
    
    # PHASE 7C: TRAIN MULTIPLE BASE MODELS
    print("\n" + "="*70)
    print("PHASE 7C: TRAINING MULTIPLE BASE MODELS")
    print("="*70)
    print("Note: Training on cleaned data with temporal features")
    
    base_models = {}
    base_model_scores = {}
    
    # Model 1: Random Forest
    print("\n[8/12] Training Random Forest (Model 1/4)...")
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
    print("\n[9/12] Training XGBoost (Model 2/4)...")
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
    print("\n[10/12] Training CatBoost (Model 3/4)...")
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
    print("\n[11/12] Training LightGBM (Model 4/4)...")
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
    
    # FEATURE IMPORTANCE ANALYSIS
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    print("\n  Extracting feature importances from all models...")
    feature_names = X_train.columns.tolist()
    
    # Collect importances from all models
    importances_dict = {}
    
    # Random Forest
    if hasattr(rf_model, 'feature_importances_'):
        importances_dict['RandomForest'] = rf_model.feature_importances_
        print(f"  [OK] Random Forest: {len(feature_names)} features")
    
    # XGBoost
    if hasattr(xgb_model, 'feature_importances_'):
        importances_dict['XGBoost'] = xgb_model.feature_importances_
        print(f"  [OK] XGBoost: {len(feature_names)} features")
    
    # CatBoost
    if hasattr(cat_model, 'feature_importances_'):
        importances_dict['CatBoost'] = cat_model.feature_importances_
        print(f"  [OK] CatBoost: {len(feature_names)} features")
    
    # LightGBM
    if hasattr(lgb_model, 'feature_importances_'):
        importances_dict['LightGBM'] = lgb_model.feature_importances_
        print(f"  [OK] LightGBM: {len(feature_names)} features")
    
    # Calculate average importance across all models
    avg_importance = np.mean([imp for imp in importances_dict.values()], axis=0)
    
    # Create DataFrame with average importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'avg_importance': avg_importance,
        'RandomForest': importances_dict.get('RandomForest', np.zeros(len(feature_names))),
        'XGBoost': importances_dict.get('XGBoost', np.zeros(len(feature_names))),
        'CatBoost': importances_dict.get('CatBoost', np.zeros(len(feature_names))),
        'LightGBM': importances_dict.get('LightGBM', np.zeros(len(feature_names)))
    }).sort_values('avg_importance', ascending=False)
    
    # Display top 20 features
    print("\n  Top 20 Most Important Features (Averaged across all models):")
    print("  " + "-" * 68)
    print(f"  {'Rank':<6} {'Feature':<30} {'Avg Importance':<15} {'Type':<10}")
    print("  " + "-" * 68)
    
    for i, row in importance_df.head(20).iterrows():
        # Determine feature type
        feature_name = row['feature']
        if 'elo' in feature_name.lower():
            feat_type = 'ELO'
        elif any(x in feature_name.lower() for x in ['momentum', 'trend', 'volatility', 'streak', 'days_since', 'games_last', 'rest']):
            feat_type = 'Temporal'
        elif any(x in feature_name.lower() for x in ['h2h', 'head_to_head', 'vs_']):
            feat_type = 'H2H'
        elif any(x in feature_name.lower() for x in ['performance', 'win_rate', 'form']):
            feat_type = 'Perf'
        else:
            feat_type = 'Other'
        
        rank = importance_df.index.get_loc(i) + 1
        print(f"  {rank:<6} {feature_name:<30} {row['avg_importance']:<15.6f} {feat_type:<10}")
    
    # Save to CSV
    outputs_dir = Path('outputs')
    outputs_dir.mkdir(exist_ok=True)
    
    feature_importance_path = outputs_dir / 'feature_importance.csv'
    importance_df.to_csv(feature_importance_path, index=False)
    print(f"\n  [OK] Saved complete feature importance to: {feature_importance_path}")
    
    # Create visualization
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        print("\n  Creating feature importance visualization...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Top 20 features (average importance)
        top_20 = importance_df.head(20).copy()
        top_20 = top_20.iloc[::-1]  # Reverse for better display
        
        ax1.barh(range(len(top_20)), top_20['avg_importance'], color='steelblue')
        ax1.set_yticks(range(len(top_20)))
        ax1.set_yticklabels(top_20['feature'], fontsize=9)
        ax1.set_xlabel('Average Importance', fontsize=11)
        ax1.set_title('Top 20 Feature Importances\n(Ensemble Average)', fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Plot 2: Model comparison for top 10 features
        top_10 = importance_df.head(10)
        x = np.arange(len(top_10))
        width = 0.2
        
        ax2.barh(x - 1.5*width, top_10['RandomForest'], width, label='Random Forest', alpha=0.8)
        ax2.barh(x - 0.5*width, top_10['XGBoost'], width, label='XGBoost', alpha=0.8)
        ax2.barh(x + 0.5*width, top_10['CatBoost'], width, label='CatBoost', alpha=0.8)
        ax2.barh(x + 1.5*width, top_10['LightGBM'], width, label='LightGBM', alpha=0.8)
        
        ax2.set_yticks(x)
        ax2.set_yticklabels(top_10['feature'], fontsize=9)
        ax2.set_xlabel('Importance', fontsize=11)
        ax2.set_title('Top 10 Features by Model\n(Model Comparison)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9, loc='lower right')
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = outputs_dir / 'feature_importance.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  [OK] Saved visualization to: {plot_path}")
        
    except Exception as e:
        print(f"  [WARNING] Could not create visualization: {e}")
    
    # Summarize feature importance by category
    print("\n  Feature Importance Summary by Category:")
    print("  " + "-" * 50)
    
    # Categorize features
    categories = {
        'ELO Features': [],
        'Temporal Features': [],
        'Head-to-Head': [],
        'Performance': [],
        'Other': []
    }
    
    for _, row in importance_df.iterrows():
        feature_name = row['feature']
        importance = row['avg_importance']
        
        if 'elo' in feature_name.lower():
            categories['ELO Features'].append(importance)
        elif any(x in feature_name.lower() for x in ['momentum', 'trend', 'volatility', 'streak', 'days_since', 'games_last', 'rest']):
            categories['Temporal Features'].append(importance)
        elif any(x in feature_name.lower() for x in ['h2h', 'head_to_head', 'vs_']):
            categories['Head-to-Head'].append(importance)
        elif any(x in feature_name.lower() for x in ['performance', 'win_rate', 'form']):
            categories['Performance'].append(importance)
        else:
            categories['Other'].append(importance)
    
    for category, importances in categories.items():
        if importances:
            avg_imp = np.mean(importances)
            total_imp = np.sum(importances)
            count = len(importances)
            print(f"  {category:<25} Count: {count:3d}  Avg: {avg_imp:.6f}  Total: {total_imp:.6f}")
    
    # PHASE 7D: WEIGHTED ENSEMBLE
    print("\n" + "="*70)
    print("PHASE 7D: WEIGHTED ENSEMBLE (SIMPLE AVERAGING)")
    print("="*70)
    
    print("\n[12/12] Creating weighted ensemble...")
    
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
    
    print(f"\n  Ensemble validation accuracy: {ensemble_val_acc:.4f} ***")
    
    # Compare with best base model
    improvement = (ensemble_val_acc - base_model_scores[best_base_model]) * 100
    print(f"  Improvement over best base model: {improvement:+.2f}pp")
    
    if improvement > 0:
        print(f"  [OK] Ensemble outperforms best base model!")
    else:
        print(f"  [X] Ensemble underperforms - using best base model instead")
    
    # Use best performing approach
    if ensemble_val_acc >= max(base_model_scores.values()):
        print(f"\n  -> Using weighted ensemble (accuracy: {ensemble_val_acc:.4f})")
        use_ensemble = True
    else:
        print(f"\n  -> Using best base model: {best_base_model} (accuracy: {base_model_scores[best_base_model]:.4f})")
        use_ensemble = False
        ensemble_preds = base_models[best_base_model].predict(X_val)
        ensemble_proba = base_models[best_base_model].predict_proba(X_val)
        ensemble_val_acc = base_model_scores[best_base_model]
    
    # Evaluate final model
    print("\n  Evaluating final model...")
    
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
    print(f"  [OK] Saved model to {model_path}")
    
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
    print(f"  [OK] Saved metadata to {metadata_path}")
    
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
            'phase_6_ensemble': 0.5253,
            'phase_7_clean_temporal': float(metrics['val_accuracy']),
            'improvement_vs_phase6_pp': float((metrics['val_accuracy'] - 0.5253) * 100)
        },
        'phase_7_stats': {
            'initial_games': quality_report.get('total_games', 0),
            'games_removed': cleaning_stats.get('total_removed', 0),
            'final_games': cleaning_stats.get('final_count', 0),
            'temporal_features_added': added_features
        }
    }
    
    metrics_json_path = outputs_dir / 'model_metrics.json'
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_output, f, indent=2)
    print(f"  [OK] Saved metrics to {metrics_json_path}")
    
    # 7. Generate predictions and update test JSONs
    print("\n  Generating predictions for test data...")
    test_dir = 'data/test'
    df_predictions = make_predictions(test_dir, str(model_path), str(metadata_path))
    
    # Update test JSON files
    print("\n  Updating test JSON files...")
    update_test_jsons(df_predictions, test_dir)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE - PHASE 7")
    print("="*70)
    print(f"\n[STATS] PHASE 7 FINAL RESULTS:")
    print(f"  Train Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"  Val Accuracy:   {metrics['val_accuracy']:.4f} ***")
    print(f"  Val F1:         {metrics['val_f1']:.4f}")
    
    print(f"\n[GOAL] BASE MODEL COMPARISON:")
    for name, score in sorted(base_model_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"   {name:15s}: {score:.4f}")
    print(f"   {'ENSEMBLE':15s}: {metrics['val_accuracy']:.4f} ***")
    
    print(f"\n[CHART] DATA CLEANING STATS:")
    print(f"   Initial games:  {quality_report.get('total_games', 0):,}")
    print(f"   Games removed:  {cleaning_stats.get('total_removed', 0):,} ({cleaning_stats.get('removal_percentage', 0):.2f}%)")
    print(f"   Final games:    {cleaning_stats.get('final_count', 0):,}")
    
    ensemble_improvement = (metrics['val_accuracy'] - max(base_model_scores.values())) * 100
    print(f"\n[CHART] ENSEMBLE IMPROVEMENT: {ensemble_improvement:+.2f}pp over best base model")
    
    print(f"\n[STATS] PHASE PROGRESSION:")
    print(f"   Phase 4 (LightGBM):          52.11%")
    print(f"   Phase 5 (RandomForest):      52.34% (+0.23pp)")
    print(f"   Phase 6 (Ensemble):          52.53% (+0.19pp)")
    print(f"   Phase 7 (Clean+Temporal):    {metrics['val_accuracy']:.2%} ({(metrics['val_accuracy'] - 0.5253)*100:+.2f}pp)")
    
    total_improvement = (metrics['val_accuracy'] - 0.5211) * 100
    print(f"\n[LAUNCH] TOTAL IMPROVEMENT FROM PHASE 4: {total_improvement:+.2f}pp")
    
    print(f"\n[OK] Model saved in {models_dir}/")
    print(f"[OK] Metrics saved in {outputs_dir}/")
    print(f"[OK] Test predictions saved in {test_dir}/")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    run_pipeline()
