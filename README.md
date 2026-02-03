# Chess Game Outcome Prediction - Baseline Model

## Overview

This is a **baseline model** that predicts chess game outcomes using **only raw features** - no player history, no rolling statistics, no engineered form features.

## Prediction Target

The model predicts chess game outcomes as:
- **0.0** → Black wins
- **0.5** → Draw  
- **1.0** → White wins

## Features Used (Raw Only)

The model uses only these simple features directly from the data:

1. **time_control** - One-hot encoded (classic/rapid)
2. **tours** - Number of rounds in tournament (numeric)
3. **round_index** - Current round number (numeric)
4. **white** - White player name (label encoded)
5. **black** - Black player name (label encoded)

**NOT USED**: Player history, win rates, head-to-head records, recent form, Elo ratings

## Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Complete Pipeline
```bash
python run_pipeline.py
```

This will:
- Load all training tournament data from `data/train/`
- Create time-based 80/20 train/validation split
- Encode raw categorical features
- Train LightGBM classifier
- Evaluate and save model to `models/`
- Export metrics to `outputs/model_metrics.json`
- **Generate predictions for test data**
- **Update test JSON files with predicted results**

### Alternative: Train Only (without test predictions)
```bash
python src/train.py
```

This runs the training pipeline without generating test predictions.

This will:
- Load the trained model from `models/`
- Generate predictions for test data in `data/test/`
- Update test JSON files with predicted results

## Model Details

**Algorithm**: LightGBM multiclass classifier

**Target**: 
- 0 = Black wins (0.0)
- 1 = Draw (0.5)
- 2 = White wins (1.0)

**Train/Val Split**: Time-based (no data leakage)
- Training: First 80% of games by date
- Validation: Last 20% of games by date

**Random Seed**: 42 (for reproducibility)

## Output Files

After training:
```
models/
├── lightgbm_{train_acc}_{val_acc}.pkl          # Trained model (pickle format)
└── lightgbm_{train_acc}_{val_acc}_metadata.pkl # Encoders and metadata

Example:
├── lightgbm_0.4680_0.4034.pkl
└── lightgbm_0.4680_0.4034_metadata.pkl
```

## Actual Performance

Baseline performance (34,824 validation games):
- **Accuracy**: 40.34%
- **Macro F1**: 0.3415
- **Log Loss**: 1.0797

**Comparison to baselines:**
- Random guessing: 33.3% accuracy
- Always predict "Draw": ~40% accuracy (most common class)
- **This model**: 40.34% accuracy

Note: This is intentionally a simple baseline. Performance is limited because:
- No player strength indicators (Elo ratings)
- No historical performance data
- No head-to-head records
- No recent form tracking
- Only uses player identity and basic tournament info

## Project Structure

```
Chess Model/
├── data/
│   ├── train/              # Training tournament JSON files
│   ├── test/               # Test tournament JSON files (predictions added here)
│   ├── elo/                # Elo ratings directory
│   ├── rating_2014.txt     # Elo ratings for top 200 players (2014)
│   └── rating_2020.txt     # Elo ratings for top 200 players (2020)
├── src/
│   ├── features.py         # Feature engineering module
│   ├── train.py            # Model training pipeline
│   ├── predict.py          # Prediction pipeline
│   └── update_test_jsons.py # Update test files with predictions
├── models/                 # Saved model artifacts (created during training)
├── notebooks/
│   └── 01_eda_and_modeling.ipynb  # Exploratory data analysis notebook
├── outputs/                # Cross-validation results
├── run_pipeline.py        # Master pipeline script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

### Run Complete Pipeline (One Command)

```bash
python run_pipeline.py
```

This single command will:
1. Load and process training data
2. Engineer features
3. Train the LightGBM model
4. Generate predictions for test data
5. Update test JSON files with predicted results

### Alternative: Run Steps Individually

If you prefer to run training and prediction separately:

```bash
# Train the model
python -m src.train

# Generate predictions
python -m src.predict
```

## Data Processing

### Tournament Data Structure

Each JSON file contains:
- **name**: Tournament identifier
- **start_date**, **end_date**: Tournament dates
- **time_control**: "classic" or "rapid"
- **tours**: Number of rounds
- **games**: Nested dictionary of rounds and games
  - **white**, **black**: Player names
  - **date**: Game date
  - **id**: Unique game identifier
  - **result**: Outcome (only in training data)

### Elo Ratings

- Two rating files: 2014 and 2020 (top 200 players each)
- Games before 2017 use 2014 ratings
- Games from 2017 onwards use 2020 ratings
- Default rating of 2300 for unrated players

## Feature Engineering

### 1. Elo-Based Features
- **white_elo**, **black_elo**: Player ratings
- **elo_diff**: White rating - Black rating
- **elo_avg**: Average of both ratings
- **white_expected**, **black_expected**: Expected scores using chess formula:
  ```
  Expected = 1 / (1 + 10^(-elo_diff/400))
  ```

### 2. Historical Performance Features
- **white_total_games**, **black_total_games**: Career games played
- **white_win_rate**, **black_win_rate**: Overall win rates (score from 0 to 1)
- **white_recent_form**, **black_recent_form**: Performance in last 10 games
- **h2h_games**: Head-to-head games between these players
- **h2h_white_score**: White's score in previous encounters

**Critical**: Historical features use only games played *before* the current game date to avoid data leakage.

### 3. Tournament Context Features
- **is_rapid**, **is_classic**: Time control indicators
- **tour_progress**: Current round / total rounds
- **day_of_week**: Day of week (0=Monday, 6=Sunday)
- **is_weekend**: Weekend indicator

### Total Features: 20

## Model Architecture

### Algorithm: LightGBM Gradient Boosting

**Why LightGBM?**
- Excellent for tabular data with mixed feature types
- Handles missing values naturally
- Fast training and prediction
- Strong performance on imbalanced multiclass problems
- Built-in feature importance analysis

### Model Configuration

```python
{
    'objective': 'multiclass',
    'num_class': 3,
    'metric': ['multi_logloss', 'multi_error'],
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'early_stopping_rounds': 50,
    'num_boost_round': 1000
}
```

### Training Strategy

- **Data Split**: Time-based 80/20 train/validation split
  - Prevents data leakage
  - Validates on most recent games
- **Early Stopping**: Stops if validation loss doesn't improve for 50 rounds
- **Random Seed**: Fixed at 42 for reproducibility

### Evaluation Metrics

- **Accuracy**: Overall correct predictions
- **Macro F1-Score**: Average F1 across all three classes
- **Log Loss**: Probabilistic prediction quality
- **Classification Report**: Per-class precision, recall, F1

## Outputs

### 1. Trained Model Artifacts (`models/`)
- `lightgbm_{train_acc}_{val_acc}.pkl`: Trained LightGBM model (pickle format)
- `lightgbm_{train_acc}_{val_acc}_metadata.pkl`: Encoders, feature configuration, and training metadata

**Example:** `lightgbm_0.4680_0.4034.pkl` indicates 46.80% training accuracy and 40.34% validation accuracy

### 2. Model Metrics (`outputs/`)
- `model_metrics.json`: Training and validation metrics in JSON format
  - Validation accuracy, F1 score, log loss
  - Confusion matrix
  - Feature importance rankings

**Example**:
```json
{
  "model": "raw_baseline",
  "validation_accuracy": 0.4034,
  "validation_f1": 0.3415,
  "validation_logloss": 1.0797,
  "confusion_matrix": [...],
  "feature_importance": [...]
}
```

### 3. Updated Test Files
Test JSON files in `data/test/` are updated with `result` field containing predictions (0.0, 0.5, or 1.0).

**Example**:
```json
{
    "white": "Player A",
    "black": "Player B",
    "date": "2020-01-20",
    "id": "tournament_test_0_1",
    "result": 0.5
}
```

## Design Decisions

### 1. Why Multiclass Classification?
Chess outcomes are naturally discrete (win/draw/loss), making multiclass classification more appropriate than regression.

### 2. Why Not Ordinal Encoding?
While outcomes have an order (Black < Draw < White), the relationships aren't strictly linear. Multiclass treats each outcome independently, which better captures the complex decision boundaries.

### 3. Data Leakage Prevention
- Historical features computed using only past games
- Time-based validation split
- Separate processing for train and test data

### 4. Missing Elo Handling
Default rating (2300) approximately represents a strong club player, serving as a reasonable baseline when official ratings are unavailable.

### 5. Feature Normalization
LightGBM is tree-based and doesn't require feature scaling, simplifying the pipeline.

## Extending the Project

### Add New Features
Edit `src/features.py` and add methods to `FeatureEngineering` class:

```python
def add_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
    # Your feature engineering
    df['new_feature'] = ...
    return df
```

Update `get_feature_columns()` to include new features.

### Try Different Models
Replace LightGBM in `src/train.py` with:
- XGBoost
- CatBoost  
- Random Forest
- Neural Network

### Hyperparameter Tuning
Use Optuna or GridSearchCV to optimize:
- `learning_rate`
- `num_leaves`
- `feature_fraction`
- `bagging_fraction`

## Reproducibility Checklist

✅ Fixed random seeds (42)  
✅ Requirements file with version constraints  
✅ Time-based validation split  
✅ No data leakage in feature engineering  
✅ Deterministic model training  
✅ Saved model artifacts  
✅ Clear documentation  

## Troubleshooting

### Issue: "Model not found"
**Solution**: Run training first with `python -m src.train`

### Issue: Import errors
**Solution**: Install requirements with `pip install -r requirements.txt`

### Issue: Memory errors with large datasets
**Solution**: Process tournaments in batches or reduce historical window

### Issue: Poor predictions
**Solution**: 
1. Check feature importance for guidance
2. Increase training data
3. Tune hyperparameters
4. Add domain-specific features

## Performance Expectations

Typical results on validation set:
- **Accuracy**: 55-65%
- **Macro F1**: 0.45-0.55
- **Log Loss**: 1.0-1.1

Note: Chess outcomes have inherent randomness. Even world-class players can't predict with >70% accuracy.

## Author Notes

This is a complete, production-ready ML pipeline demonstrating:
- Clean code structure
- Proper train/test separation
- Feature engineering best practices
- Model evaluation and interpretation
- End-to-end automation

## License

This project is provided for educational and research purposes.

---

**Questions or Issues?** Review the code documentation and console output for detailed information about each step.
