# ♟️ Chess Outcome Prediction

A machine learning pipeline that predicts the result of chess games (white win, draw, or black win) from tournament history. It loads 174,000 games from 803 tournaments played by about 2,100 players between 2014 and 2020. From these it builds Elo, performance, head-to-head, form, and temporal features using only games played before each match. It then trains Random Forest, XGBoost, CatBoost, and LightGBM models and combines them in a weighted ensemble. The ensemble reaches about **52.6% validation accuracy** on a three-class problem where random guessing gets 33%.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189FDD?style=flat-square&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=flat-square&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?style=flat-square&logoColor=black)
![Optuna](https://img.shields.io/badge/Optuna-2C5BB4?style=flat-square&logoColor=white)

## ✨ What it does
- 📥 **Loads** tournament JSON files from `data/train/` and Elo ratings for the top 200 players in 2014 and 2020 from `data/elo/`
- 🧹 **Cleans** the data: removes duplicates, validates results and dates, and checks for highly correlated features
- 🧮 **Builds features** without data leakage (only games before the current one are used):
  - Elo ratings, rating difference, and expected score
  - Player win rates and previous tournament performance
  - Head-to-head record and recent form
  - Rating momentum, streaks, rest days, and tournament pressure
- ⏱️ **Splits by time**: the first 80% of games for training (2014 to 2018) and the last 20% for validation (2018 to 2020)
- 🌲 **Trains four models** (Random Forest, XGBoost, CatBoost, LightGBM) and blends their probabilities with a weighted ensemble
- 📊 **Saves** metrics, a confusion matrix, and feature importance to `outputs/`, and the model to `models/`
- 🔮 **Predicts** the 400 games in `data/test/` and writes the results back into those JSON files

## 📈 Results

| Model | Validation accuracy |
|-------|---------------------|
| Random Forest | 52.2% |
| XGBoost | 52.1% |
| CatBoost | 51.9% |
| LightGBM | 51.5% |
| **Weighted ensemble** | **52.6%** |

The project was built in phases: a baseline at about 46%, Elo and performance features at 50.2%, head-to-head and form features at 52.0%, then tuning and ensembling. The strongest features are each player's previous tournament performance and the difference between them (see `outputs/feature_importance.png`).

## 🚀 Run locally

```bash
git clone https://github.com/ziauldin/chess-outcome-prediction.git
cd chess-outcome-prediction
pip install -r requirements.txt
python run_pipeline.py
```

A full run takes about 5 to 10 minutes on a laptop. It overwrites `outputs/` and the `result` fields in `data/test/`.

`notebooks/01_eda_and_modeling.ipynb` holds the exploratory analysis and early modeling.

## 🗂️ Project structure

```
├── data/
│   ├── train/          # 803 tournament JSON files with results
│   ├── test/           # 8 tournament JSON files to predict
│   └── elo/            # Elo ratings for 2014 and 2020
├── src/
│   ├── features.py           # Data loading, Elo, performance, head-to-head and form features
│   ├── temporal_features.py  # Momentum, streaks, rest days, tournament pressure
│   ├── data_cleaning.py      # Quality checks, cleaning, correlation filtering
│   ├── hyperparameter_tuning.py  # Optuna search for LightGBM
│   ├── train.py              # Standalone LightGBM training with Optuna tuning
│   ├── predict.py            # Loads a saved model and predicts test games
│   └── update_test_jsons.py  # Writes predictions back into the test files
├── notebooks/          # EDA and modeling notebook
├── outputs/            # Metrics and feature importance
└── run_pipeline.py     # End-to-end pipeline
```

## 🛠️ Tech stack
Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · CatBoost · Optuna · Matplotlib
