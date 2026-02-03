"""
Feature engineering module for chess game prediction.

Features used (NO player history, NO rolling stats):
- time_control (classic/rapid) - one-hot encoded
- tours (number of rounds) - numeric
- round_index (round number) - numeric  
- white (player name) - label encoded
- black (player name) - label encoded
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


def load_tournament_data(data_dir):
    """Load all tournament JSON files and flatten to game records."""
    print(f"Loading tournaments from {data_dir}...")
    data_path = Path(data_dir)
    games = []
    
    for json_file in sorted(data_path.glob('*.json')):
        with open(json_file, 'r', encoding='utf-8') as f:
            tournament = json.load(f)
        
        tournament_name = tournament['name']
        time_control = tournament.get('time_control', 'unknown')
        tours = tournament.get('tours', 0)
        
        for tour_name, tour_games in tournament['games'].items():
            round_index = int(tour_name.split('_')[1])
            
            for game in tour_games:
                game_record = {
                    'tournament_name': tournament_name,
                    'time_control': time_control,
                    'tours': tours,
                    'round_index': round_index,
                    'white': game['white'],
                    'black': game['black'],
                    'date': game['date'],
                    'game_id': game['id'],
                    'result': game.get('result', None)
                }
                games.append(game_record)
    
    df = pd.DataFrame(games)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Loaded {len(df)} games from {len(list(data_path.glob('*.json')))} tournaments")
    return df


def encode_features(df_train, df_val=None, df_test=None):
    """
    Encode raw categorical features.
    
    Returns:
        Tuple of (X_train, X_val, X_test, encoders, time_control_columns)
    """
    print("\nEncoding features...")
    
    encoders = {}
    
    # Combine all datasets to fit encoders
    df_all_list = [df_train]
    if df_val is not None:
        df_all_list.append(df_val)
    if df_test is not None:
        df_all_list.append(df_test)
    df_all = pd.concat(df_all_list, ignore_index=True)
    
    # Time control - one-hot encoding
    time_control_dummies = pd.get_dummies(df_all['time_control'], prefix='tc')
    time_control_columns = time_control_dummies.columns.tolist()
    
    # Player names - label encoding
    encoders['white'] = LabelEncoder()
    encoders['black'] = LabelEncoder()
    
    all_players = set(df_all['white'].unique()) | set(df_all['black'].unique())
    encoders['white'].fit(list(all_players))
    encoders['black'].fit(list(all_players))
    
    def transform_features(df):
        """Transform a dataframe using fitted encoders."""
        X = pd.DataFrame()
        
        # Time control one-hot
        tc_dummies = pd.get_dummies(df['time_control'], prefix='tc')
        for col in time_control_columns:
            X[col] = tc_dummies[col] if col in tc_dummies.columns else 0
        
        # Numeric features
        X['tours'] = df['tours'].values
        X['round_index'] = df['round_index'].values
        
        # Encoded player names
        X['white_encoded'] = encoders['white'].transform(df['white'])
        X['black_encoded'] = encoders['black'].transform(df['black'])
        
        return X
    
    X_train = transform_features(df_train)
    X_val = transform_features(df_val) if df_val is not None else None
    X_test = transform_features(df_test) if df_test is not None else None
    
    print(f"Feature columns: {X_train.columns.tolist()}")
    print(f"Feature shape: {X_train.shape}")
    
    return X_train, X_val, X_test, encoders, time_control_columns


def prepare_labels(df):
    """Convert results to class labels (0, 1, 2)."""
    label_map = {0.0: 0, 0.5: 1, 1.0: 2}
    y = df['result'].map(label_map).values
    return y
