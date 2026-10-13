"""
Temporal and advanced feature engineering for chess prediction.

Implements time-based and psychological features:
- Rating momentum and trends
- Win/loss/draw streaks
- Rest days and fatigue
- Tournament pressure
- Clutch performance
"""

import numpy as np
import pandas as pd
from datetime import timedelta
from collections import defaultdict


def calculate_rating_momentum(df, window_sizes=[20, 40]):
    """
    Calculate ELO rating momentum over different time windows.
    
    Momentum = (current_rating - avg_rating_N_games_ago) / N
    Positive = improving, Negative = declining
    
    Parameters
    ----------
    df : DataFrame
        Tournament data sorted by date
    window_sizes : list
        Number of games to look back (default: [20, 40] for 3mo, 6mo)
        
    Returns
    -------
    df : DataFrame
        Data with momentum features added
    """
    print("\n  Calculating rating momentum...")
    
    # Track player rating history
    player_ratings = defaultdict(list)
    
    # Initialize momentum columns
    for window in window_sizes:
        df[f'white_momentum_{window}'] = 0.0
        df[f'black_momentum_{window}'] = 0.0
    
    df['white_rating_volatility'] = 0.0
    df['black_rating_volatility'] = 0.0
    df['white_peak_distance'] = 0.0
    df['black_peak_distance'] = 0.0
    
    for idx, row in df.iterrows():
        white = row['white']
        black = row['black']
        white_elo = row.get('white_elo', 1500)
        black_elo = row.get('black_elo', 1500)
        
        # Calculate momentum for white
        if len(player_ratings[white]) > 0:
            for window in window_sizes:
                if len(player_ratings[white]) >= window:
                    old_rating = player_ratings[white][-window]
                    momentum = (white_elo - old_rating) / window
                    df.at[idx, f'white_momentum_{window}'] = momentum
            
            # Volatility (std dev of recent ratings)
            if len(player_ratings[white]) >= 10:
                recent_ratings = player_ratings[white][-10:]
                df.at[idx, 'white_rating_volatility'] = np.std(recent_ratings)
            
            # Peak distance (current - historical peak)
            peak_rating = max(player_ratings[white])
            df.at[idx, 'white_peak_distance'] = white_elo - peak_rating
        
        # Calculate momentum for black
        if len(player_ratings[black]) > 0:
            for window in window_sizes:
                if len(player_ratings[black]) >= window:
                    old_rating = player_ratings[black][-window]
                    momentum = (black_elo - old_rating) / window
                    df.at[idx, f'black_momentum_{window}'] = momentum
            
            # Volatility
            if len(player_ratings[black]) >= 10:
                recent_ratings = player_ratings[black][-10:]
                df.at[idx, 'black_rating_volatility'] = np.std(recent_ratings)
            
            # Peak distance
            peak_rating = max(player_ratings[black])
            df.at[idx, 'black_peak_distance'] = black_elo - peak_rating
        
        # Update rating history
        player_ratings[white].append(white_elo)
        player_ratings[black].append(black_elo)
    
    print(f"    Added {2 * len(window_sizes) + 6} momentum/volatility features")
    return df


def calculate_streaks(df):
    """
    Calculate win/loss/draw streaks for each player.
    
    Streaks are psychological factors - players on winning streaks
    have confidence, players on losing streaks may be demoralized.
    
    Parameters
    ----------
    df : DataFrame
        Tournament data sorted by date
        
    Returns
    -------
    df : DataFrame
        Data with streak features added
    """
    print("\n  Calculating win/loss/draw streaks...")
    
    # Track player streaks
    player_streaks = defaultdict(lambda: {'win': 0, 'loss': 0, 'draw': 0, 'last_result': None})
    
    df['white_win_streak'] = 0
    df['white_loss_streak'] = 0
    df['white_draw_streak'] = 0
    df['black_win_streak'] = 0
    df['black_loss_streak'] = 0
    df['black_draw_streak'] = 0
    
    for idx, row in df.iterrows():
        white = row['white']
        black = row['black']
        result = row['result']
        
        # Set current streaks (before this game)
        df.at[idx, 'white_win_streak'] = player_streaks[white]['win']
        df.at[idx, 'white_loss_streak'] = player_streaks[white]['loss']
        df.at[idx, 'white_draw_streak'] = player_streaks[white]['draw']
        
        df.at[idx, 'black_win_streak'] = player_streaks[black]['win']
        df.at[idx, 'black_loss_streak'] = player_streaks[black]['loss']
        df.at[idx, 'black_draw_streak'] = player_streaks[black]['draw']
        
        # Update streaks based on result (for next game)
        # White perspective
        if result == 1.0:  # White wins
            player_streaks[white]['win'] += 1
            player_streaks[white]['loss'] = 0
            player_streaks[white]['draw'] = 0
            player_streaks[white]['last_result'] = 'win'
        elif result == 0.0:  # Black wins (white loss)
            player_streaks[white]['win'] = 0
            player_streaks[white]['loss'] += 1
            player_streaks[white]['draw'] = 0
            player_streaks[white]['last_result'] = 'loss'
        else:  # Draw
            player_streaks[white]['win'] = 0
            player_streaks[white]['loss'] = 0
            player_streaks[white]['draw'] += 1
            player_streaks[white]['last_result'] = 'draw'
        
        # Black perspective
        if result == 0.0:  # Black wins
            player_streaks[black]['win'] += 1
            player_streaks[black]['loss'] = 0
            player_streaks[black]['draw'] = 0
            player_streaks[black]['last_result'] = 'win'
        elif result == 1.0:  # White wins (black loss)
            player_streaks[black]['win'] = 0
            player_streaks[black]['loss'] += 1
            player_streaks[black]['draw'] = 0
            player_streaks[black]['last_result'] = 'loss'
        else:  # Draw
            player_streaks[black]['win'] = 0
            player_streaks[black]['loss'] = 0
            player_streaks[black]['draw'] += 1
            player_streaks[black]['last_result'] = 'draw'
    
    print(f"    Added 6 streak features")
    return df


def calculate_rest_days(df):
    """
    Calculate days since last game for each player.
    
    Rest affects performance - too much rest = rustiness,
    too little rest = fatigue.
    
    Parameters
    ----------
    df : DataFrame
        Tournament data sorted by date
        
    Returns
    -------
    df : DataFrame
        Data with rest day features added
    """
    print("\n  Calculating rest days and activity...")
    
    # Track last game date for each player
    last_game_date = {}
    
    df['white_days_since_last_game'] = 0
    df['black_days_since_last_game'] = 0
    df['white_games_last_30_days'] = 0
    df['black_games_last_30_days'] = 0
    
    # Track games in rolling 30-day window
    player_games_30d = defaultdict(list)
    
    for idx, row in df.iterrows():
        white = row['white']
        black = row['black']
        game_date = row['date']
        
        # Days since last game for white
        if white in last_game_date:
            days_diff = (game_date - last_game_date[white]).days
            df.at[idx, 'white_days_since_last_game'] = days_diff
        
        # Days since last game for black
        if black in last_game_date:
            days_diff = (game_date - last_game_date[black]).days
            df.at[idx, 'black_days_since_last_game'] = days_diff
        
        # Count games in last 30 days for white
        cutoff_date = game_date - timedelta(days=30)
        player_games_30d[white] = [d for d in player_games_30d[white] if d >= cutoff_date]
        df.at[idx, 'white_games_last_30_days'] = len(player_games_30d[white])
        
        # Count games in last 30 days for black
        player_games_30d[black] = [d for d in player_games_30d[black] if d >= cutoff_date]
        df.at[idx, 'black_games_last_30_days'] = len(player_games_30d[black])
        
        # Update last game date
        last_game_date[white] = game_date
        last_game_date[black] = game_date
        
        # Add to 30-day window
        player_games_30d[white].append(game_date)
        player_games_30d[black].append(game_date)
    
    print(f"    Added 4 rest/activity features")
    return df


def calculate_tournament_pressure(df):
    """
    Calculate tournament-specific pressure features.
    
    Pressure affects performance - early rounds are low pressure,
    late rounds in close tournaments are high pressure.
    
    Parameters
    ----------
    df : DataFrame
        Tournament data sorted by date
        
    Returns
    -------
    df : DataFrame
        Data with tournament pressure features added
    """
    print("\n  Calculating tournament pressure features...")
    
    df['tournament_progress'] = 0.0
    df['rounds_from_end'] = 0
    
    # Group by tournament
    for tournament in df['tours'].unique():
        tour_mask = df['tours'] == tournament
        tour_data = df[tour_mask]
        
        if 'round_index' in df.columns:
            max_round = tour_data['round_index'].max()
            
            for idx in tour_data.index:
                current_round = df.at[idx, 'round_index']
                
                # Tournament progress (0.0 = start, 1.0 = end)
                if max_round > 0:
                    progress = current_round / max_round
                    df.at[idx, 'tournament_progress'] = progress
                
                # Rounds remaining
                df.at[idx, 'rounds_from_end'] = max_round - current_round
    
    print(f"    Added 2 tournament pressure features")
    return df


def calculate_performance_features(df):
    """
    Calculate advanced performance metrics.
    
    Parameters
    ----------
    df : DataFrame
        Tournament data sorted by date
        
    Returns
    -------
    df : DataFrame
        Data with performance features added
    """
    print("\n  Calculating advanced performance features...")
    
    # Expected score based on ELO difference
    if 'elo_diff' in df.columns:
        # Formula: E = 1 / (1 + 10^(-diff/400))
        df['white_expected_score'] = 1 / (1 + 10**(-df['elo_diff'] / 400))
    else:
        df['white_expected_score'] = 0.5
    
    # Underdog factor (1 if underdog, 0 if favorite)
    df['white_is_underdog'] = (df.get('elo_diff', 0) < 0).astype(int)
    
    # Close game indicator (ELO difference < 50)
    df['is_close_matchup'] = (df.get('elo_diff', 0).abs() < 50).astype(int)
    
    print(f"    Added 3 performance features")
    return df


def add_temporal_features(df, verbose=True):
    """
    Add all temporal and psychological features.
    
    Complete feature engineering pipeline including:
    - Rating momentum (3mo, 6mo)
    - Rating volatility
    - Peak distance
    - Win/loss/draw streaks
    - Rest days
    - Activity levels
    - Tournament pressure
    - Performance expectations
    
    Parameters
    ----------
    df : DataFrame
        Tournament data sorted by date
    verbose : bool
        Print progress
        
    Returns
    -------
    df : DataFrame
        Data with all temporal features added
    """
    if verbose:
        print("\n" + "="*70)
        print("TEMPORAL FEATURE ENGINEERING")
        print("="*70)
    
    initial_features = df.shape[1]
    
    # Sort by date to ensure chronological processing
    df = df.sort_values('date').reset_index(drop=True)
    
    # Add all temporal features
    df = calculate_rating_momentum(df, window_sizes=[20, 40])
    df = calculate_streaks(df)
    df = calculate_rest_days(df)
    df = calculate_tournament_pressure(df)
    df = calculate_performance_features(df)
    
    final_features = df.shape[1]
    added_features = final_features - initial_features
    
    if verbose:
        print(f"\n  Total temporal features added: {added_features}")
        print(f"  Features: {initial_features} → {final_features}")
        print("\n" + "="*70)
        print("TEMPORAL FEATURES COMPLETE")
        print("="*70 + "\n")
    
    return df
