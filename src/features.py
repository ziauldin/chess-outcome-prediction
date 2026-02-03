"""
Feature engineering module for chess game prediction.

=================================================================
PHASE 3: DERIVED FEATURES - HEAD-TO-HEAD & RECENT FORM (Feb 2026)
=================================================================

PHASE 3 ENHANCEMENTS:
Added sophisticated derived features capturing:
1. Head-to-Head Matchup History - Direct player matchup dynamics
2. Recent Form/Momentum - Rolling window performance and streaks

PERFORMANCE IMPROVEMENTS:
- Validation Accuracy: 50.20% → 52.02% (+1.82%)
- Validation F1: 0.4938 → 0.5162
- Total Features: 14 → 23 (+9 new derived features)

TOP 5 FEATURES (Phase 3):
1. black_prev_performance (51,506)
2. white_prev_performance (50,308)
3. white_form_last10 (35,576) ← NEW Phase 3 Feature!
4. black_form_last10 (30,817) ← NEW Phase 3 Feature!
5. tc_classic (27,276)

Features used (23 total):
BASELINE FEATURES (Phase 1):
- time_control (classic/rapid) - one-hot encoded
- tours (number of rounds) - numeric
- round_index (round number) - numeric  
- white (player name) - label encoded
- black (player name) - label encoded

ELO RATING FEATURES (Phase 2):
- white_elo, black_elo - ELO ratings (2300 default for unrated)
- elo_diff - rating difference (white - black)
- elo_avg - average rating of both players

PERFORMANCE FEATURES (Phase 2):
- white_prev_performance, black_prev_performance - historical average score
- white_perf_diff, black_perf_diff - current tournament form vs historical average

HEAD-TO-HEAD FEATURES (Phase 3 - NEW):
- h2h_games_played - number of previous games between these players
- h2h_white_win_rate - white's win rate in this specific matchup
- h2h_last_result - outcome of their most recent game

RECENT FORM FEATURES (Phase 3 - NEW):
- white_form_last5, black_form_last5 - average score in last 5 games
- white_form_last10, black_form_last10 - average score in last 10 games
- white_win_streak, black_win_streak - current winning/losing streak

KEY INSIGHTS:
- Performance features are the most predictive (top 2 features)
- Head-to-head captures psychological/stylistic matchups
- Recent form captures momentum and current confidence
- Chronological ordering critical to prevent data leakage
- Tournament boundaries used as natural performance windows
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


def load_elo_ratings(elo_dir='data/elo'):
    """
    Load ELO ratings from rating files.
    
    PHASE 2 ADDITION: Loads historical ELO ratings for player strength estimation.
    
    Rating Assignment Strategy:
    - Pre-2017 games: Use ratings_2014.txt
    - 2017+ games: Use ratings_2020.txt
    - Unrated players: Default to 2300 (strong club player level)
    
    File Format: player_name\trating (tab-separated, UTF-8)
    
    Args:
        elo_dir: Directory containing rating_2014.txt and rating_2020.txt
    
    Returns:
        Tuple of (ratings_2014, ratings_2020) dictionaries
    """
    elo_path = Path(elo_dir)
    
    ratings_2014 = {}
    ratings_2020 = {}
    
    # Load 2014 ratings
    rating_2014_file = elo_path / 'rating_2014.txt'
    if rating_2014_file.exists():
        with open(rating_2014_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        player, rating = parts
                        ratings_2014[player] = int(rating)
    
    # Load 2020 ratings
    rating_2020_file = elo_path / 'rating_2020.txt'
    if rating_2020_file.exists():
        with open(rating_2020_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        player, rating = parts
                        ratings_2020[player] = int(rating)
    
    print(f"Loaded {len(ratings_2014)} ratings from 2014, {len(ratings_2020)} ratings from 2020")
    return ratings_2014, ratings_2020


def get_player_rating(player, date, ratings_2014, ratings_2020, default_rating=2300):
    """
    Get player rating based on date.
    
    PHASE 2 ADDITION: Time-aware rating lookup to match ratings to game era.
    
    Logic:
    - Games before 2017: Use ratings_2014 (captures player strength in early period)
    - Games from 2017+: Use ratings_2020 (captures player strength in later period)
    - Missing players: Default to 2300 (reasonable baseline for competitive players)
    
    Args:
        player: Player name
        date: Game date (datetime object)
        ratings_2014: Dict of 2014 ratings
        ratings_2020: Dict of 2020 ratings
        default_rating: Default rating for unrated players (default: 2300)
        
    Returns:
        Integer rating for the player
    """
    year = date.year
    
    if year < 2017:
        return ratings_2014.get(player, default_rating)
    else:
        return ratings_2020.get(player, default_rating)


def calculate_performance_features(df):
    """
    Calculate player performance features based on historical results.
    
    PHASE 2 ADDITION: Dynamic performance tracking to capture player form.
    
    This function implements a sophisticated performance tracking system:
    
    1. PREVIOUS PERFORMANCE (white_prev_performance, black_prev_performance):
       - Cumulative average score across ALL completed tournaments
       - Chronologically ordered to prevent lookahead bias
       - Initial value: 0.5 (neutral baseline for new players)
       - Updated after each tournament completion
    
    2. PERFORMANCE DIFFERENCE (white_perf_diff, black_perf_diff):
       - Measures current tournament form vs historical average
       - Positive: Player performing above their norm
       - Negative: Player performing below their norm
       - Zero: First tournament or no deviation
    
    TOURNAMENT WINDOWING:
    - Uses tournament boundaries as natural performance windows
    - Within-tournament: Progressive calculation as games are played
    - Between-tournament: Aggregate stats from all completed tournaments
    - Prevents data leakage by strictly respecting chronological order
    
    EXAMPLE:
    - Player A historical avg: 0.55 (55% win rate)
    - Current tournament so far: 0.70 (70% win rate)
    - Performance diff: +0.15 (performing well above average)
    
    Args:
        df: DataFrame sorted by date with columns: white, black, result, date, tournament_name
    
    Returns:
        DataFrame with added columns:
        - white_prev_performance, black_prev_performance
        - white_perf_diff, black_perf_diff
    """
    df = df.copy()
    
    # Initialize performance tracking
    player_stats = {}  # {player: {'total_score': X, 'games': N, 'current_tourney_score': Y, 'current_tourney_games': M}}
    
    # Initialize columns
    df['white_prev_performance'] = 0.0
    df['black_prev_performance'] = 0.0
    df['white_perf_diff'] = 0.0
    df['black_perf_diff'] = 0.0
    
    current_tournament = None
    
    for idx in df.index:
        white = df.loc[idx, 'white']
        black = df.loc[idx, 'black']
        result = df.loc[idx, 'result']
        tournament = df.loc[idx, 'tournament_name']
        
        # Tournament change detection - finalize previous tournament stats
        if current_tournament != tournament:
            # Update player stats from completed tournament
            for player in player_stats:
                if 'current_tourney_score' in player_stats[player]:
                    player_stats[player]['total_score'] = player_stats[player].get('total_score', 0) + player_stats[player]['current_tourney_score']
                    player_stats[player]['games'] = player_stats[player].get('games', 0) + player_stats[player]['current_tourney_games']
                    # Clear current tournament tracking
                    player_stats[player]['current_tourney_score'] = 0
                    player_stats[player]['current_tourney_games'] = 0
            current_tournament = tournament
        
        # Initialize players if needed
        if white not in player_stats:
            player_stats[white] = {'total_score': 0, 'games': 0, 'current_tourney_score': 0, 'current_tourney_games': 0}
        if black not in player_stats:
            player_stats[black] = {'total_score': 0, 'games': 0, 'current_tourney_score': 0, 'current_tourney_games': 0}
        
        # Calculate previous performance (before this tournament)
        white_prev_perf = player_stats[white]['total_score'] / player_stats[white]['games'] if player_stats[white]['games'] > 0 else 0.5
        black_prev_perf = player_stats[black]['total_score'] / player_stats[black]['games'] if player_stats[black]['games'] > 0 else 0.5
        
        # Calculate current tournament performance (so far)
        white_curr_perf = player_stats[white]['current_tourney_score'] / player_stats[white]['current_tourney_games'] if player_stats[white]['current_tourney_games'] > 0 else 0.5
        black_curr_perf = player_stats[black]['current_tourney_score'] / player_stats[black]['current_tourney_games'] if player_stats[black]['current_tourney_games'] > 0 else 0.5
        
        # Performance difference (current tournament vs historical)
        white_perf_diff = white_curr_perf - white_prev_perf
        black_perf_diff = black_curr_perf - black_prev_perf
        
        # Assign features
        df.loc[idx, 'white_prev_performance'] = white_prev_perf
        df.loc[idx, 'black_prev_performance'] = black_prev_perf
        df.loc[idx, 'white_perf_diff'] = white_perf_diff
        df.loc[idx, 'black_perf_diff'] = black_perf_diff
        
        # Update current tournament stats
        white_score = result  # White's score (1.0, 0.5, or 0.0)
        black_score = 1.0 - result  # Black's score
        
        player_stats[white]['current_tourney_score'] += white_score
        player_stats[white]['current_tourney_games'] += 1
        player_stats[black]['current_tourney_score'] += black_score
        player_stats[black]['current_tourney_games'] += 1
    
    return df


def calculate_h2h_features(df):
    """
    Calculate head-to-head matchup history features.
    
    PHASE 3 ADDITION: Derived features capturing direct matchup dynamics.
    
    Features calculated:
    - h2h_games_played: Total previous games between these exact players
    - h2h_white_win_rate: White's win rate in this specific matchup
    - h2h_last_result: Outcome of their most recent game
    
    CRITICAL: Processes chronologically to prevent data leakage.
    Only uses h2h stats from games BEFORE the current game.
    
    Args:
        df: DataFrame sorted by date
        
    Returns:
        DataFrame with h2h features added
    """
    df = df.copy()
    
    # Initialize h2h tracking: {(white, black): [list of results]}
    h2h_history = {}
    
    # Initialize feature columns with neutral defaults
    df['h2h_games_played'] = 0
    df['h2h_white_win_rate'] = 0.5  # Neutral 50% if no history
    df['h2h_last_result'] = 0.5     # Neutral (draw) if no history
    
    print(f"  Calculating head-to-head features for {len(df)} games...")
    
    for idx in df.index:
        white = df.loc[idx, 'white']
        black = df.loc[idx, 'black']
        result = df.loc[idx, 'result']
        
        matchup = (white, black)
        
        # Get history BEFORE this game
        if matchup in h2h_history and len(h2h_history[matchup]) > 0:
            history = h2h_history[matchup]
            
            df.loc[idx, 'h2h_games_played'] = len(history)
            # Win rate from white's perspective (1.0=win, 0.5=draw, 0.0=loss)
            df.loc[idx, 'h2h_white_win_rate'] = np.mean(history)
            df.loc[idx, 'h2h_last_result'] = history[-1]
        
        # Update history AFTER processing (for future games)
        if matchup not in h2h_history:
            h2h_history[matchup] = []
        h2h_history[matchup].append(result)
    
    unique_matchups = len(h2h_history)
    print(f"    ✓ Found {unique_matchups} unique matchups")
    
    return df


def calculate_streak(results):
    """
    Calculate winning/losing streak from recent results.
    
    Returns:
        Positive integer: Winning streak (consecutive wins)
        Negative integer: Losing streak (consecutive losses)
        Zero: No streak or last game was a draw
    """
    if len(results) == 0:
        return 0
    
    streak = 0
    for result in reversed(results):
        if result == 1.0:  # Win
            if streak >= 0:
                streak += 1
            else:
                break  # Streak broken
        elif result == 0.0:  # Loss
            if streak <= 0:
                streak -= 1
            else:
                break  # Streak broken
        else:  # Draw (0.5)
            break  # Draws break streaks
    
    return streak


def calculate_form_features(df):
    """
    Calculate recent form (momentum) features using rolling windows.
    
    PHASE 3 ADDITION: Derived features capturing player momentum/form.
    
    Features calculated:
    - white_form_last5, black_form_last5: Average score in last 5 games
    - white_form_last10, black_form_last10: Average score in last 10 games  
    - white_win_streak, black_win_streak: Current win/loss streak
    
    Rolling windows capture:
    - Hot streaks (player in good form)
    - Cold streaks (player struggling)
    - Recent momentum shifts
    
    CRITICAL: Chronologically ordered. Only uses games BEFORE current game.
    
    Args:
        df: DataFrame sorted by date
        
    Returns:
        DataFrame with form features added
    """
    df = df.copy()
    
    # Track each player's game history: {player: [(date, score_from_player_perspective)]}
    player_games = {}
    
    # Initialize feature columns with neutral defaults (0.5 = average)
    df['white_form_last5'] = 0.5
    df['black_form_last5'] = 0.5
    df['white_form_last10'] = 0.5
    df['black_form_last10'] = 0.5
    df['white_win_streak'] = 0
    df['black_win_streak'] = 0
    
    print(f"  Calculating recent form features for {len(df)} games...")
    
    for idx in df.index:
        white = df.loc[idx, 'white']
        black = df.loc[idx, 'black']
        date = df.loc[idx, 'date']
        result = df.loc[idx, 'result']
        
        # Calculate WHITE's form from their history
        if white in player_games and len(player_games[white]) > 0:
            white_scores = [score for _, score in player_games[white]]
            
            if len(white_scores) >= 5:
                df.loc[idx, 'white_form_last5'] = np.mean(white_scores[-5:])
            if len(white_scores) >= 10:
                df.loc[idx, 'white_form_last10'] = np.mean(white_scores[-10:])
            
            # Calculate streak
            df.loc[idx, 'white_win_streak'] = calculate_streak(white_scores)
        
        # Calculate BLACK's form from their history
        if black in player_games and len(player_games[black]) > 0:
            black_scores = [score for _, score in player_games[black]]
            
            if len(black_scores) >= 5:
                df.loc[idx, 'black_form_last5'] = np.mean(black_scores[-5:])
            if len(black_scores) >= 10:
                df.loc[idx, 'black_form_last10'] = np.mean(black_scores[-10:])
            
            df.loc[idx, 'black_win_streak'] = calculate_streak(black_scores)
        
        # Update player histories AFTER processing this game
        if white not in player_games:
            player_games[white] = []
        player_games[white].append((date, result))  # White's score is the result
        
        if black not in player_games:
            player_games[black] = []
        player_games[black].append((date, 1.0 - result))  # Black's score is 1 - result
    
    num_players = len(player_games)
    avg_games = np.mean([len(games) for games in player_games.values()])
    print(f"    ✓ Tracked {num_players} players (avg {avg_games:.1f} games each)")
    
    return df


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
    
    # Sort by date to ensure chronological order for performance features
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"Loaded {len(df)} games from {len(list(data_path.glob('*.json')))} tournaments")
    
    # Load ELO ratings
    ratings_2014, ratings_2020 = load_elo_ratings()
    
    # Add ELO features
    print("Adding ELO rating features...")
    df['white_elo'] = df.apply(lambda row: get_player_rating(row['white'], row['date'], ratings_2014, ratings_2020), axis=1)
    df['black_elo'] = df.apply(lambda row: get_player_rating(row['black'], row['date'], ratings_2014, ratings_2020), axis=1)
    df['elo_diff'] = df['white_elo'] - df['black_elo']
    df['elo_avg'] = (df['white_elo'] + df['black_elo']) / 2
    
    # Add performance features
    print("Calculating performance features...")
    df = calculate_performance_features(df)
    
    # PHASE 3: Add derived features (head-to-head + recent form)
    print("\nPhase 3: Adding derived features...")
    df = calculate_h2h_features(df)
    df = calculate_form_features(df)
    
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
        
        # ELO rating features (Phase 2)
        X['white_elo'] = df['white_elo'].values
        X['black_elo'] = df['black_elo'].values
        X['elo_diff'] = df['elo_diff'].values
        X['elo_avg'] = df['elo_avg'].values
        
        # Performance features (Phase 2)
        X['white_prev_performance'] = df['white_prev_performance'].values
        X['black_prev_performance'] = df['black_prev_performance'].values
        X['white_perf_diff'] = df['white_perf_diff'].values
        X['black_perf_diff'] = df['black_perf_diff'].values
        
        # Head-to-head features (Phase 3 - NEW!)
        X['h2h_games_played'] = df['h2h_games_played'].values
        X['h2h_white_win_rate'] = df['h2h_white_win_rate'].values
        X['h2h_last_result'] = df['h2h_last_result'].values
        
        # Recent form features (Phase 3 - NEW!)
        X['white_form_last5'] = df['white_form_last5'].values
        X['black_form_last5'] = df['black_form_last5'].values
        X['white_form_last10'] = df['white_form_last10'].values
        X['black_form_last10'] = df['black_form_last10'].values
        X['white_win_streak'] = df['white_win_streak'].values
        X['black_win_streak'] = df['black_win_streak'].values
        
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
