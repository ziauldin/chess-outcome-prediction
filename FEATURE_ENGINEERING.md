# Feature Engineering Documentation

## Overview
This document describes the feature engineering implemented for the chess game prediction model.

## Features

### 1. **Tournament Context Features**
- **time_control**: Game time control (classic/rapid) - one-hot encoded
- **tours**: Total number of rounds in tournament - numeric
- **round_index**: Current round number - numeric

### 2. **ELO Rating Features**
- **white_elo**: White player's ELO rating
- **black_elo**: Black player's ELO rating
- **elo_diff**: Rating difference (white_elo - black_elo)
- **elo_avg**: Average rating ((white_elo + black_elo) / 2)

**Rating Assignment Logic:**
- Games before 2017: Use `rating_2014.txt`
- Games from 2017 onwards: Use `rating_2020.txt`
- Default rating: 2300 (for unrated players)

**Rating Files:**
- Located in `data/elo/`
- Format: `player_name\trating` (tab-separated)
- Contains top 200 players per year

### 3. **Performance Features**
- **white_prev_performance**: White player's average result in all previous games (chronologically)
- **black_prev_performance**: Black player's average result in all previous games
- **white_perf_diff**: White player's current tournament performance vs historical average
- **black_perf_diff**: Black player's current tournament performance vs historical average

**Calculation Methodology:**
1. **Previous Performance**: Average score (0.0, 0.5, 1.0) across all games played before current tournament
   - Initial value: 0.5 (baseline for first-time players)
   - Updated after each tournament completion

2. **Performance Difference**: 
   - `current_tournament_avg - historical_avg`
   - Measures if player is performing above/below their typical level
   - Updated game-by-game within current tournament

**Example:**
- Player A has historical average of 0.55 (55% points)
- In current tournament, they've scored 0.625 so far (62.5%)
- Performance difference: 0.625 - 0.55 = +0.075 (performing above average)

### 4. **Player Identity Features**
- **white_encoded**: White player name - label encoded
- **black_encoded**: Black player name - label encoded

## Feature Importance

Based on current model (LightGBM):

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | white_prev_performance | 67418.54 | Performance |
| 2 | black_prev_performance | 66589.36 | Performance |
| 3 | tc_classic | 32058.81 | Context |
| 4 | white_perf_diff | 14856.64 | Performance |
| 5 | black_perf_diff | 12912.80 | Performance |
| 6 | white_encoded | 11080.37 | Identity |
| 7 | black_encoded | 10293.32 | Identity |
| 8 | round_index | 8991.81 | Context |
| 9 | tours | 4364.74 | Context |
| 10 | elo_diff | 3169.06 | Rating |

**Key Insights:**
- **Performance features dominate**: Previous performance and performance differences are the top predictors
- **ELO ratings moderate importance**: While included, elo_diff ranks 10th in importance
- **Context matters**: Time control (classic vs rapid) is the 3rd most important feature

## Data Processing Pipeline

### 1. Load Tournament Data
```python
df = load_tournament_data('data/train')
```
- Loads all JSON files from directory
- Flattens tournament structure to game-level records
- Sorts chronologically by date (critical for performance features)

### 2. Add ELO Ratings
```python
ratings_2014, ratings_2020 = load_elo_ratings()
df['white_elo'] = get_player_rating(white, date, ratings_2014, ratings_2020)
# ... similar for black_elo, elo_diff, elo_avg
```

### 3. Calculate Performance Features
```python
df = calculate_performance_features(df)
```
- Processes games chronologically
- Tracks player statistics across tournaments
- Calculates rolling averages and differences

### 4. Encode Features
```python
X_train, X_val, X_test, encoders, tc_columns = encode_features(df_train, df_val, df_test)
```
- One-hot encodes categorical features
- Label encodes player names
- Ensures consistent feature ordering

## Feature Statistics

**Total Features:** 14
- Time control: 2 (one-hot)
- Tournament context: 2 (tours, round_index)
- ELO ratings: 4 (white_elo, black_elo, elo_diff, elo_avg)
- Performance: 4 (white_prev, black_prev, white_diff, black_diff)
- Player identity: 2 (white_encoded, black_encoded)

## Performance Tracking Details

### Tournament-Based Windowing
- Performance features use **tournament boundaries** as natural windows
- Within-tournament stats are calculated progressively
- Between-tournament stats are aggregated from completed tournaments

### Chronological Ordering
- Critical: Data must be sorted by date before feature calculation
- Prevents data leakage from future games
- Ensures realistic feature availability at prediction time

### State Tracking
For each player, the system maintains:
```python
{
    'total_score': X,      # Cumulative score from completed tournaments
    'games': N,            # Total games from completed tournaments
    'current_tourney_score': Y,  # Score in ongoing tournament
    'current_tourney_games': M   # Games in ongoing tournament
}
```

### Cold Start Handling
- New players: prev_performance = 0.5, perf_diff = 0.0
- First tournament: Only perf_diff is meaningful (current vs default)
- Subsequent tournaments: Full history-based features available

## Model Performance

**Current Results (with new features):**
- Training Accuracy: 50.67%
- Validation Accuracy: 50.20%
- Validation F1 (macro): 0.4938

**Comparison to Baseline:**
The enhanced feature set (ELO + performance) maintains similar accuracy to the baseline model while providing more interpretable features that capture player strength and form.

## Future Enhancements

Potential additional features:
1. **Head-to-head records**: Win/loss/draw history between specific players
2. **Recent form**: Moving window performance (last N games)
3. **Opponent strength**: Average rating of recent opponents
4. **Home advantage**: Performance in specific venues/locations
5. **Streak features**: Current winning/losing streak
6. **Time pressure**: Performance by time control type
7. **Color preference**: Win rate as white vs black
8. **Rating momentum**: ELO rating change over time

## Code Locations

- **Feature Engineering**: `src/features.py`
  - `load_elo_ratings()`: Load rating files
  - `get_player_rating()`: Retrieve player rating by date
  - `calculate_performance_features()`: Compute performance metrics
  - `encode_features()`: Transform features for model

- **Prediction**: `src/predict.py`
  - Applies same feature transformations to test data

- **Training**: `src/train.py`
  - Uses encoded features for model training

## Rating File Format

**Location:** `data/elo/rating_2014.txt`, `data/elo/rating_2020.txt`

**Format:**
```
PlayerName\tRating
贾叶珍\t2742
吕亚光\t2239
...
```

**Characteristics:**
- Tab-separated values
- UTF-8 encoding (supports international characters)
- One player per line
- 200 top players per file
- Ratings range: 2000-2800

---

*Last Updated: February 3, 2026*
*Model Version: lightgbm_0.5067_0.5020.pkl*
