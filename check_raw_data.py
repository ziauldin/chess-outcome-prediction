"""
Data cleaning check on raw tournament data (no feature engineering).
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime


def load_raw_tournaments(data_dir):
    """Load raw tournament JSON files without any feature engineering."""
    data_path = Path(data_dir)
    
    all_games = []
    tournament_files = sorted(data_path.glob('*.json'))
    
    print(f"Loading {len(tournament_files)} tournament files...")
    
    for file_path in tournament_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            tournament = json.load(f)
            
        # games is a dict with rounds like {"tour_1": [...], "tour_2": [...]}
        games_dict = tournament.get('games', {})
        for round_name, round_games in games_dict.items():
            for game in round_games:
                game_data = {
                    'tournament': tournament['name'],
                    'date': pd.to_datetime(game['date']),
                    'round': round_name,
                    'white': game['white'],
                    'black': game['black'],
                    'result': game['result'],
                    'time_control': game.get('time_control', 'unknown')
                }
                all_games.append(game_data)
    
    df = pd.DataFrame(all_games)
    print(f"✓ Loaded {len(df):,} games from {len(tournament_files)} tournaments\n")
    
    return df


def check_raw_data_quality(df):
    """Check data quality on raw tournament data."""
    print("="*70)
    print("RAW DATA QUALITY CHECK")
    print("="*70)
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"  Total games:     {len(df):,}")
    print(f"  Date range:      {df['date'].min()} to {df['date'].max()}")
    print(f"  Unique players:  {len(set(df['white']) | set(df['black']))}")
    print(f"  Tournaments:     {df['tournament'].nunique()}")
    
    # 1. Missing values
    print(f"\n🔍 [1] MISSING VALUES CHECK:")
    missing = df.isnull().sum()
    has_missing = False
    for col in df.columns:
        if missing[col] > 0:
            pct = missing[col] / len(df) * 100
            print(f"  ✗ {col:20s}: {missing[col]:6,} missing ({pct:5.2f}%)")
            has_missing = True
    
    if not has_missing:
        print(f"  ✓ No missing values found!")
    
    # 2. Null values (explicitly check for None, NaN, empty strings)
    print(f"\n🔍 [2] NULL/EMPTY VALUES CHECK:")
    has_nulls = False
    for col in ['white', 'black', 'result', 'tournament']:
        null_count = df[col].isna().sum()
        empty_count = (df[col] == '').sum() if df[col].dtype == 'object' else 0
        total_null = null_count + empty_count
        
        if total_null > 0:
            print(f"  ✗ {col:20s}: {total_null:6,} null/empty values")
            has_nulls = True
    
    if not has_nulls:
        print(f"  ✓ No null or empty values in critical columns!")
    
    # 3. Duplicate games
    print(f"\n🔍 [3] DUPLICATE GAMES CHECK:")
    duplicates = df.duplicated(subset=['white', 'black', 'date', 'result'], keep=False)
    dup_count = duplicates.sum()
    
    if dup_count > 0:
        pct = dup_count / len(df) * 100
        print(f"  ✗ Found {dup_count:,} duplicate games ({pct:.2f}%)")
        
        # Show some examples
        dup_games = df[duplicates].head(5)
        print(f"\n  Example duplicates:")
        for idx, game in dup_games.iterrows():
            print(f"    {game['date'].date()} | {game['white']} vs {game['black']} | Result: {game['result']}")
    else:
        print(f"  ✓ No duplicate games found!")
    
    # 4. Invalid results
    print(f"\n🔍 [4] RESULT VALIDATION:")
    valid_results = df['result'].isin([0.0, 0.5, 1.0])
    invalid_count = (~valid_results).sum()
    
    if invalid_count > 0:
        print(f"  ✗ Found {invalid_count:,} invalid results")
        print(f"  Invalid values: {df[~valid_results]['result'].unique()}")
    else:
        print(f"  ✓ All results are valid (0.0, 0.5, 1.0)")
    
    # 5. Result distribution
    print(f"\n  Result distribution:")
    result_counts = df['result'].value_counts().sort_index()
    for result, count in result_counts.items():
        pct = count / len(df) * 100
        label = "Black wins" if result == 0.0 else ("Draw" if result == 0.5 else "White wins")
        print(f"    {result} ({label:12s}): {count:6,} ({pct:5.1f}%)")
    
    # 6. Date validation
    print(f"\n🔍 [5] DATE VALIDATION:")
    future_dates = (df['date'] > pd.Timestamp.now()).sum()
    
    if future_dates > 0:
        print(f"  ✗ Found {future_dates:,} games with future dates")
    else:
        print(f"  ✓ No future dates found")
    
    print(f"  Date range: {(df['date'].max() - df['date'].min()).days} days")
    
    # 7. Player names check
    print(f"\n🔍 [6] PLAYER NAMES CHECK:")
    
    # Check for empty player names
    empty_white = (df['white'].isna() | (df['white'] == '')).sum()
    empty_black = (df['black'].isna() | (df['black'] == '')).sum()
    
    if empty_white > 0 or empty_black > 0:
        print(f"  ✗ Empty white names: {empty_white:,}")
        print(f"  ✗ Empty black names: {empty_black:,}")
    else:
        print(f"  ✓ All player names present")
    
    # Check for self-play
    self_play = (df['white'] == df['black']).sum()
    if self_play > 0:
        print(f"  ✗ Found {self_play:,} games where player plays themselves")
    else:
        print(f"  ✓ No self-play games")
    
    # 8. Time control check
    print(f"\n🔍 [7] TIME CONTROL CHECK:")
    print(f"  Unique time controls: {df['time_control'].nunique()}")
    
    tc_counts = df['time_control'].value_counts().head(10)
    for tc, count in tc_counts.items():
        pct = count / len(df) * 100
        print(f"    {tc:20s}: {count:6,} ({pct:5.1f}%)")
    
    # Summary
    print(f"\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    issues_found = []
    if has_missing:
        issues_found.append(f"Missing values")
    if has_nulls:
        issues_found.append(f"Null/empty values")
    if dup_count > 0:
        issues_found.append(f"{dup_count:,} duplicates ({dup_count/len(df)*100:.2f}%)")
    if invalid_count > 0:
        issues_found.append(f"{invalid_count:,} invalid results")
    if future_dates > 0:
        issues_found.append(f"{future_dates:,} future dates")
    if empty_white > 0 or empty_black > 0:
        issues_found.append(f"{empty_white + empty_black:,} empty player names")
    if self_play > 0:
        issues_found.append(f"{self_play:,} self-play games")
    
    if issues_found:
        print(f"\n⚠️  ISSUES FOUND:")
        for issue in issues_found:
            print(f"  • {issue}")
        print(f"\n💡 Recommendation: Run data cleaning to remove these issues")
        print(f"   Expected removal: ~{(dup_count + invalid_count + future_dates)/len(df)*100:.2f}% of dataset")
    else:
        print(f"\n✅ DATA IS CLEAN!")
        print(f"   No major issues found - data is ready for use")
    
    print(f"\n" + "="*70)
    
    return {
        'total_games': len(df),
        'duplicates': dup_count,
        'invalid_results': invalid_count,
        'future_dates': future_dates,
        'empty_names': empty_white + empty_black,
        'self_play': self_play
    }


def main():
    print("\n" + "="*70)
    print("RAW DATA QUALITY DIAGNOSTIC")
    print("NO FEATURE ENGINEERING - JUST RAW DATA CHECK")
    print("="*70 + "\n")
    
    # Load raw tournament data
    df = load_raw_tournaments('data/train')
    
    # Check quality
    stats = check_raw_data_quality(df)
    
    print("\nDIAGNOSTIC COMPLETE\n")


if __name__ == "__main__":
    main()
