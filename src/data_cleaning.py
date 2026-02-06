"""
Data cleaning and validation for chess tournament data.

Implements quality checks and cleaning operations:
- Missing value handling
- Duplicate removal
- Outlier detection
- Data validation
- Feature correlation analysis
"""

import numpy as np
import pandas as pd
from datetime import datetime


def check_data_quality(df, verbose=True):
    """
    Perform comprehensive data quality checks.
    
    Parameters
    ----------
    df : DataFrame
        Raw tournament data
    verbose : bool
        Print detailed statistics
        
    Returns
    -------
    quality_report : dict
        Dictionary containing quality metrics
    """
    report = {}
    
    if verbose:
        print("\n" + "="*70)
        print("DATA QUALITY ASSESSMENT")
        print("="*70)
    
    # 1. Missing values
    missing = df.isnull().sum()
    report['missing_values'] = missing[missing > 0].to_dict()
    
    if verbose and len(report['missing_values']) > 0:
        print("\n[1] Missing Values:")
        for col, count in report['missing_values'].items():
            pct = count / len(df) * 100
            print(f"  {col:25s}: {count:6d} ({pct:5.2f}%)")
    elif verbose:
        print("\n[1] Missing Values: [OK] None")
    
    # 2. Duplicate games
    duplicates = df.duplicated(subset=['white', 'black', 'date', 'result'], keep=False)
    report['duplicate_count'] = duplicates.sum()
    
    if verbose:
        print(f"\n[2] Duplicate Games: {report['duplicate_count']:,}")
        if report['duplicate_count'] > 0:
            pct = report['duplicate_count'] / len(df) * 100
            print(f"    ({pct:.2f}% of dataset)")
    
    # 3. ELO rating validation
    if 'white_elo' in df.columns and 'black_elo' in df.columns:
        invalid_elo_white = ((df['white_elo'] < 1200) | (df['white_elo'] > 2900)).sum()
        invalid_elo_black = ((df['black_elo'] < 1200) | (df['black_elo'] > 2900)).sum()
        report['invalid_elo_count'] = invalid_elo_white + invalid_elo_black
        
        if verbose:
            print(f"\n[3] Invalid ELO Ratings:")
            print(f"  White ELO out of range [1200-2900]: {invalid_elo_white:,}")
            print(f"  Black ELO out of range [1200-2900]: {invalid_elo_black:,}")
            if report['invalid_elo_count'] == 0:
                print("  [OK] All ELO ratings valid")
    
    # 4. Result validation
    if 'result' in df.columns:
        valid_results = df['result'].isin([0.0, 0.5, 1.0])
        report['invalid_result_count'] = (~valid_results).sum()
        
        if verbose:
            print(f"\n[4] Invalid Results: {report['invalid_result_count']:,}")
            if report['invalid_result_count'] == 0:
                print("  [OK] All results valid (0.0, 0.5, 1.0)")
            else:
                print(f"  Invalid values: {df[~valid_results]['result'].unique()}")
    
    # 5. Date validation
    if 'date' in df.columns:
        future_dates = (df['date'] > pd.Timestamp.now()).sum()
        report['future_dates'] = future_dates
        
        if verbose:
            print(f"\n[5] Date Issues:")
            print(f"  Future dates: {future_dates:,}")
            if future_dates == 0:
                print("  [OK] All dates valid")
            print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    
    # 6. Dataset statistics
    report['total_games'] = len(df)
    report['unique_players'] = len(set(df['white']) | set(df['black']))
    
    if verbose:
        print(f"\n[6] Dataset Statistics:")
        print(f"  Total games: {report['total_games']:,}")
        print(f"  Unique players: {report['unique_players']:,}")
        print(f"  Date range: {(df['date'].max() - df['date'].min()).days} days")
    
    if verbose:
        print("\n" + "="*70)
        print("ASSESSMENT COMPLETE")
        print("="*70 + "\n")
    
    return report


def clean_data(df, verbose=True):
    """
    Clean tournament data by removing/fixing quality issues.
    
    Cleaning operations:
    1. Remove rows with missing critical values
    2. Remove duplicate games
    3. Cap/remove invalid ELO ratings
    4. Remove invalid results
    5. Remove future dates
    
    Parameters
    ----------
    df : DataFrame
        Raw tournament data
    verbose : bool
        Print cleaning statistics
        
    Returns
    -------
    df_clean : DataFrame
        Cleaned tournament data
    cleaning_stats : dict
        Statistics about cleaning operations
    """
    if verbose:
        print("\n" + "="*70)
        print("DATA CLEANING PIPELINE")
        print("="*70)
    
    df_clean = df.copy()
    initial_count = len(df_clean)
    stats = {'initial_count': initial_count}
    
    # 1. Remove missing critical values
    if verbose:
        print("\n[1/6] Handling missing values...")
    
    critical_cols = ['white', 'black', 'result', 'date']
    before = len(df_clean)
    df_clean = df_clean.dropna(subset=critical_cols)
    removed = before - len(df_clean)
    stats['missing_removed'] = removed
    
    if verbose:
        print(f"  Removed {removed:,} games with missing critical values")
    
    # Fill missing ELO with median (better than dropping)
    if 'white_elo' in df_clean.columns:
        median_elo = df_clean['white_elo'].median()
        white_missing = df_clean['white_elo'].isnull().sum()
        black_missing = df_clean['black_elo'].isnull().sum()
        
        df_clean['white_elo'].fillna(median_elo, inplace=True)
        df_clean['black_elo'].fillna(median_elo, inplace=True)
        
        if verbose and (white_missing > 0 or black_missing > 0):
            print(f"  Imputed {white_missing + black_missing:,} missing ELO ratings with median ({median_elo:.0f})")
    
    # 2. Remove duplicates
    if verbose:
        print("\n[2/6] Removing duplicate games...")
    
    before = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['white', 'black', 'date', 'result'], keep='first')
    removed = before - len(df_clean)
    stats['duplicates_removed'] = removed
    
    if verbose:
        print(f"  Removed {removed:,} duplicate games")
    
    # 3. Clean ELO ratings
    if verbose:
        print("\n[3/6] Cleaning ELO ratings...")
    
    if 'white_elo' in df_clean.columns and 'black_elo' in df_clean.columns:
        before = len(df_clean)
        
        # Remove extremely low ratings (< 1200) - likely errors
        df_clean = df_clean[(df_clean['white_elo'] >= 1200) & (df_clean['black_elo'] >= 1200)]
        
        # Cap extremely high ratings at 2900 (no human has exceeded 2882)
        df_clean.loc[df_clean['white_elo'] > 2900, 'white_elo'] = 2900
        df_clean.loc[df_clean['black_elo'] > 2900, 'black_elo'] = 2900
        
        removed = before - len(df_clean)
        stats['invalid_elo_removed'] = removed
        
        if verbose:
            print(f"  Removed {removed:,} games with invalid ELO ratings")
            capped = ((df_clean['white_elo'] == 2900) | (df_clean['black_elo'] == 2900)).sum()
            if capped > 0:
                print(f"  Capped {capped:,} extreme ELO ratings at 2900")
    
    # 4. Validate results
    if verbose:
        print("\n[4/6] Validating results...")
    
    before = len(df_clean)
    df_clean = df_clean[df_clean['result'].isin([0.0, 0.5, 1.0])]
    removed = before - len(df_clean)
    stats['invalid_results_removed'] = removed
    
    if verbose:
        print(f"  Removed {removed:,} games with invalid results")
    
    # 5. Remove future dates
    if verbose:
        print("\n[5/6] Validating dates...")
    
    before = len(df_clean)
    df_clean = df_clean[df_clean['date'] <= pd.Timestamp.now()]
    removed = before - len(df_clean)
    stats['future_dates_removed'] = removed
    
    if verbose:
        print(f"  Removed {removed:,} games with future dates")
    
    # 6. Reset index
    df_clean = df_clean.reset_index(drop=True)
    
    # Final statistics
    stats['final_count'] = len(df_clean)
    stats['total_removed'] = initial_count - stats['final_count']
    stats['removal_percentage'] = (stats['total_removed'] / initial_count) * 100
    
    if verbose:
        print(f"\n[6/6] Cleaning complete")
        print(f"\n  Initial games:  {initial_count:,}")
        print(f"  Final games:    {stats['final_count']:,}")
        print(f"  Removed:        {stats['total_removed']:,} ({stats['removal_percentage']:.2f}%)")
        print("\n" + "="*70)
        print("CLEANING COMPLETE")
        print("="*70 + "\n")
    
    return df_clean, stats


def analyze_feature_correlation(X_train, threshold=0.95, verbose=True):
    """
    Identify highly correlated features for potential removal.
    
    Parameters
    ----------
    X_train : DataFrame
        Training features
    threshold : float
        Correlation threshold (default: 0.95)
    verbose : bool
        Print correlation pairs
        
    Returns
    -------
    correlated_pairs : list
        List of (feature1, feature2, correlation) tuples
    """
    if verbose:
        print("\n" + "="*70)
        print("FEATURE CORRELATION ANALYSIS")
        print("="*70)
    
    # Calculate correlation matrix
    corr_matrix = X_train.corr().abs()
    
    # Find pairs above threshold (excluding diagonal)
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    correlated_pairs = []
    for column in upper_triangle.columns:
        high_corr = upper_triangle[column][upper_triangle[column] > threshold]
        for idx in high_corr.index:
            correlated_pairs.append((column, idx, high_corr[idx]))
    
    if verbose:
        print(f"\nHigh correlation pairs (> {threshold}):")
        if len(correlated_pairs) > 0:
            for feat1, feat2, corr in sorted(correlated_pairs, key=lambda x: x[2], reverse=True):
                print(f"  {feat1:25s} <-> {feat2:25s}: {corr:.4f}")
            print(f"\n  Total pairs: {len(correlated_pairs)}")
            print(f"  Recommendation: Consider removing one feature from each pair")
        else:
            print("  [OK] No highly correlated features found")
        
        print("\n" + "="*70 + "\n")
    
    return correlated_pairs


def remove_correlated_features(X_train, X_val, correlated_pairs, verbose=True):
    """
    Remove one feature from each highly correlated pair.
    
    Strategy: Remove the feature that appears in more pairs.
    
    Parameters
    ----------
    X_train, X_val : DataFrame
        Training and validation features
    correlated_pairs : list
        List of (feature1, feature2, correlation) tuples
    verbose : bool
        Print removal statistics
        
    Returns
    -------
    X_train_reduced, X_val_reduced : DataFrame
        Features with correlated ones removed
    removed_features : list
        List of removed feature names
    """
    if len(correlated_pairs) == 0:
        return X_train, X_val, []
    
    # Count how many times each feature appears
    feature_counts = {}
    for feat1, feat2, _ in correlated_pairs:
        feature_counts[feat1] = feature_counts.get(feat1, 0) + 1
        feature_counts[feat2] = feature_counts.get(feat2, 0) + 1
    
    # Remove features that appear most frequently
    removed_features = set()
    for feat1, feat2, _ in correlated_pairs:
        if feat1 not in removed_features and feat2 not in removed_features:
            # Remove the one that appears in more pairs
            if feature_counts[feat1] >= feature_counts[feat2]:
                removed_features.add(feat1)
            else:
                removed_features.add(feat2)
    
    removed_features = list(removed_features)
    
    # Remove from both train and val
    X_train_reduced = X_train.drop(columns=removed_features)
    X_val_reduced = X_val.drop(columns=removed_features)
    
    if verbose:
        print(f"\nRemoved {len(removed_features)} correlated features:")
        for feat in removed_features:
            print(f"  - {feat}")
        print(f"\nFeatures: {X_train.shape[1]} -> {X_train_reduced.shape[1]}")
    
    return X_train_reduced, X_val_reduced, removed_features
