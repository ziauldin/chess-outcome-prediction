"""
Data balancing utilities for chess game prediction.

Implements class balancing techniques to handle imbalanced dataset:
- SMOTE (Synthetic Minority Oversampling)
- Random undersampling
- Color swap data augmentation
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import shuffle


def apply_smote_balancing(X_train, y_train, random_state=42):
    """
    Apply SMOTE to oversample minority classes.
    
    Original distribution: ~25% Black, ~40% Draw, ~35% White
    Target distribution: 33% each class
    
    Parameters
    ----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    X_balanced, y_balanced : arrays
        Balanced training data
    """
    
    # Count current distribution
    unique, counts = np.unique(y_train, return_counts=True)
    class_dist = dict(zip(unique, counts))
    
    print(f"\n  Original distribution:")
    for cls, count in class_dist.items():
        print(f"    Class {int(cls)}: {count:,} ({count/len(y_train)*100:.1f}%)")
    
    # Calculate target counts (aim for balance)
    target_count = int(len(y_train) * 0.35)  # Target ~35% each for minorities
    
    # SMOTE strategy: oversample minorities to target
    smote_strategy = {}
    for cls in unique:
        if class_dist[cls] < target_count:
            smote_strategy[cls] = target_count
    
    if smote_strategy:
        print(f"\n  Applying SMOTE oversampling...")
        smote = SMOTE(sampling_strategy=smote_strategy, random_state=random_state, k_neighbors=5)
        X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
    else:
        X_balanced, y_balanced = X_train, y_train
    
    # Undersample majority class (draws) if still too dominant
    unique, counts = np.unique(y_balanced, return_counts=True)
    class_dist_after = dict(zip(unique, counts))
    
    max_count = max(class_dist_after.values())
    target_max = int(len(y_balanced) * 0.38)  # Keep draws at ~38%
    
    if max_count > target_max:
        print(f"  Applying undersampling to majority class...")
        under_strategy = {cls: min(count, target_max) for cls, count in class_dist_after.items()}
        undersampler = RandomUnderSampler(sampling_strategy=under_strategy, random_state=random_state)
        X_balanced, y_balanced = undersampler.fit_resample(X_balanced, y_balanced)
    
    # Final distribution
    unique, counts = np.unique(y_balanced, return_counts=True)
    class_dist_final = dict(zip(unique, counts))
    
    print(f"\n  Balanced distribution:")
    for cls, count in class_dist_final.items():
        print(f"    Class {int(cls)}: {count:,} ({count/len(y_balanced)*100:.1f}%)")
    
    print(f"\n  Total samples: {len(y_train):,} → {len(y_balanced):,}")
    
    return X_balanced, y_balanced


def apply_color_swap_augmentation(df_train):
    """
    Data augmentation by swapping white/black for each game.
    
    Creates mirror games where colors are swapped:
    - White features → Black features
    - Black features → White features  
    - Result: 1.0 → 0.0, 0.0 → 1.0, 0.5 → 0.5
    
    Doubles the dataset size.
    
    Parameters
    ----------
    df_train : DataFrame
        Training data with white/black columns
        
    Returns
    -------
    df_augmented : DataFrame
        Original + mirrored games
    """
    
    print(f"\n  Applying color swap augmentation...")
    print(f"  Original games: {len(df_train):,}")
    
    # Create mirrored copy
    df_mirrored = df_train.copy()
    
    # Swap all white/black columns
    white_cols = [col for col in df_train.columns if col.startswith('white_')]
    black_cols = [col for col in df_train.columns if col.startswith('black_')]
    
    for w_col in white_cols:
        b_col = w_col.replace('white_', 'black_')
        if b_col in df_train.columns:
            # Swap values
            df_mirrored[w_col] = df_train[b_col]
            df_mirrored[b_col] = df_train[w_col]
    
    # Swap white/black player names
    if 'white' in df_train.columns and 'black' in df_train.columns:
        df_mirrored['white'] = df_train['black']
        df_mirrored['black'] = df_train['white']
    
    # Invert result (0.0 ↔ 1.0, 0.5 stays same)
    if 'result' in df_train.columns:
        df_mirrored['result'] = 1.0 - df_train['result']
    
    # Invert ELO diff if present
    if 'elo_diff' in df_train.columns:
        df_mirrored['elo_diff'] = -df_train['elo_diff']
    
    # Combine original + mirrored
    df_augmented = pd.concat([df_train, df_mirrored], ignore_index=True)
    
    # Shuffle to mix original and mirrored games
    df_augmented = shuffle(df_augmented, random_state=42)
    
    print(f"  Augmented games: {len(df_augmented):,} (2x original)")
    print(f"  ✓ Color bias removed through mirroring")
    
    return df_augmented


def prepare_balanced_data(df_train, X_train, y_train, use_augmentation=True, use_smote=True):
    """
    Complete data balancing pipeline.
    
    Steps:
    1. Color swap augmentation (optional)
    2. SMOTE + undersampling (optional)
    
    Parameters
    ----------
    df_train : DataFrame
        Raw training data
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    use_augmentation : bool
        Apply color swap augmentation
    use_smote : bool
        Apply SMOTE balancing
        
    Returns
    -------
    X_balanced, y_balanced : arrays
        Balanced training data
    df_balanced : DataFrame
        Augmented dataframe (if augmentation used)
    """
    
    print(f"\n{'='*70}")
    print("DATA BALANCING PIPELINE")
    print(f"{'='*70}")
    
    df_balanced = df_train.copy()
    
    # Step 1: Color swap augmentation
    if use_augmentation:
        print("\n[Step 1/2] Color Swap Augmentation...")
        df_balanced = apply_color_swap_augmentation(df_balanced)
        
        # Re-extract features after augmentation
        # Note: This assumes X_train columns match df_train
        # In practice, you'd re-run encode_features on df_balanced
        print("  ⚠️  Note: Re-run feature engineering on augmented data")
    
    # Step 2: SMOTE + undersampling
    if use_smote:
        print("\n[Step 2/2] SMOTE + Undersampling...")
        X_balanced, y_balanced = apply_smote_balancing(X_train, y_train)
    else:
        X_balanced, y_balanced = X_train, y_train
    
    print(f"\n{'='*70}")
    print("BALANCING COMPLETE")
    print(f"{'='*70}\n")
    
    return X_balanced, y_balanced, df_balanced
