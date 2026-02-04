"""
Comprehensive Feature Correlation Analysis
Analyzes correlations between all features to identify redundancies and relationships.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.features import load_tournament_data, encode_features
from src.temporal_features import add_temporal_features
from src.data_cleaning import clean_data


def load_feature_data():
    """Load and prepare feature data for correlation analysis."""
    
    print("="*80)
    print("LOADING FEATURE DATA")
    print("="*80)
    
    print("\n  Loading tournament data...")
    df = load_tournament_data('data/train')
    print(f"  ✓ Loaded {len(df)} games")
    
    print("\n  Cleaning data...")
    df, _ = clean_data(df, verbose=False)
    print(f"  ✓ After cleaning: {len(df)} games")
    
    print("\n  Adding temporal features...")
    df = add_temporal_features(df, verbose=False)
    print(f"  ✓ Total columns: {df.shape[1]}")
    
    # Create train/val split (use just train for correlation)
    df = df.sort_values('date').reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].copy()
    
    print("\n  Encoding features...")
    X_train, _, _, _, _ = encode_features(df_train, df_train)
    print(f"  ✓ Encoded features: {X_train.shape[1]}")
    
    return X_train


def analyze_correlations(X, threshold=0.7):
    """Analyze feature correlations."""
    
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)
    
    # Calculate correlation matrix
    print("\n  Calculating correlation matrix...")
    corr_matrix = X.corr()
    print(f"  ✓ Matrix size: {corr_matrix.shape[0]} x {corr_matrix.shape[1]}")
    
    # Find highly correlated pairs
    print(f"\n  Finding feature pairs with |correlation| > {threshold}...")
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > threshold:
                high_corr_pairs.append({
                    'feature_1': corr_matrix.columns[i],
                    'feature_2': corr_matrix.columns[j],
                    'correlation': corr_val,
                    'abs_correlation': abs(corr_val)
                })
    
    high_corr_df = pd.DataFrame(high_corr_pairs)
    if len(high_corr_df) > 0:
        high_corr_df = high_corr_df.sort_values('abs_correlation', ascending=False)
    
    print(f"  ✓ Found {len(high_corr_pairs)} highly correlated pairs")
    
    return corr_matrix, high_corr_df


def categorize_feature(feature_name):
    """Categorize a feature by its name."""
    feature = feature_name.lower()
    
    if 'elo' in feature:
        return 'ELO'
    elif any(x in feature for x in ['momentum', 'trend', 'volatility', 'streak', 
                                     'days_since', 'games_last', 'rest', 'peak',
                                     'activity', 'pressure']):
        return 'Temporal'
    elif any(x in feature for x in ['h2h', 'head_to_head', 'vs_']):
        return 'Head-to-Head'
    elif any(x in feature for x in ['performance', 'win_rate', 'form', 
                                     'win_pct', 'loss', 'draw', 'perf_diff']):
        return 'Performance'
    elif any(x in feature for x in ['time_control', 'tournament', 'tc_', 'tours', 'round']):
        return 'Context'
    elif any(x in feature for x in ['encoded', 'white', 'black']) and 'elo' not in feature:
        return 'Identity'
    else:
        return 'Other'


def create_visualizations(corr_matrix, high_corr_df, output_dir):
    """Create correlation visualizations."""
    
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # Visualization 1: Full correlation heatmap
    print("\n  Creating full correlation heatmap...")
    fig, ax = plt.subplots(figsize=(20, 18))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r', 
                center=0, vmin=-1, vmax=1, square=True, 
                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title('Feature Correlation Matrix (All Features)', 
                 fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    heatmap_path = output_dir / 'correlation_heatmap_full.png'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to: {heatmap_path}")
    
    # Visualization 2: High correlation pairs
    if len(high_corr_df) > 0:
        print("\n  Creating high correlation visualization...")
        
        top_n = min(30, len(high_corr_df))
        top_pairs = high_corr_df.head(top_n).copy()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        y_pos = np.arange(len(top_pairs))
        colors = ['red' if c < 0 else 'green' for c in top_pairs['correlation']]
        
        bars = ax.barh(y_pos, top_pairs['correlation'], color=colors, alpha=0.7)
        
        labels = [f"{row['feature_1'][:20]} ↔ {row['feature_2'][:20]}" 
                 for _, row in top_pairs.iterrows()]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Correlation Coefficient', fontsize=11)
        ax.set_title(f'Top {top_n} Feature Correlations (|r| > 0.7)', 
                    fontsize=13, fontweight='bold', pad=15)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.axvline(x=0.7, color='red', linestyle='--', linewidth=1, alpha=0.5, label='±0.7 threshold')
        ax.axvline(x=-0.7, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(axis='x', alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        pairs_path = output_dir / 'correlation_high_pairs.png'
        plt.savefig(pairs_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved to: {pairs_path}")
    
    # Visualization 3: Correlation by category
    print("\n  Creating category-based correlation analysis...")
    
    # Add categories to features
    features = corr_matrix.columns.tolist()
    categories = [categorize_feature(f) for f in features]
    
    # Create category correlation matrix
    unique_cats = sorted(set(categories))
    cat_corr = pd.DataFrame(0.0, index=unique_cats, columns=unique_cats)
    cat_counts = pd.DataFrame(0, index=unique_cats, columns=unique_cats)
    
    for i, feat1 in enumerate(features):
        cat1 = categories[i]
        for j, feat2 in enumerate(features):
            cat2 = categories[j]
            if i != j:
                cat_corr.loc[cat1, cat2] += abs(corr_matrix.iloc[i, j])
                cat_counts.loc[cat1, cat2] += 1
    
    # Average correlation
    for cat1 in unique_cats:
        for cat2 in unique_cats:
            if cat_counts.loc[cat1, cat2] > 0:
                cat_corr.loc[cat1, cat2] /= cat_counts.loc[cat1, cat2]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cat_corr, annot=True, fmt='.3f', cmap='YlOrRd', 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Average Absolute Correlation Between Feature Categories', 
                 fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    
    cat_path = output_dir / 'correlation_by_category.png'
    plt.savefig(cat_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to: {cat_path}")
    
    # Visualization 4: Distribution of correlations
    print("\n  Creating correlation distribution plot...")
    
    # Get upper triangle correlations (exclude diagonal)
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    correlations = upper_tri.stack().values
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(correlations, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero correlation')
    ax1.axvline(x=correlations.mean(), color='green', linestyle='--', 
                linewidth=2, label=f'Mean: {correlations.mean():.3f}')
    ax1.set_xlabel('Correlation Coefficient', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Distribution of Feature Correlations', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Box plot
    ax2 = axes[1]
    box_data = [
        correlations[correlations >= 0],
        correlations[correlations < 0],
        np.abs(correlations)
    ]
    ax2.boxplot(box_data, labels=['Positive', 'Negative', 'Absolute'])
    ax2.set_ylabel('Correlation Coefficient', fontsize=11)
    ax2.set_title('Correlation Statistics', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    dist_path = output_dir / 'correlation_distribution.png'
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to: {dist_path}")


def main():
    """Main analysis function."""
    
    print("="*80)
    print("COMPREHENSIVE FEATURE CORRELATION ANALYSIS")
    print("="*80)
    
    # Load data
    X = load_feature_data()
    
    # Analyze correlations at different thresholds
    thresholds = [0.5, 0.7, 0.8, 0.9]
    
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS AT MULTIPLE THRESHOLDS")
    print("="*80)
    
    for threshold in thresholds:
        _, high_corr_df = analyze_correlations(X, threshold=threshold)
        print(f"\n  Threshold {threshold}: {len(high_corr_df)} pairs")
    
    # Full analysis at 0.7 threshold
    corr_matrix, high_corr_df = analyze_correlations(X, threshold=0.7)
    
    # Save results
    outputs_dir = Path('outputs')
    outputs_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save full correlation matrix
    corr_path = outputs_dir / 'correlation_matrix.csv'
    corr_matrix.to_csv(corr_path)
    print(f"\n  ✓ Saved correlation matrix to: {corr_path}")
    
    # Save high correlation pairs
    if len(high_corr_df) > 0:
        pairs_path = outputs_dir / 'correlation_high_pairs.csv'
        high_corr_df.to_csv(pairs_path, index=False)
        print(f"  ✓ Saved high correlation pairs to: {pairs_path}")
    
    # Create visualizations
    create_visualizations(corr_matrix, high_corr_df, outputs_dir)
    
    # Display results
    print("\n" + "="*80)
    print("HIGH CORRELATION PAIRS (|r| > 0.7)")
    print("="*80)
    
    if len(high_corr_df) > 0:
        print(f"\n{'Rank':<6} {'Feature 1':<30} {'Feature 2':<30} {'Correlation':<12} {'Type':<15}")
        print("-" * 100)
        
        for idx, row in high_corr_df.head(30).iterrows():
            rank = high_corr_df.index.get_loc(idx) + 1
            cat1 = categorize_feature(row['feature_1'])
            cat2 = categorize_feature(row['feature_2'])
            pair_type = f"{cat1}-{cat2}" if cat1 != cat2 else cat1
            
            print(f"{rank:<6} {row['feature_1']:<30} {row['feature_2']:<30} "
                  f"{row['correlation']:<12.4f} {pair_type:<15}")
    else:
        print("\n  No feature pairs with |correlation| > 0.7")
    
    # Statistical summary
    print("\n" + "="*80)
    print("CORRELATION STATISTICS")
    print("="*80)
    
    # Get upper triangle (exclude diagonal)
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    correlations = upper_tri.stack().values
    
    print(f"\nTotal feature pairs: {len(correlations)}")
    print(f"Mean correlation: {correlations.mean():.4f}")
    print(f"Median correlation: {np.median(correlations):.4f}")
    print(f"Std deviation: {correlations.std():.4f}")
    print(f"Min correlation: {correlations.min():.4f}")
    print(f"Max correlation: {correlations.max():.4f}")
    
    print(f"\nPositive correlations: {(correlations > 0).sum()} ({(correlations > 0).mean()*100:.1f}%)")
    print(f"Negative correlations: {(correlations < 0).sum()} ({(correlations < 0).mean()*100:.1f}%)")
    print(f"Near-zero (|r| < 0.1): {(np.abs(correlations) < 0.1).sum()} ({(np.abs(correlations) < 0.1).mean()*100:.1f}%)")
    print(f"Weak (0.1 ≤ |r| < 0.3): {((np.abs(correlations) >= 0.1) & (np.abs(correlations) < 0.3)).sum()}")
    print(f"Moderate (0.3 ≤ |r| < 0.7): {((np.abs(correlations) >= 0.3) & (np.abs(correlations) < 0.7)).sum()}")
    print(f"Strong (0.7 ≤ |r| < 0.9): {((np.abs(correlations) >= 0.7) & (np.abs(correlations) < 0.9)).sum()}")
    print(f"Very strong (|r| ≥ 0.9): {(np.abs(correlations) >= 0.9).sum()}")
    
    # Category-based analysis
    print("\n" + "="*80)
    print("CORRELATION BY FEATURE CATEGORY")
    print("="*80)
    
    features = corr_matrix.columns.tolist()
    categories = [categorize_feature(f) for f in features]
    
    # Count features per category
    from collections import Counter
    cat_counts = Counter(categories)
    
    print(f"\n{'Category':<20} {'Count':<8} {'Avg Intra-Category Corr':<25}")
    print("-" * 55)
    
    for cat in sorted(cat_counts.keys()):
        cat_features = [f for f, c in zip(features, categories) if c == cat]
        if len(cat_features) > 1:
            cat_indices = [features.index(f) for f in cat_features]
            cat_corr_vals = []
            for i in cat_indices:
                for j in cat_indices:
                    if i < j:
                        cat_corr_vals.append(abs(corr_matrix.iloc[i, j]))
            avg_corr = np.mean(cat_corr_vals) if cat_corr_vals else 0.0
        else:
            avg_corr = 0.0
        
        print(f"{cat:<20} {cat_counts[cat]:<8} {avg_corr:<25.4f}")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if len(high_corr_df) > 0:
        very_high = high_corr_df[high_corr_df['abs_correlation'] >= 0.9]
        if len(very_high) > 0:
            print(f"\n⚠️  Found {len(very_high)} feature pairs with very high correlation (≥0.9)")
            print("   Consider removing one feature from each pair to reduce redundancy:")
            for _, row in very_high.head(10).iterrows():
                print(f"   - {row['feature_1']} ↔ {row['feature_2']} (r={row['correlation']:.4f})")
        
        high = high_corr_df[(high_corr_df['abs_correlation'] >= 0.7) & (high_corr_df['abs_correlation'] < 0.9)]
        if len(high) > 0:
            print(f"\n📊 Found {len(high)} feature pairs with high correlation (0.7-0.9)")
            print("   These features may be redundant but could provide complementary information.")
    else:
        print("\n✓ No highly correlated features found (all |r| < 0.7)")
        print("  Features appear to be relatively independent.")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    print(f"\n📊 Output Files:")
    print(f"  - {outputs_dir / 'correlation_matrix.csv'}")
    if len(high_corr_df) > 0:
        print(f"  - {outputs_dir / 'correlation_high_pairs.csv'}")
    print(f"  - {outputs_dir / 'correlation_heatmap_full.png'}")
    if len(high_corr_df) > 0:
        print(f"  - {outputs_dir / 'correlation_high_pairs.png'}")
    print(f"  - {outputs_dir / 'correlation_by_category.png'}")
    print(f"  - {outputs_dir / 'correlation_distribution.png'}")
    print("\n✓ Feature correlation analysis complete!")


if __name__ == '__main__':
    main()
