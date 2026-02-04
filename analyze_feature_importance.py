"""
Comprehensive Feature Importance Analysis
Extracts and analyzes feature importance from all trained models.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def load_model_and_metadata(model_path):
    """Load model and its metadata."""
    metadata_path = str(model_path).replace('.pkl', '_metadata.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    return model, metadata


def extract_feature_importance(model, model_name, feature_names):
    """Extract feature importance from a model."""
    
    # For ensemble models, extract from base models
    if hasattr(model, 'models'):
        print(f"\n  {model_name} is an ensemble - extracting from base models:")
        importances_dict = {}
        
        for name, base_model in model.models.items():
            if hasattr(base_model, 'feature_importances_'):
                importances_dict[name] = base_model.feature_importances_
                print(f"    ✓ {name}: {len(base_model.feature_importances_)} features")
        
        if importances_dict:
            # Average across base models
            avg_importance = np.mean(list(importances_dict.values()), axis=0)
            return avg_importance, importances_dict
        else:
            return None, None
    
    # For single models
    elif hasattr(model, 'feature_importances_'):
        return model.feature_importances_, None
    
    else:
        return None, None


def analyze_all_models():
    """Analyze feature importance from all trained models."""
    
    print("="*80)
    print("COMPREHENSIVE FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    models_dir = Path('models')
    model_files = sorted(models_dir.glob('*.pkl'))
    model_files = [f for f in model_files if 'metadata' not in f.name]
    
    print(f"\nFound {len(model_files)} trained models:")
    for mf in model_files:
        print(f"  - {mf.name}")
    
    # Store all importances
    all_importances = {}
    feature_names = None
    
    # Process each model
    print("\n" + "="*80)
    print("EXTRACTING FEATURE IMPORTANCES")
    print("="*80)
    
    for model_file in model_files:
        model_name = model_file.stem
        print(f"\nProcessing: {model_name}")
        
        try:
            model, metadata = load_model_and_metadata(model_file)
            
            # Get feature names (should be same across all models)
            if feature_names is None:
                feature_names = metadata.get('feature_columns', [])
                print(f"  Feature count: {len(feature_names)}")
            
            # Extract importance
            importance, base_importances = extract_feature_importance(
                model, model_name, feature_names
            )
            
            if importance is not None:
                all_importances[model_name] = importance
                
                # If ensemble, also store base model importances
                if base_importances:
                    for base_name, base_imp in base_importances.items():
                        key = f"{model_name}_{base_name}"
                        all_importances[key] = base_imp
                
                print(f"  ✓ Successfully extracted importance")
            else:
                print(f"  ✗ No feature importance available")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    if not all_importances:
        print("\n❌ No feature importances could be extracted!")
        return
    
    # Create comprehensive DataFrame
    print("\n" + "="*80)
    print("CREATING COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    # Build DataFrame with all models
    importance_data = {'feature': feature_names}
    
    for model_name, importance in all_importances.items():
        importance_data[model_name] = importance
    
    df = pd.DataFrame(importance_data)
    
    # Calculate statistics across all models
    numeric_cols = [col for col in df.columns if col != 'feature']
    df['mean'] = df[numeric_cols].mean(axis=1)
    df['std'] = df[numeric_cols].std(axis=1)
    df['min'] = df[numeric_cols].min(axis=1)
    df['max'] = df[numeric_cols].max(axis=1)
    df['cv'] = df['std'] / (df['mean'] + 1e-10)  # Coefficient of variation
    
    # Sort by mean importance
    df = df.sort_values('mean', ascending=False)
    
    print(f"\n  Total features analyzed: {len(df)}")
    print(f"  Total models analyzed: {len(numeric_cols)}")
    
    # Save complete results
    outputs_dir = Path('outputs')
    outputs_dir.mkdir(exist_ok=True)
    
    # Save full analysis
    full_output_path = outputs_dir / 'feature_importance_all_features.csv'
    df.to_csv(full_output_path, index=False)
    print(f"\n  ✓ Saved complete analysis to: {full_output_path}")
    
    # Save summary (top 50)
    summary_path = outputs_dir / 'feature_importance_top50.csv'
    df.head(50).to_csv(summary_path, index=False)
    print(f"  ✓ Saved top 50 to: {summary_path}")
    
    # Display results
    print("\n" + "="*80)
    print("TOP 30 MOST IMPORTANT FEATURES (All Features)")
    print("="*80)
    print(f"\n{'Rank':<6} {'Feature':<35} {'Mean':<12} {'Std':<12} {'CV':<10} {'Type':<12}")
    print("-" * 95)
    
    for idx, row in df.head(30).iterrows():
        rank = df.index.get_loc(idx) + 1
        feature = row['feature']
        
        # Categorize feature
        if 'elo' in feature.lower():
            feat_type = 'ELO'
        elif any(x in feature.lower() for x in ['momentum', 'trend', 'volatility', 'streak', 
                                                  'days_since', 'games_last', 'rest', 'peak',
                                                  'activity', 'pressure']):
            feat_type = 'Temporal'
        elif any(x in feature.lower() for x in ['h2h', 'head_to_head', 'vs_']):
            feat_type = 'Head-to-Head'
        elif any(x in feature.lower() for x in ['performance', 'win_rate', 'form', 
                                                  'win_pct', 'loss', 'draw']):
            feat_type = 'Performance'
        elif any(x in feature.lower() for x in ['time_control', 'tournament']):
            feat_type = 'Context'
        else:
            feat_type = 'Other'
        
        print(f"{rank:<6} {feature:<35} {row['mean']:<12.6f} {row['std']:<12.6f} "
              f"{row['cv']:<10.4f} {feat_type:<12}")
    
    # Category summary
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE BY CATEGORY")
    print("="*80)
    
    categories = {
        'ELO Features': [],
        'Temporal Features': [],
        'Performance Features': [],
        'Head-to-Head Features': [],
        'Context Features': [],
        'Other Features': []
    }
    
    for _, row in df.iterrows():
        feature = row['feature']
        importance = row['mean']
        
        if 'elo' in feature.lower():
            categories['ELO Features'].append(importance)
        elif any(x in feature.lower() for x in ['momentum', 'trend', 'volatility', 'streak',
                                                  'days_since', 'games_last', 'rest', 'peak',
                                                  'activity', 'pressure']):
            categories['Temporal Features'].append(importance)
        elif any(x in feature.lower() for x in ['performance', 'win_rate', 'form',
                                                  'win_pct', 'loss', 'draw']):
            categories['Performance Features'].append(importance)
        elif any(x in feature.lower() for x in ['h2h', 'head_to_head', 'vs_']):
            categories['Head-to-Head Features'].append(importance)
        elif any(x in feature.lower() for x in ['time_control', 'tournament']):
            categories['Context Features'].append(importance)
        else:
            categories['Other Features'].append(importance)
    
    print(f"\n{'Category':<30} {'Count':<8} {'Mean Imp':<12} {'Total Imp':<12} {'% Total':<10}")
    print("-" * 80)
    
    total_importance = df['mean'].sum()
    for category, importances in sorted(categories.items(), key=lambda x: sum(x[1]) if x[1] else 0, reverse=True):
        if importances:
            count = len(importances)
            mean_imp = np.mean(importances)
            total_imp = np.sum(importances)
            pct = (total_imp / total_importance) * 100
            print(f"{category:<30} {count:<8} {mean_imp:<12.6f} {total_imp:<12.6f} {pct:<10.2f}%")
    
    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # Visualization 1: Top 30 features with error bars
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Plot 1: Top 30 features
    ax1 = axes[0, 0]
    top_30 = df.head(30).iloc[::-1]
    y_pos = np.arange(len(top_30))
    
    ax1.barh(y_pos, top_30['mean'], xerr=top_30['std'], 
             alpha=0.8, color='steelblue', capsize=3)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_30['feature'], fontsize=8)
    ax1.set_xlabel('Mean Importance', fontsize=10)
    ax1.set_title('Top 30 Features (Mean ± Std across all models)', 
                  fontsize=11, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: Category distribution
    ax2 = axes[0, 1]
    cat_data = []
    cat_labels = []
    for category, importances in sorted(categories.items(), 
                                        key=lambda x: sum(x[1]) if x[1] else 0, 
                                        reverse=True):
        if importances:
            cat_data.append(np.sum(importances))
            cat_labels.append(category.replace(' Features', ''))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(cat_data)))
    wedges, texts, autotexts = ax2.pie(cat_data, labels=cat_labels, autopct='%1.1f%%',
                                         colors=colors, startangle=90)
    ax2.set_title('Feature Importance by Category\n(Total Contribution)', 
                  fontsize=11, fontweight='bold')
    
    # Plot 3: Top 15 with model comparison
    ax3 = axes[1, 0]
    top_15 = df.head(15)
    
    # Get main model columns (not base models from ensemble)
    main_models = [col for col in numeric_cols if '_' not in col or 'ensemble' in col.lower()][:5]
    
    x = np.arange(len(top_15))
    width = 0.15
    
    for i, model in enumerate(main_models):
        offset = (i - len(main_models)/2) * width
        ax3.barh(x + offset, top_15[model], width, 
                label=model.replace('_', ' ').title()[:20], alpha=0.8)
    
    ax3.set_yticks(x)
    ax3.set_yticklabels(top_15['feature'], fontsize=8)
    ax3.set_xlabel('Importance', fontsize=10)
    ax3.set_title('Top 15 Features: Model Comparison', 
                  fontsize=11, fontweight='bold')
    ax3.legend(fontsize=8, loc='lower right')
    ax3.grid(axis='x', alpha=0.3)
    
    # Plot 4: Feature importance distribution
    ax4 = axes[1, 1]
    ax4.hist(df['mean'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax4.axvline(df['mean'].median(), color='red', linestyle='--', 
                linewidth=2, label=f'Median: {df["mean"].median():.6f}')
    ax4.axvline(df['mean'].mean(), color='green', linestyle='--', 
                linewidth=2, label=f'Mean: {df["mean"].mean():.6f}')
    ax4.set_xlabel('Mean Importance', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.set_title('Distribution of Feature Importances', 
                  fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = outputs_dir / 'feature_importance_comprehensive.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved comprehensive visualization to: {plot_path}")
    
    # Create separate detailed plot for top 50
    fig2, ax = plt.subplots(figsize=(12, 16))
    top_50 = df.head(50).iloc[::-1]
    y_pos = np.arange(len(top_50))
    
    # Color by category
    colors_map = {
        'ELO': '#1f77b4',
        'Temporal': '#ff7f0e',
        'Performance': '#2ca02c',
        'Head-to-Head': '#d62728',
        'Context': '#9467bd',
        'Other': '#8c564b'
    }
    
    bar_colors = []
    for feature in top_50['feature']:
        if 'elo' in feature.lower():
            bar_colors.append(colors_map['ELO'])
        elif any(x in feature.lower() for x in ['momentum', 'trend', 'volatility', 'streak',
                                                  'days_since', 'games_last', 'rest']):
            bar_colors.append(colors_map['Temporal'])
        elif any(x in feature.lower() for x in ['performance', 'win_rate', 'form']):
            bar_colors.append(colors_map['Performance'])
        elif any(x in feature.lower() for x in ['h2h', 'head_to_head', 'vs_']):
            bar_colors.append(colors_map['Head-to-Head'])
        elif any(x in feature.lower() for x in ['time_control', 'tournament']):
            bar_colors.append(colors_map['Context'])
        else:
            bar_colors.append(colors_map['Other'])
    
    ax.barh(y_pos, top_50['mean'], color=bar_colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_50['feature'], fontsize=9)
    ax.set_xlabel('Mean Importance', fontsize=11)
    ax.set_title('Top 50 Features - Color Coded by Category', 
                 fontsize=13, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors_map[cat], label=cat, alpha=0.8) 
                      for cat in colors_map.keys()]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    plot_path_50 = outputs_dir / 'feature_importance_top50.png'
    plt.savefig(plot_path_50, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved top 50 visualization to: {plot_path_50}")
    
    plt.close('all')
    
    # Statistical summary
    print("\n" + "="*80)
    print("STATISTICAL SUMMARY")
    print("="*80)
    
    print(f"\nTotal Features: {len(df)}")
    print(f"Mean Importance: {df['mean'].mean():.6f}")
    print(f"Median Importance: {df['mean'].median():.6f}")
    print(f"Std Dev: {df['mean'].std():.6f}")
    print(f"Min: {df['mean'].min():.6f} ({df.loc[df['mean'].idxmin(), 'feature']})")
    print(f"Max: {df['mean'].max():.6f} ({df.loc[df['mean'].idxmax(), 'feature']})")
    
    # Top 10 with highest variance
    print(f"\n{'='*80}")
    print("TOP 10 FEATURES WITH HIGHEST VARIANCE ACROSS MODELS")
    print("="*80)
    print("(These features are valued differently by different models)\n")
    
    high_var = df.nlargest(10, 'std')
    print(f"{'Rank':<6} {'Feature':<35} {'Mean':<12} {'Std':<12} {'CV':<10}")
    print("-" * 80)
    
    for idx, row in high_var.iterrows():
        rank = df.index.get_loc(idx) + 1
        print(f"{rank:<6} {row['feature']:<35} {row['mean']:<12.6f} "
              f"{row['std']:<12.6f} {row['cv']:<10.4f}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n📊 Output Files:")
    print(f"  - {full_output_path}")
    print(f"  - {summary_path}")
    print(f"  - {plot_path}")
    print(f"  - {plot_path_50}")
    print("\n✓ All features have been analyzed!")


if __name__ == '__main__':
    analyze_all_models()
