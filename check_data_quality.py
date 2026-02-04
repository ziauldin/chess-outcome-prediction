"""
Quick data quality diagnostic script.
Checks for missing values, duplicates, and other data quality issues.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.features import load_tournament_data
from src.data_cleaning import check_data_quality, clean_data


def main():
    print("="*70)
    print("DATA QUALITY DIAGNOSTIC")
    print("="*70)
    print("\nLoading training data...")
    
    # Load data
    df = load_tournament_data('data/train')
    
    print(f"\nInitial dataset: {len(df):,} games")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Unique players: {len(set(df['white']) | set(df['black']))}")
    
    # Check data quality
    print("\n" + "="*70)
    quality_report = check_data_quality(df, verbose=True)
    
    # Show what cleaning would do
    print("\n" + "="*70)
    print("SIMULATING DATA CLEANING (DRY RUN)")
    print("="*70)
    df_clean, cleaning_stats = clean_data(df, verbose=True)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n📊 Dataset Changes:")
    print(f"  Initial games:     {quality_report['total_games']:,}")
    print(f"  Games to remove:   {cleaning_stats['total_removed']:,} ({cleaning_stats['removal_percentage']:.2f}%)")
    print(f"  Final clean games: {cleaning_stats['final_count']:,}")
    
    print(f"\n🔍 Quality Issues Found:")
    if quality_report['missing_values']:
        print(f"  ✗ Missing values in {len(quality_report['missing_values'])} columns")
    else:
        print(f"  ✓ No missing values")
    
    print(f"  {'✗' if quality_report['duplicate_count'] > 0 else '✓'} {quality_report['duplicate_count']:,} duplicate games")
    print(f"  {'✗' if quality_report['invalid_elo_count'] > 0 else '✓'} {quality_report['invalid_elo_count']:,} invalid ELO ratings")
    print(f"  {'✗' if quality_report['invalid_result_count'] > 0 else '✓'} {quality_report['invalid_result_count']:,} invalid results")
    print(f"  {'✗' if quality_report['future_dates'] > 0 else '✓'} {quality_report['future_dates']:,} future dates")
    
    print(f"\n💡 Recommendation:")
    if cleaning_stats['total_removed'] > 0:
        print(f"  Clean data will improve model accuracy by removing {cleaning_stats['removal_percentage']:.2f}% problematic games")
    else:
        print(f"  Data is already clean - no cleaning needed!")
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
