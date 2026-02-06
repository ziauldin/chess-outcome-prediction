"""
Utility script to update test JSON files with predictions.
"""

import json
import pandas as pd
from pathlib import Path


def update_test_jsons(df_predictions, test_dir, output_dir=None):
    """
    Update test JSON files with model predictions.
    
    Args:
        df_predictions: DataFrame with predictions (must have game_id and predicted_result)
        test_dir: Directory containing test JSON files
        output_dir: Optional output directory (if None, overwrites original files)
    """
    print("\n" + "="*70)
    print("UPDATING TEST JSON FILES")
    print("="*70)
    
    test_path = Path(test_dir)
    output_path = Path(output_dir) if output_dir else test_path
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Create a mapping of game_id to predicted result
    prediction_map = dict(zip(df_predictions['game_id'], df_predictions['predicted_result']))
    
    updated_count = 0
    
    for json_file in sorted(test_path.glob('*.json')):
        with open(json_file, 'r', encoding='utf-8') as f:
            tournament = json.load(f)
        
        # Update each game with prediction
        for tour_name, tour_games in tournament['games'].items():
            for game in tour_games:
                game_id = game['id']
                if game_id in prediction_map:
                    game['result'] = float(prediction_map[game_id])
                    updated_count += 1
        
        # Save updated JSON
        output_file = output_path / json_file.name
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(tournament, f, indent=2)
        
        print(f"[OK] Updated {json_file.name}")
    
    print(f"\n[OK] Updated {updated_count} games in {len(list(test_path.glob('*.json')))} files")
    print(f"[OK] Files saved to {output_path}")


if __name__ == '__main__':
    import sys
    from pathlib import Path
    from predict import make_predictions
    
    # Find the most recent model file
    models_dir = Path('models')
    model_files = list(models_dir.glob('lightgbm_*.pkl'))
    
    if not model_files:
        print("Error: No model files found in models/ directory")
        print("Please run training first: python run_pipeline.py")
        sys.exit(1)
    
    # Get the most recently modified model
    model_path = max(model_files, key=lambda p: p.stat().st_mtime)
    
    # Derive metadata path from model path
    model_name = model_path.stem  # e.g., 'lightgbm_0.4680_0.4034'
    metadata_path = models_dir / f"{model_name}_metadata.pkl"
    
    if not metadata_path.exists():
        print(f"Error: Metadata file not found: {metadata_path}")
        sys.exit(1)
    
    print(f"Using model: {model_path.name}")
    print(f"Using metadata: {metadata_path.name}")
    
    test_dir = 'data/test'
    
    # Generate predictions
    df_predictions = make_predictions(test_dir, str(model_path), str(metadata_path))
    
    # Update test JSONs
    update_test_jsons(df_predictions, test_dir)
