#!/usr/bin/env python3
from Moisturizer import MeteoGaliciaCollector, SoilMoistureSequenceDataset
from pathlib import Path
import numpy as np

collector = MeteoGaliciaCollector(data_dir='./meteogalicia_data')
coverage, filtered_params = collector.analyze_parameter_coverage(coverage_threshold=0.25)
dense_path = Path('./meteogalicia_data') / 'dense_features.npz'

# Load WITHOUT precomputed (like precompute_augmented.py does)
base_dataset = SoilMoistureSequenceDataset(
    timeseries=str(collector.timeseries_file),
    stations=str(collector.stations_file),
    nearest=str(collector.nearest_file),
    seq_length=64,
    n_nearest=5,
    feature_params=filtered_params,
    precomputed_path=None,  # No precomputed!
    dense_array_path=str(dense_path) if dense_path.exists() else None,
    normalize=False  # No normalization
)

print(f'Dataset loaded: {len(base_dataset.sample_index)} samples')
print()

# Get first sample
if len(base_dataset.sample_index) > 0:
    sample = base_dataset[0]

    print('Sample 0 structure:')
    for key, val in sample.items():
        if hasattr(val, 'numpy'):
            arr = val.numpy()
            print(f'  {key}: shape={arr.shape}, dtype={arr.dtype}, type={type(val)}')
            if key == 'target':
                print(f'    Value: {arr}')
                print(f'    Is it a scalar? {arr.shape == () or (arr.shape == (1,) and len(arr) == 1)}')
        else:
            print(f'  {key}: {type(val)} = {val}')

    features = sample['features'].numpy()
    target = sample['target'].numpy()
    mask = sample['mask'].numpy()

    print()
    print('Sample 0 data ranges:')
    print(f'  Features: [{features.min():.6f}, {features.max():.6f}]')
    print(f'  Target: {target} (min={float(target.min()):.6f}, max={float(target.max()):.6f})')
    print(f'  Mask: [{mask.min():.6f}, {mask.max():.6f}]')
    print(f'  Mask dtype: {mask.dtype}, unique values: {np.unique(mask)}')
