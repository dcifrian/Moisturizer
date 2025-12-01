#!/usr/bin/env python3
"""
Compute normalization stats for augmented dataset from base dataset (5 nearby).

Key insight: Since augmentation shuffles which 4 of 5 nearby stations appear:
- Target features: Use base stats (unchanged)
- Nearby slot features: Use min/max across ALL 5 nearby stations
- Distance: Use min/max across ALL 5 distances

This should exactly match the augmented dataset stats.
"""
from Moisturizer import MeteoGaliciaCollector, SoilMoistureSequenceDataset
from pathlib import Path
import numpy as np
from tqdm import tqdm

print("="*70)
print("COMPUTING AUGMENTED STATS FROM BASE DATASET")
print("="*70)

# Load base dataset with 5 nearby
collector = MeteoGaliciaCollector(data_dir='./meteogalicia_data')
coverage, filtered_params = collector.analyze_parameter_coverage(coverage_threshold=0.25)
dense_path = Path('./meteogalicia_data') / 'dense_features.npz'

seq_length = 2  # Use 2 for testing, 64 for production
n_nearby = 5

base_dataset = SoilMoistureSequenceDataset(
    timeseries=str(collector.timeseries_file),
    stations=str(collector.stations_file),
    nearest=str(collector.nearest_file),
    seq_length=seq_length,
    n_nearest=n_nearby,
    feature_params=filtered_params,
    precomputed_path=None,
    dense_array_path=str(dense_path),
    normalize=False
)

print(f"\nBase dataset: {len(base_dataset)} samples")
print(f"Target features: {len(filtered_params)}")
print(f"Nearby stations: {n_nearby}")
print()

# Sample to compute stats (use subset for speed)
num_samples = min(1000, len(base_dataset))
indices = np.random.choice(len(base_dataset), size=num_samples, replace=False)

print(f"Sampling {num_samples} samples for stats...")
print()

# Initialize min/max tracking
target_features_count = len(filtered_params)
nearby_features_per_station = 1 + len(filtered_params) + 1  # distance + features + soil

# For augmented dataset with 4 nearby
augmented_total_features = target_features_count + (nearby_features_per_station * 4)

feature_mins = np.full(augmented_total_features, np.inf, dtype=np.float32)
feature_maxs = np.full(augmented_total_features, -np.inf, dtype=np.float32)
target_min = np.inf
target_max = -np.inf

print("Computing stats from base dataset...")
for idx in tqdm(indices):
    sample = base_dataset[int(idx)]
    features = sample['features'].numpy()  # (seq_length, total_features_with_5_nearby)
    target = sample['target'].numpy()[0]

    # Target stats
    if target != -1000.0 and target != -9999.0:
        target_min = min(target_min, target)
        target_max = max(target_max, target)

    # Target station features (unchanged)
    target_feats = features[:, :target_features_count]  # (seq_length, 26)
    for feat_idx in range(target_features_count):
        feat_values = target_feats[:, feat_idx]
        valid = feat_values[(feat_values != -1000.0) & (feat_values != -9999.0)]
        if len(valid) > 0:
            feature_mins[feat_idx] = min(feature_mins[feat_idx], valid.min())
            feature_maxs[feat_idx] = max(feature_maxs[feat_idx], valid.max())

    # Nearby stations: Extract all 5 stations' data
    # Base has 5 nearby, each with (distance + 26 features + soil) = 28 values
    nearby_start = target_features_count
    nearby_base = features[:, nearby_start:].reshape(seq_length, n_nearby, nearby_features_per_station)

    # For each feature across nearby stations (distance, features, soil):
    # The augmented dataset will have 4 stations, each slot can be ANY of the 5
    # So the range for each slot is the min/max across all 5 stations

    for nearby_feat_idx in range(nearby_features_per_station):
        # Get this feature across all 5 nearby stations and all timesteps
        feat_across_stations = nearby_base[:, :, nearby_feat_idx]  # (seq_length, 5)
        valid = feat_across_stations[(feat_across_stations != -1000.0) & (feat_across_stations != -9999.0)]

        if len(valid) > 0:
            # In augmented dataset, this feature appears in 4 slots (nearby1, nearby2, nearby3, nearby4)
            # Each slot has the same range (min/max across all 5 base stations)
            for slot in range(4):
                aug_feat_idx = target_features_count + (slot * nearby_features_per_station) + nearby_feat_idx
                feature_mins[aug_feat_idx] = min(feature_mins[aug_feat_idx], valid.min())
                feature_maxs[aug_feat_idx] = max(feature_maxs[aug_feat_idx], valid.max())

print()
print("="*70)
print("COMPUTED AUGMENTED STATS FROM BASE DATASET")
print("="*70)
print(f"Feature range: [{feature_mins[~np.isinf(feature_mins)].min():.2f}, {feature_maxs[~np.isinf(feature_maxs)].max():.2f}]")
print(f"Target range: [{target_min:.2f}, {target_max:.2f}]")
print()

# Compare with base dataset stats (4 nearby, non-augmented)
base_stats_path = Path('./meteogalicia_data') / 'normalization_stats.npz'
if base_stats_path.exists():
    print("="*70)
    print("COMPARING WITH BASE DATASET STATS (4 nearby, non-augmented)")
    print("="*70)
    base_stats = np.load(base_stats_path)
    print(f"Base stats feature range: [{base_stats['feature_mins'].min():.2f}, {base_stats['feature_maxs'].max():.2f}]")
    print(f"Base stats target range: [{base_stats['target_min']:.2f}, {base_stats['target_max']:.2f}]")
    print()

    # These should be DIFFERENT because base has exactly 4 specific nearby stations per sample
    # Augmented uses all permutations of 4 out of 5, so ranges are wider

# Compare with actual augmented dataset stats
aug_stats_path = Path('./meteogalicia_data') / 'normalization_stats_augmented.npz'
if aug_stats_path.exists():
    print("="*70)
    print("COMPARING WITH ACTUAL AUGMENTED DATASET STATS")
    print("="*70)
    aug_stats = np.load(aug_stats_path)
    print(f"Augmented stats feature range: [{aug_stats['feature_mins'].min():.2f}, {aug_stats['feature_maxs'].max():.2f}]")
    print(f"Augmented stats target range: [{aug_stats['target_min']:.2f}, {aug_stats['target_max']:.2f}]")
    print()

    # Calculate differences
    feature_min_diff = np.abs(feature_mins - aug_stats['feature_mins'])
    feature_max_diff = np.abs(feature_maxs - aug_stats['feature_maxs'])

    # Ignore inf values
    valid_mask = ~(np.isinf(feature_mins) | np.isinf(feature_maxs))

    if valid_mask.sum() > 0:
        print(f"Feature min diff: max={feature_min_diff[valid_mask].max():.6f}, mean={feature_min_diff[valid_mask].mean():.6f}")
        print(f"Feature max diff: max={feature_max_diff[valid_mask].max():.6f}, mean={feature_max_diff[valid_mask].mean():.6f}")
        print()

        if feature_min_diff[valid_mask].max() < 0.001 and feature_max_diff[valid_mask].max() < 0.001:
            print("✅ PERFECT MATCH! Computed stats match augmented dataset stats.")
            print("   This approach is correct for --use-base-stats.")
        else:
            print("⚠️  Stats differ - may need more samples or different approach")

            # Show top 10 features with largest differences
            print("\nTop 10 features with largest max differences:")
            diffs = feature_max_diff[valid_mask]
            top_indices = np.argsort(diffs)[-10:][::-1]
            valid_indices = np.where(valid_mask)[0]

            feature_names = []
            for i in range(target_features_count):
                feature_names.append(f"target_{filtered_params[i]}")
            for slot in range(4):
                feature_names.append(f"nearby{slot+1}_distance")
                for param in filtered_params:
                    feature_names.append(f"nearby{slot+1}_{param}")
                feature_names.append(f"nearby{slot+1}_soil")

            for i in top_indices:
                feat_idx = valid_indices[i]
                print(f"  {feature_names[feat_idx]:50s} diff={diffs[i]:.6f}")
