#!/usr/bin/env python3
"""
Normalize existing precomputed data to avoid 24h recomputation

This script:
1. Loads your existing precomputed_sequences.npz
2. Computes normalization statistics
3. Normalizes all features and targets
4. Saves normalized version with is_normalized=True flag

Run this instead of regenerating everything!
"""

import numpy as np
from pathlib import Path
from Moisturizer import MeteoGaliciaCollector

def normalize_precomputed_data(precomputed_path, output_path, norm_stats_path):
    """
    Normalize existing precomputed data

    Args:
        precomputed_path: Path to existing precomputed_sequences.npz
        output_path: Path to save normalized version
        norm_stats_path: Path to save normalization statistics
    """
    print("=" * 60)
    print("NORMALIZING EXISTING PRECOMPUTED DATA")
    print("=" * 60)
    print(f"Loading from: {precomputed_path}")
    print(f"This will save you 24 hours of recomputation!")
    print("=" * 60)

    # Load existing precomputed data
    print("\n1. Loading existing data...")
    data = np.load(precomputed_path)

    features = data['features']
    targets = data['targets']
    masks = data['masks']
    target_stations = data['target_stations']
    end_dates = data['end_dates']
    start_dates = data['start_dates']

    print(f"   Loaded {len(features)} samples")
    print(f"   Features shape: {features.shape}")

    # Check if already normalized
    if 'is_normalized' in data and data['is_normalized'][0]:
        print("\n✓ Data is already normalized!")
        print(f"  Nothing to do. Your file at {precomputed_path} is ready to use.")
        return

    # Compute normalization statistics
    print("\n2. Computing normalization statistics...")
    print("   (excluding invalid markers: -9999, -1000)")

    n_features = features.shape[2]
    feature_mins = np.full(n_features, np.inf, dtype=np.float32)
    feature_maxs = np.full(n_features, -np.inf, dtype=np.float32)

    invalid_markers = [-9999.0, -1000.0]

    # Compute min/max for each feature (in batches to save memory)
    batch_size = 1000
    for i in range(0, len(features), batch_size):
        end_i = min(i + batch_size, len(features))
        if i % 10000 == 0:
            print(f"   Progress: {i}/{len(features)} ({100*i/len(features):.1f}%)")

        features_batch = features[i:end_i]
        masks_batch = masks[i:end_i]

        for feat_idx in range(n_features):
            feat_data = features_batch[:, :, feat_idx]
            feat_mask = masks_batch[:, :, feat_idx]

            # Get valid data (masked and not invalid marker)
            valid_mask = feat_mask > 0
            for marker in invalid_markers:
                valid_mask &= (feat_data != marker)

            valid_data = feat_data[valid_mask]

            if len(valid_data) > 0:
                feature_mins[feat_idx] = min(feature_mins[feat_idx], valid_data.min())
                feature_maxs[feat_idx] = max(feature_maxs[feat_idx], valid_data.max())

    # Compute for targets
    valid_targets = targets.copy()
    for marker in invalid_markers:
        valid_targets = valid_targets[valid_targets != marker]

    target_min = valid_targets.min() if len(valid_targets) > 0 else 0.0
    target_max = valid_targets.max() if len(valid_targets) > 0 else 1.0

    print(f"\n   Feature min range: [{feature_mins.min():.2f}, {feature_mins.max():.2f}]")
    print(f"   Feature max range: [{feature_maxs.min():.2f}, {feature_maxs.max():.2f}]")
    print(f"   Target range: [{target_min:.2f}, {target_max:.2f}]")

    # Save normalization statistics
    print(f"\n3. Saving normalization stats to {norm_stats_path}...")
    np.savez(
        norm_stats_path,
        feature_mins=feature_mins,
        feature_maxs=feature_maxs,
        target_min=target_min,
        target_max=target_max
    )

    # Normalize all data
    print("\n4. Normalizing all data...")
    print("   This will take a few minutes...")

    normalized_invalid_marker = -2.0

    for idx in range(len(features)):
        if idx % 1000 == 0:
            print(f"   Progress: {idx}/{len(features)} ({100*idx/len(features):.1f}%)")

        # Normalize features
        for feat_idx in range(n_features):
            feat_min = feature_mins[feat_idx]
            feat_max = feature_maxs[feat_idx]

            # Handle invalid markers
            invalid_mask = np.zeros(features[idx].shape[0], dtype=bool)
            for marker in invalid_markers:
                invalid_mask |= (features[idx][:, feat_idx] == marker)

            # Normalize valid values to [-1, 1]
            if feat_max > feat_min:
                features[idx][:, feat_idx] = 2.0 * (features[idx][:, feat_idx] - feat_min) / (feat_max - feat_min) - 1.0

            # Set invalid markers to -2
            features[idx][invalid_mask, feat_idx] = normalized_invalid_marker

        # Normalize target
        target_invalid = False
        for marker in invalid_markers:
            if np.any(targets[idx] == marker):
                target_invalid = True
                break

        if target_invalid:
            targets[idx][:] = normalized_invalid_marker
        elif target_max > target_min:
            targets[idx][:] = 2.0 * (targets[idx] - target_min) / (target_max - target_min) - 1.0

    print(f"   Done! Normalized {len(features)} samples")

    # Save normalized data
    print(f"\n5. Saving normalized data to {output_path}...")
    np.savez_compressed(
        output_path,
        features=features,
        targets=targets,
        masks=masks,
        target_stations=target_stations,
        end_dates=end_dates,
        start_dates=start_dates,
        is_normalized=np.array([True], dtype=bool)
    )

    print("\n" + "=" * 60)
    print("✓ CONVERSION COMPLETE!")
    print("=" * 60)
    print(f"✓ Normalized data saved to: {output_path}")
    print(f"✓ Normalization stats saved to: {norm_stats_path}")
    print(f"\nNext steps:")
    print(f"1. Backup old file (optional):")
    print(f"     cp {precomputed_path} {precomputed_path}.backup")
    print(f"2. Replace with normalized version:")
    print(f"     mv {output_path} {precomputed_path}")
    print(f"3. Run your training:")
    print(f"     python Moisturizer.py")
    print(f"\nExpected performance: 1,000-10,000+ samples/sec!")
    print("=" * 60)

if __name__ == "__main__":
    collector = MeteoGaliciaCollector()

    precomputed_path = collector.data_dir / "precomputed_sequences.npz"
    output_path = collector.data_dir / "precomputed_sequences_normalized.npz"
    norm_stats_path = collector.data_dir / "normalization_stats.npz"

    if not precomputed_path.exists():
        print(f"✗ Precomputed file not found at {precomputed_path}")
        print("  Nothing to convert!")
        exit(1)

    normalize_precomputed_data(
        str(precomputed_path),
        str(output_path),
        str(norm_stats_path)
    )
