#!/usr/bin/env python3
"""
Check if the leakage exists in RAW data or is just a normalization coincidence
"""

import numpy as np
from pathlib import Path
from Moisturizer import MeteoGaliciaCollector

collector = MeteoGaliciaCollector()

print("=" * 70)
print("CHECKING RAW VS NORMALIZED LEAKAGE")
print("=" * 70)

# Load normalization stats
norm_stats_path = collector.data_dir / "normalization_stats.npz"
if not norm_stats_path.exists():
    print(f"\n✗ Normalization stats not found: {norm_stats_path}")
    exit(1)

norm_stats = np.load(norm_stats_path)
feature_mins = norm_stats['feature_mins']
feature_maxs = norm_stats['feature_maxs']
target_min = norm_stats['target_min']
target_max = norm_stats['target_max']

print(f"\nNormalization stats loaded:")
print(f"  Features: {len(feature_mins)} columns")
print(f"  Target range: [{target_min:.2f}, {target_max:.2f}]")

# Check if we have dense arrays (raw data)
dense_path = collector.data_dir / "dense_features.npz"
if not dense_path.exists():
    print(f"\n✗ Dense features not found: {dense_path}")
    exit(1)

dense_data = np.load(dense_path)
dense_features = dense_data['features']
dense_feature_params = dense_data['feature_params'].tolist()

print(f"\nDense arrays loaded:")
print(f"  Shape: {dense_features.shape}")
print(f"  Parameters: {len(dense_feature_params)}")

# Feature 15 in the final sequences
# Structure: 23 target features + 4*(1 distance + 23 weather + 1 soil)
# So feature 15 is target station feature at index 15
print(f"\n" + "=" * 70)
print(f"ANALYZING FEATURE 15")
print(f"=" * 70)

# Feature 15 should be the 16th parameter in self.feature_params (0-indexed)
# From the diagnostic, feature_params are sorted:
feature_params_sorted = ['BH_SUM_1.5m', 'DVP_MODA_2m', 'DV_CONDICION_2m', 'ET0_SUM_1.5m',
                         'HFRIO7_RECUENTO_1.5m', 'HF_SUM_2m', 'HR_AVG_1.5m', 'HR_MAX_1.5m',
                         'HR_MIN_1.5m', 'HSOL_SUM_1.5m', 'INS_RATIO_1.5m', 'IRD_SUM_1.5m',
                         'PP_SUM_1.5m', 'PRED_AVG_1.5m', 'PR_AVG_1.5m', 'TA_AVG_0.1m',
                         'TA_AVG_1.5m', 'TA_MAX_1.5m', 'TA_MIN_1.5m', 'TO_AVG_1.5m',
                         'TS_AVG_-0.1m', 'VV_AVG_2m', 'VV_MAX_2m']

if len(feature_params_sorted) > 15:
    feature_15_name = feature_params_sorted[15]
    print(f"Feature 15 should be: {feature_15_name}")

    # Get its normalization stats
    feat_15_min = feature_mins[15]
    feat_15_max = feature_maxs[15]
    print(f"  Normalization range: [{feat_15_min:.2f}, {feat_15_max:.2f}]")
else:
    print(f"⚠️  Only {len(feature_params_sorted)} features, can't determine feature 15")
    feature_15_name = "unknown"

print(f"\nTarget (soil moisture) normalization range: [{target_min:.2f}, {target_max:.2f}]")

# Now let's denormalize the leaky sample's values
print(f"\n" + "=" * 70)
print(f"SAMPLE 8 ANALYSIS")
print(f"=" * 70)

# From check_precomputed_leakage.py:
# Sample 8: Target (normalized) = -0.2340, Feature 15 (normalized, timestep 1) = -0.2336

target_norm = -0.2340
feature_15_norm = -0.2336

# Denormalize
# Normalized value = 2.0 * (raw - min) / (max - min) - 1.0
# Solving for raw: raw = (normalized + 1.0) / 2.0 * (max - min) + min

target_raw = (target_norm + 1.0) / 2.0 * (target_max - target_min) + target_min
feature_15_raw = (feature_15_norm + 1.0) / 2.0 * (feat_15_max - feat_15_min) + feat_15_min

print(f"\nDenormalized values:")
print(f"  Target (soil moisture): {target_raw:.4f} (normalized: {target_norm:.4f})")
print(f"  Feature 15 ({feature_15_name}): {feature_15_raw:.4f} (normalized: {feature_15_norm:.4f})")
print(f"  Raw difference: {abs(target_raw - feature_15_raw):.4f}")

if abs(target_raw - feature_15_raw) < 0.01:
    print(f"\n❌ RAW VALUES ARE VERY CLOSE!")
    print(f"  This suggests REAL DATA LEAKAGE!")
    print(f"  Feature 15 contains actual soil moisture data instead of {feature_15_name}!")
else:
    print(f"\n✓ Raw values are DIFFERENT")
    print(f"  This is likely a COINCIDENCE - different parameters normalized to similar values")
    print(f"  Feature 15 is correctly {feature_15_name}, not soil moisture")

# Additional check: Do the normalization ranges overlap significantly?
print(f"\n" + "=" * 70)
print(f"NORMALIZATION RANGE ANALYSIS")
print(f"=" * 70)

# Check if soil moisture range is within the range of feature 15
if feat_15_min <= target_min <= feat_15_max or feat_15_min <= target_max <= feat_15_max:
    print(f"⚠️  Normalization ranges OVERLAP!")
    print(f"  {feature_15_name}: [{feat_15_min:.2f}, {feat_15_max:.2f}]")
    print(f"  Soil moisture: [{target_min:.2f}, {target_max:.2f}]")
    print(f"  This makes coincidental matches MORE LIKELY")
else:
    print(f"Normalization ranges don't overlap significantly")
    print(f"  {feature_15_name}: [{feat_15_min:.2f}, {feat_15_max:.2f}]")
    print(f"  Soil moisture: [{target_min:.2f}, {target_max:.2f}]")
