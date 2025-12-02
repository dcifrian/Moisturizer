#!/usr/bin/env python3
"""Check what's actually in dense_features.npz"""
import numpy as np

print("="*70)
print("CHECKING dense_features.npz")
print("="*70)

dense = np.load('./meteogalicia_data/dense_features.npz')

print("\nFiles in archive:", list(dense.keys()))

features = dense['features']
print(f"\nFeatures shape: {features.shape}")
print(f"  (stations, dates, features) = ({features.shape[0]}, {features.shape[1]}, {features.shape[2]})")

# Check soil moisture column (should be last column, index -1)
soil_moisture = features[:, :, -1]  # Last feature is soil moisture

print(f"\nSoil moisture column (last feature):")
print(f"  Shape: {soil_moisture.shape}")
print(f"  Range: [{soil_moisture.min():.2f}, {soil_moisture.max():.2f}]")
print(f"  Unique values: {len(np.unique(soil_moisture))}")

# Count valid vs invalid
invalid_count = np.sum((soil_moisture == -1000.0) | (soil_moisture == -9999.0))
valid_count = np.sum((soil_moisture != -1000.0) & (soil_moisture != -9999.0))
total = soil_moisture.size

print(f"\nSoil moisture validity:")
print(f"  Valid values: {valid_count:,} ({100*valid_count/total:.1f}%)")
print(f"  Invalid (-1000 or -9999): {invalid_count:,} ({100*invalid_count/total:.1f}%)")

# Show some valid values if any exist
valid_values = soil_moisture[(soil_moisture != -1000.0) & (soil_moisture != -9999.0)]
if len(valid_values) > 0:
    print(f"\n  Valid soil moisture range: [{valid_values.min():.3f}, {valid_values.max():.3f}]")
    print(f"  First 20 valid values: {valid_values.flatten()[:20]}")
else:
    print(f"\n  ❌ NO VALID SOIL MOISTURE VALUES FOUND!")

# Check other features too
print(f"\nChecking all features:")
for i in range(features.shape[2]):
    feat_data = features[:, :, i]
    valid = np.sum((feat_data != -1000.0) & (feat_data != -9999.0))
    print(f"  Feature {i}: {valid:,}/{total:,} valid ({100*valid/total:.1f}%)")
