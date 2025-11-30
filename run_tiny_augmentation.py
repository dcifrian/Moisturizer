#!/usr/bin/env python3
"""Run precompute_augmented with seq_length=2 to match the tiny dataset"""
from precompute_augmented import generate_all_augmentations_batched

generate_all_augmentations_batched(
    data_dir="./meteogalicia_data",
    n_nearby_available=5,
    n_nearby_in_features=4,
    coverage_threshold=0.25,
    seq_length=2,  # Match the tiny dataset!
    batch_size=10,  # Small batches
    num_workers=2,  # Just 2 workers for tiny dataset
    use_base_stats=False  # Don't use base stats - we want to see the corruption
)
