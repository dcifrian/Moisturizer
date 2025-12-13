# Code Inventory

Quick reference for the main codebase files. Line counts in brackets.

---

## Moisturizer.py (~1000 lines)

Main entry point module with utility functions and constants.

### Constants

- `INVALID_MARKER_API = -9999.0` - API missing value marker
- `INVALID_MARKER_MISSING = -1000.0` - Internal missing value marker
- `NORMALIZED_INVALID_MARKER = -2.0` - Normalized missing value (outside [-1,1])
- `DEFAULT_COVERAGE_THRESHOLD = 0.25` - 25% data coverage required

### Classes

**FeatureLayout** [L142]
Calculate feature layout dimensions for sequences.
- `n_params`, `n_nearby` - Configuration
- `n_target_features`, `nearby_features_per_station`, `n_total_features` - Computed dimensions

### Functions

| Function | Lines | Description |
|----------|-------|-------------|
| `normalize_features` | 50 | Normalize features to [-1, 1] with invalid marker handling |
| `normalize_target` | 35 | Normalize target value to [-1, 1] |
| `denormalize_target` | 15 | Convert normalized value back to original scale |
| `expand_canonical_to_augmented_stats` | 100 | Expand per-slot stats to full feature layout |
| `_load_precomputed_data` | 38 | Load precomputed sequences from directory |
| `build_dense_feature_array` | 138 | Build dense [stations, dates, features] array |
| `regenerate_nearest_stations` | 45 | Regenerate nearest_stations.csv with more neighbors |
| `buildDataset` | ~250 | Main entry: download data, build and save dataset |
| `loadDataset` | ~135 | Load dataset with optional live augmentation |

---

## MeteoGaliciaCollector.py (~706 lines)

Handles data collection from MeteoGalicia API.

### Classes

**MeteoGaliciaCollector** [L35]
- `data_dir`, `timeseries_file`, `stations_file`, `nearest_file` - Path attributes
- `ALL_SENSORS` - List of all 42 sensor parameter IDs
- `_timeseries_df`, `_stations_df`, `_nearest_df` - Cached DataFrames

| Method | Lines | Description |
|--------|-------|-------------|
| `__init__` | 25 | Initialize paths for data storage |
| `get_timeseries_df` | 10 | Get cached timeseries DataFrame |
| `get_stations_df` | 10 | Get cached stations DataFrame |
| `get_nearest_df` | 10 | Get cached nearest DataFrame |
| `get_all_stations` | 31 | Fetch all weather stations from API |
| `check_soil_moisture_availability` | 35 | Check if station has soil moisture sensor |
| `discover_stations_with_soil_moisture` | 57 | Find and cache stations with soil moisture |
| `calculate_nearest_stations` | 93 | Compute nearest neighbors for each station |
| `get_daily_data` | 41 | Fetch daily data for station/parameter |
| `parse_data_to_dataframe` | 40 | Parse JSON API response to DataFrame |
| `build_historical_dataset` | 67 | Build multi-year historical timeseries |
| `analyze_parameter_coverage` | 115 | Filter parameters by data coverage threshold |
| `get_live_prediction_data` | 88 | Fetch recent data for live predictions |
| `get_sequence_data` | 90 | Get raw sequence for station/date |

---

## WeatherSequenceDataset.py (~1483 lines)

PyTorch Dataset for weather parameter prediction sequences (default: soil moisture).

### Classes

**WeatherSequenceDataset** [L13, extends Dataset]
- `n_nearest`, `seq_length`, `feature_params` - Configuration
- `sample_index` - List of (station, date_range) samples
- `dense_arrays` - Optional fast-access dense feature arrays
- `norm_stats` - Normalization statistics dict

| Method | Lines | Description |
|--------|-------|-------------|
| `__init__` | 260 | Initialize dataset, load data, build index |
| `_build_sample_index` | 62 | Create list of valid samples |
| `_get_nearest_stations` | 25 | Get nearest neighbors for station |
| `_build_sequence_tensor` | 102 | Build sequence via dict lookup (slow) |
| `_build_sequence_from_dense` | 180 | Build sequence via dense array slicing (fast) |
| `_compute_norm_stats_from_precomputed` | 130 | Compute min/max from precomputed data |
| `_apply_normalization` | 43 | Normalize features to [-1, 1] |
| `precompute_and_save` | 235 | Precompute all sequences and save |
| `__len__` | 6 | Return number of samples |
| `__getitem__` | 83 | Get single sample |
| `get_feature_names` | 20 | Return list of feature names |
| `_split_precomputed` | 36 | Split precomputed data by stations |
| `_create_split_dataset` | 50 | Create dataset subset |
| `train_val_test_split` | 100 | Split into train/val/test by stations |

---

## create_moisture_map.py (~2317 lines)

Map visualization and virtual station prediction.

### Functions

| Function | Lines | Description |
|----------|-------|-------------|
| `load_coastline_data` | 140 | Load Galicia coastline from Natural Earth |
| `predict_for_station` | 38 | Run model prediction for real station |
| `denormalize_soil_moisture` | 11 | Convert [-1,1] back to original scale |
| `get_real_soil_moisture_from_lookup` | 11 | Get actual measurement from lookup |
| `build_fast_timeseries_lookup` | 39 | Build (station,date,param)->value dict |
| `build_sequence_for_any_station` | 125 | Build input sequence for any station |
| `apply_normalization_to_features` | 41 | Apply min-max normalization |
| `create_virtual_grid_stations` | 43 | Create grid of virtual stations |
| `find_nearest_real_stations` | 62 | Find N nearest real stations to virtual |
| `interpolate_coordinate_features` | 50 | IDW interpolation for coordinates |
| `load_srtm_elevation_data` | 68 | Load SRTM elevation raster |
| `get_elevation_from_srtm` | 33 | Get elevation from SRTM at lat/lon |
| `point_in_triangle` | 15 | Check if point is inside triangle |
| `triangle_area` | 5 | Calculate triangle area |
| `select_triangle_stations` | 68 | Select 3 stations forming smallest enclosing triangle |
| `build_sequence_for_virtual_station` | 65 | Build interpolated sequence for virtual station |
| `find_nearest_stations_with_soil_moisture` | 58 | Find nearest stations with soil sensors |
| `debug_find_worst_offenders` | 184 | Find stations with worst prediction errors |
| `create_moisture_map` | ~500 | Main: create soil moisture prediction map |
| `compute_cumulative_weather` | 61 | Compute cumulative rain/sun for visualization |
| `create_weather_visualization` | 146 | Create weather parameter map |
| `create_visualization` | 249 | Create final map PNG with legend |

---

## precompute_augmented.py (~867 lines)

Precompute augmented dataset with skip patterns and permutations.

### Functions

| Function | Lines | Description |
|----------|-------|-------------|
| `build_skip_patterns` | 29 | Build skip patterns for augmentation (shared utility) |
| `_process_batch_direct_write` | 135 | Worker: process batch and write to memmap |
| `_setup_augmentation` | ~110 | Common setup for augmentation generation |
| `_create_memmap_arrays` | 40 | Create memory-mapped arrays for output |
| `_save_augmented_dataset` | 15 | Flush memmap arrays and save metadata |
| `_print_completion_stats` | 20 | Print completion statistics |
| `generate_all_augmentations_sequential` | ~130 | Generate augmented dataset sequentially (low memory) |
| `generate_all_augmentations_batched` | ~250 | Generate augmented dataset in parallel batches |

Key concepts:
- Skip patterns: For n_available=5, n_features=4, creates 5 patterns each dropping 1 station
- Permutations: 24 orderings of 4 nearby stations
- Total augmentation: 5 * 24 = 120x (or 24x if n_available == n_features)

---

## augmented_live.py (~661 lines)

Live (on-the-fly) augmentation during training. Imports `build_skip_patterns` from precompute_augmented.

### Classes

**AugmentedLiveDataset** [L21]
PyTorch Dataset that augments on-the-fly during training.
- `n_nearby_available`, `n_nearby_in_features` - Nearby station counts
- `skip_patterns`, `all_permutations` - Augmentation patterns
- `_aug_column_indices` - Precomputed column indices per augmentation
- `total_augmentations` - Number of augmentations per base sample

| Method | Lines | Description |
|--------|-------|-------------|
| `__init__` | 69 | Initialize with base data and augmentation config |
| `from_base_dataset` | 114 | Factory: create from file paths |
| `_from_soil_moisture_dataset` | 59 | Factory: create from existing dataset |
| `_load_base_dataset` | 26 | Load precomputed base sequences |
| `_infer_n_params` | 10 | Infer n_params from feature dimensions |
| `_build_augmentation_patterns` | 5 | Build patterns using build_skip_patterns() |
| `_build_column_indices` | 37 | Precompute column indices for fast slicing |
| `load_normalization_stats` | 53 | Load and expand canonical stats |
| `__len__` | 6 | Return total samples (base * augmentations) |
| `_get_base_and_aug_idx` | 18 | Convert flat index to (base_idx, aug_idx) |
| `__getitem__` | 45 | Get augmented sample via column indexing |
| `_apply_normalization` | 26 | Apply normalization with invalid markers |
| `train_val_test_split` | 61 | Split by station preserving augmentation |
| `_create_split` | 61 | Create subset for train/val/test |
| `get_feature_names` | 18 | Return feature names |

---

## tests.py (~439 lines)

Comprehensive test suite for the dataset pipeline.

### Functions

| Function | Lines | Description |
|----------|-------|-------------|
| `test_feature_layout` | 30 | Test FeatureLayout class calculations |
| `test_normalize_functions` | 50 | Test normalization edge cases |
| `test_dataset_build` | 70 | Test buildDataset() and dataset creation |
| `test_normalization_stats` | 60 | Test canonical stats computation |
| `test_live_augmented_dataset` | 80 | Test AugmentedLiveDataset |
| `test_precomputed_augmented_dataset` | 80 | Test precompute_augmented |
| `main` | 50 | Run all tests and report results |

---

## Feature Layout

For n_params weather parameters and n_nearby nearby stations:
- Target features: n_params (weather for target station)
- Per nearby station: 1 (distance) + n_params (weather) + 1 (target param value)
- Total: n_params + n_nearby * (n_params + 2)

Example with n_params=23, n_nearby=4:
- Target: 23 features
- Nearby: 4 * 25 = 100 features
- Total: 123 features per timestep
