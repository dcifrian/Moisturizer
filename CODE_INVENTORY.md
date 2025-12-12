# Code Inventory

Quick reference for the main codebase files. Line counts in brackets.

---

## Moisturizer.py (~3300 lines)

Main dataset building and loading module.

### Classes

**MeteoGaliciaCollector** [L32]
Handles data collection from MeteoGalicia API.
- `data_dir`, `timeseries_file`, `stations_file`, `nearest_file` - Path attributes
- `ALL_SENSORS` - List of all 42 sensor parameter IDs

| Method | Lines | Description |
|--------|-------|-------------|
| `__init__` | 9 | Initialize paths for data storage |
| `get_all_stations` | 31 | Fetch all weather stations from API |
| `check_soil_moisture_availability` | 35 | Check if station has soil moisture sensor |
| `discover_stations_with_soil_moisture` | 57 | Find and cache stations with soil moisture |
| `calculate_nearest_stations` | 93 | Compute nearest neighbors for each station |
| `get_daily_data` | 41 | Fetch daily data for station/parameter |
| `parse_data_to_dataframe` | 40 | Parse JSON API response to DataFrame |
| `build_historical_dataset` | 67 | Build multi-year historical timeseries |
| `create_ml_ready_dataset` | 102 | Create ML-ready sequences (deprecated) |
| `analyze_parameter_coverage` | 79 | Filter parameters by data coverage threshold |
| `get_live_prediction_data` | 88 | Fetch recent data for live predictions |

**SoilMoistureSequenceDataset** [L815, extends _BaseDataset]
PyTorch Dataset for soil moisture prediction sequences.
- `n_nearest`, `seq_length`, `feature_params` - Configuration
- `sample_index` - List of (station, date_range) samples
- `dense_arrays` - Optional fast-access dense feature arrays
- `norm_stats` - Normalization statistics dict

| Method | Lines | Description |
|--------|-------|-------------|
| `__init__` | 260 | Initialize dataset, load data, build index |
| `_build_sample_index` | 62 | Create list of valid samples |
| `_get_nearest_stations` | 8 | Get nearest neighbors for station |
| `_build_sequence_tensor` | 102 | Build sequence via dict lookup (slow) |
| `_build_sequence_from_dense` | 106 | Build sequence via dense array slicing (fast) |
| `_compute_norm_stats_from_precomputed` | 53 | Compute min/max from precomputed data |
| `_apply_normalization` | 43 | Normalize features to [-1, 1] |
| `precompute_and_save` | 120 | Precompute all sequences and save |
| `__len__` | 3 | Return number of samples |
| `__getitem__` | 83 | Get single sample |
| `get_feature_names` | 17 | Return list of feature names |
| `_split_precomputed` | 36 | Split precomputed data by stations |
| `_create_split_dataset` | 43 | Create dataset subset |
| `train_val_test_split` | 91 | Split into train/val/test by stations |
| `get_sequence_data` | 90 | Get raw sequence for station/date |

### Functions

| Function | Lines | Description |
|----------|-------|-------------|
| `_decompress_npz_if_needed` | 50 | Convert compressed .npz to memory-mappable .npy |
| `_load_precomputed_data` | 38 | Load precomputed sequences from directory |
| `build_dense_feature_array` | 138 | Build dense [stations, dates, features] array |
| `buildDataset` | ~150 | Main entry: download data, build and save dataset |
| `precomputeDataset` | 44 | Precompute sequences for existing dataset |
| `loadDataset` | ~120 | Load dataset with optional live augmentation |

---

## create_moisture_map.py (~2200 lines)

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
| `get_elevation_from_open_elevation` | 28 | Fallback: get elevation from API |
| `point_in_triangle` | 15 | Check if point is inside triangle |
| `triangle_area` | 5 | Calculate triangle area |
| `select_triangle_stations` | 68 | Select 3 stations forming smallest enclosing triangle |
| `build_sequence_for_virtual_station` | 65 | Build interpolated sequence for virtual station |
| `find_nearest_stations_with_soil_moisture` | 58 | Find nearest stations with soil sensors |
| `debug_find_worst_offenders` | 184 | Debug: find stations with worst prediction errors |
| `create_moisture_map` | ~500 | Main: create soil moisture prediction map |
| `compute_cumulative_weather` | 61 | Compute cumulative rain/sun for visualization |
| `create_weather_visualization` | 146 | Create weather parameter map |
| `create_visualization` | 249 | Create final map PNG with legend |

---

## precompute_augmented.py (~1000 lines)

Precompute augmented dataset with skip patterns and permutations.

### Functions

| Function | Lines | Description |
|----------|-------|-------------|
| `_process_batch_direct_write` | 183 | Worker: process batch and write to memmap |
| `_process_samples_worker_v2` | 97 | Alternative worker for sample processing |
| `generate_all_augmentations_sequential` | ~450 | Generate augmented dataset sequentially (low memory) |
| `generate_all_augmentations_batched` | ~250 | Generate augmented dataset in parallel batches |

Key concepts:
- Skip patterns: For n_available=5, n_features=4, creates 5 patterns each dropping 1 station
- Permutations: 24 orderings of 4 nearby stations
- Total augmentation: 5 * 24 = 120x (or 24x if n_available == n_features)

---

## augmented_live.py (~1050 lines)

Live (on-the-fly) augmentation during training.

### Classes

**AugmentedLiveDataset** [L12]
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
| `_build_augmentation_patterns` | 15 | Build skip patterns and permutations |
| `_build_column_indices` | 37 | Precompute column indices for fast slicing |
| `_compute_normalization_stats` | 83 | Expand canonical stats to augmented layout |
| `__len__` | 6 | Return total samples (base * augmentations) |
| `_get_base_and_aug_idx` | 18 | Convert flat index to (base_idx, aug_idx) |
| `__getitem__` | 45 | Get augmented sample via column indexing |
| `_apply_normalization` | 41 | Apply normalization with invalid markers |
| `train_val_test_split` | 61 | Split by station preserving augmentation |
| `_create_split` | 61 | Create subset for train/val/test |
| `get_feature_names` | 18 | Return feature names |

**AugmentedPrecomputedDataset** [L699]
Dataset for loading precomputed augmented data.
- `features`, `targets`, `masks` - Memory-mapped arrays

| Method | Lines | Description |
|--------|-------|-------------|
| `__init__` | 45 | Load precomputed augmented arrays |
| `__len__` | 5 | Return number of samples |
| `__getitem__` | 13 | Get sample from memmap |
| `train_val_test_split` | 46 | Split by station |
| `_create_split` | 25 | Create subset |

### Functions

| Function | Lines | Description |
|----------|-------|-------------|
| `benchmark_dataset` | 215 | Benchmark dataset loading speed |

---

## Key Constants

- `INVALID_MARKER_API = -9999.0` - API missing value marker
- `INVALID_MARKER_MISSING = -1000.0` - Internal missing value marker
- `NORMALIZED_INVALID = -2.0` - Normalized missing value (outside [-1,1])
- `DEFAULT_COVERAGE_THRESHOLD = 0.25` - 25% data coverage required

## Feature Layout (FeatureLayout class in Moisturizer.py)

For n_params weather parameters and n_nearby nearby stations:
- Target features: n_params (weather for target station)
- Per nearby station: 1 (distance) + n_params (weather) + 1 (soil moisture)
- Total: n_params + n_nearby * (n_params + 2)

Example with n_params=23, n_nearby=4:
- Target: 23 features
- Nearby: 4 * 25 = 100 features
- Total: 123 features per timestep
