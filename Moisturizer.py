"""
MeteoGalicia Weather Station Data Collector
Collects historical and live data from MeteoGalicia API for ML model training
with focus on soil moisture prediction from nearby stations
"""


import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Union
from pathlib import Path
from MeteoGaliciaCollector import MeteoGaliciaCollector

# Model loader - optional, only needed for inference
try:
    from model_loader import load_model
    MODEL_LOADER_AVAILABLE = True
except ImportError:
    MODEL_LOADER_AVAILABLE = False
    load_model = None  # Will raise error if actually used

# Invalid/missing data markers
INVALID_MARKER_API = -9999.0       # Value returned by MeteoGalicia API for missing data
INVALID_MARKER_MISSING = -1000.0   # Value we use to mark missing/unavailable data
NORMALIZED_INVALID_MARKER = -2.0   # Value used after normalization (outside [-1, 1] range)

# Default parameters
DEFAULT_COVERAGE_THRESHOLD = 0.25  # Minimum data coverage for a parameter to be included


def normalize_features(features, feature_mins, feature_maxs,
                       invalid_markers=None, normalized_invalid_marker=None,
                       inplace=False):
    """
    Normalize features to [-1, 1] range using min-max scaling.

    Vectorized implementation that handles invalid markers.

    Args:
        features: np.ndarray of shape [..., n_features] - features to normalize
        feature_mins: np.ndarray of shape [n_features] - minimum values per feature
        feature_maxs: np.ndarray of shape [n_features] - maximum values per feature
        invalid_markers: List of values to treat as invalid (default: [-9999.0, -1000.0])
        normalized_invalid_marker: Value to use for invalid data (default: -2.0)
        inplace: If True, modify features array in place (default: False)

    Returns:
        Normalized features array (same shape as input)
    """
    if invalid_markers is None:
        invalid_markers = [INVALID_MARKER_API, INVALID_MARKER_MISSING]
    if normalized_invalid_marker is None:
        normalized_invalid_marker = NORMALIZED_INVALID_MARKER

    if not inplace:
        features = features.copy()

    # Find invalid values before normalization
    invalid_mask = np.isin(features, invalid_markers)

    # Compute ranges (vectorized)
    feat_ranges = feature_maxs - feature_mins
    valid_ranges = feat_ranges > 0

    # Normalize: scale to [-1, 1]
    # For features with shape [seq_length, n_features] or [batch, seq_length, n_features]
    # We broadcast mins/maxs along the last axis
    if features.ndim == 2:
        features[:] = 2.0 * (features - feature_mins[None, :]) / np.where(
            valid_ranges[None, :], feat_ranges[None, :], 1.0) - 1.0
    elif features.ndim == 3:
        features[:] = 2.0 * (features - feature_mins[None, None, :]) / np.where(
            valid_ranges[None, None, :], feat_ranges[None, None, :], 1.0) - 1.0
    else:
        # Fallback for other shapes - normalize along last axis
        features[:] = 2.0 * (features - feature_mins) / np.where(valid_ranges, feat_ranges, 1.0) - 1.0

    # Set invalid values to marker
    features[invalid_mask] = normalized_invalid_marker

    return features


def normalize_target(target, target_min, target_max,
                     invalid_markers=None, normalized_invalid_marker=None):
    """
    Normalize a target value to [-1, 1] range using min-max scaling.

    Args:
        target: Scalar, np.ndarray, or value to normalize
        target_min: Minimum target value
        target_max: Maximum target value
        invalid_markers: List of values to treat as invalid (default: [-9999.0, -1000.0])
        normalized_invalid_marker: Value to use for invalid data (default: -2.0)

    Returns:
        Normalized target value (same type as input for scalars, np.float32 for arrays)
    """
    if invalid_markers is None:
        invalid_markers = [INVALID_MARKER_API, INVALID_MARKER_MISSING]
    if normalized_invalid_marker is None:
        normalized_invalid_marker = NORMALIZED_INVALID_MARKER

    # Handle array input
    if isinstance(target, np.ndarray):
        target_val = target.item() if target.ndim == 0 else target[0]
    else:
        target_val = target

    # Check for invalid
    if target_val in invalid_markers:
        return normalized_invalid_marker

    # Normalize to [-1, 1]
    if target_max > target_min:
        return 2.0 * (target_val - target_min) / (target_max - target_min) - 1.0
    else:
        return target_val


def denormalize_target(normalized_value, target_min, target_max):
    """
    Convert a normalized target value back to original scale.

    Args:
        normalized_value: Value in [-1, 1] range (or NORMALIZED_INVALID_MARKER for invalid)
        target_min: Original minimum target value
        target_max: Original maximum target value

    Returns:
        Value in original scale, or None if input was invalid marker
    """
    if normalized_value == NORMALIZED_INVALID_MARKER:
        return None

    # Denormalize: [-1, 1] -> [min, max]
    return (normalized_value + 1.0) / 2.0 * (target_max - target_min) + target_min


class FeatureLayout:
    """
    Calculate and store feature layout dimensions for soil moisture prediction.

    The feature layout consists of:
    - Target station features: n_params weather/coordinate parameters
    - For each nearby station: 1 (distance) + n_params (features) + 1 (soil moisture)

    Usage:
        layout = FeatureLayout(n_params=26, n_nearby=4)
        print(layout.n_target_features)  # 26
        print(layout.nearby_features_per_station)  # 28 (1 + 26 + 1)
        print(layout.n_total_features)  # 138 (26 + 28*4)
    """
    __slots__ = ('n_params', 'n_nearby', 'n_target_features',
                 'nearby_features_per_station', 'n_total_features')

    def __init__(self, n_params: int, n_nearby: int):
        """
        Initialize feature layout.

        Args:
            n_params: Number of weather/coordinate parameters per station
            n_nearby: Number of nearby stations in the feature vector
        """
        self.n_params = n_params
        self.n_nearby = n_nearby

        # Target station features (just the parameters)
        self.n_target_features = n_params

        # Nearby station features: distance + params + soil moisture
        self.nearby_features_per_station = 1 + n_params + 1

        # Total features: target + (nearby features * n_nearby)
        self.n_total_features = self.n_target_features + (
            self.nearby_features_per_station * n_nearby
        )

    def __repr__(self):
        return (f"FeatureLayout(n_params={self.n_params}, n_nearby={self.n_nearby}, "
                f"total={self.n_total_features})")

    def nearby_start_idx(self, slot: int) -> int:
        """Get the starting feature index for a nearby station slot (0-indexed)."""
        return self.n_target_features + (slot * self.nearby_features_per_station)

    def nearby_end_idx(self, slot: int) -> int:
        """Get the ending feature index (exclusive) for a nearby station slot."""
        return self.nearby_start_idx(slot) + self.nearby_features_per_station


def expand_canonical_to_augmented_stats(canonical_stats, n_params, n_nearby_in_features,
                                        n_nearby_available=None, augmented=False):
    """
    Expand per-slot canonical stats to the feature layout needed for inference.

    Canonical stats have:
        - target_feature_mins/maxs: [n_params] for target station features
        - nearby_slot_mins/maxs: [n_nearby_slots, nearby_features_per_station] per-slot stats
        - target_min/max: scalar for target (soil moisture prediction target)
        - n_nearby_slots: number of slots stored in the canonical stats

    Output layout is:
        - [n_params] target features
        - For each of n_nearby_in_features stations: [1 + n_params + 1] = (distance, params, soil)

    For non-augmented models: use per-slot stats directly (first n_nearby_in_features slots)
    For augmented models: compute min/max across n_nearby_available slots for each feature type

    Args:
        canonical_stats: dict-like with canonical stat arrays
        n_params: number of weather parameters per station
        n_nearby_in_features: number of nearby stations in the model's input
        n_nearby_available: for augmented, how many nearby were available for permutations
                           (ignored if augmented=False)
        augmented: whether the model was trained with augmentation

    Returns:
        dict with 'feature_mins', 'feature_maxs', 'target_min', 'target_max'
    """
    target_feat_mins = np.asarray(canonical_stats['target_feature_mins'])
    target_feat_maxs = np.asarray(canonical_stats['target_feature_maxs'])

    # Handle 0-d arrays for target_min/max
    target_min_val = canonical_stats['target_min']
    target_max_val = canonical_stats['target_max']
    if hasattr(target_min_val, 'ndim'):
        target_min = float(target_min_val.item()) if target_min_val.ndim == 0 else float(target_min_val[0])
        target_max = float(target_max_val.item()) if target_max_val.ndim == 0 else float(target_max_val[0])
    else:
        target_min = float(target_min_val)
        target_max = float(target_max_val)

    # Require new per-slot format
    if 'nearby_slot_mins' not in canonical_stats:
        raise ValueError(
            "Stats file missing 'nearby_slot_mins' (old format not supported). "
            "Regenerate the base dataset with buildDataset() to create new format stats."
        )

    # New per-slot format
    nearby_slot_mins = np.asarray(canonical_stats['nearby_slot_mins'])
    nearby_slot_maxs = np.asarray(canonical_stats['nearby_slot_maxs'])
    n_slots_available = nearby_slot_mins.shape[0]
    nearby_features_per_station = nearby_slot_mins.shape[1]

    # Build output layout
    n_output_features = n_params + n_nearby_in_features * nearby_features_per_station

    feature_mins = np.full(n_output_features, np.inf, dtype=np.float32)
    feature_maxs = np.full(n_output_features, -np.inf, dtype=np.float32)

    # Target station features (same for any configuration)
    feature_mins[:n_params] = target_feat_mins[:n_params]
    feature_maxs[:n_params] = target_feat_maxs[:n_params]

    if augmented:
        # Augmented: each slot sees data from any of the n_nearby_available stations
        # So for each feature type, take min across all available slots and max across all
        if n_nearby_available is None:
            raise ValueError("n_nearby_available required for augmented=True")
        if n_nearby_available > n_slots_available:
            raise ValueError(f"n_nearby_available ({n_nearby_available}) > available slots in stats ({n_slots_available})")

        # Compute aggregated stats across the available slots
        agg_mins = nearby_slot_mins[:n_nearby_available, :].min(axis=0)
        agg_maxs = nearby_slot_maxs[:n_nearby_available, :].max(axis=0)

        # Apply to all output slots (they all see the same range due to permutations)
        for i in range(n_nearby_in_features):
            start_idx = n_params + i * nearby_features_per_station
            end_idx = start_idx + nearby_features_per_station
            feature_mins[start_idx:end_idx] = agg_mins
            feature_maxs[start_idx:end_idx] = agg_maxs
    else:
        # Non-augmented: use per-slot stats directly (full range utilization)
        if n_nearby_in_features > n_slots_available:
            raise ValueError(f"n_nearby_in_features ({n_nearby_in_features}) > available slots in stats ({n_slots_available})")

        for i in range(n_nearby_in_features):
            start_idx = n_params + i * nearby_features_per_station
            end_idx = start_idx + nearby_features_per_station
            feature_mins[start_idx:end_idx] = nearby_slot_mins[i]
            feature_maxs[start_idx:end_idx] = nearby_slot_maxs[i]

    return {
        'feature_mins': feature_mins,
        'feature_maxs': feature_maxs,
        'target_min': target_min,
        'target_max': target_max,
    }

def _load_precomputed_data(precomputed_path: str):
    """
    Load precomputed data from directory of .npy files with true memory-mapping.

    Args:
        precomputed_path: Path to directory containing features.npy, targets.npy, etc.

    Returns dict with memory-mapped array access.
    """
    precomputed_path = Path(precomputed_path)

    # Must be a directory of .npy files
    if not precomputed_path.is_dir():
        raise ValueError(f"Precomputed path must be a directory: {precomputed_path}\n"
                         f"Expected structure: {precomputed_path}/features.npy, targets.npy, etc.")

    # Check for features.npy as indicator that directory has data
    features_file = precomputed_path / 'features.npy'
    if not features_file.exists():
        raise ValueError(f"No features.npy found in {precomputed_path}\n"
                         f"Directory must contain features.npy, targets.npy, masks.npy, etc.")

    print(f"  Loading from directory of .npy files (true memory-mapping)")

    # Load each .npy file individually with mmap_mode
    npy_files = list(precomputed_path.glob("*.npy"))
    if not npy_files:
        raise ValueError(f"No .npy files found in {precomputed_path}")

    data_dict = {}
    for npy_file in npy_files:
        key = npy_file.stem  # filename without .npy
        data_dict[key] = np.load(npy_file, mmap_mode='r')
        print(f"    Loaded {key}: shape={data_dict[key].shape}")

    return data_dict

def build_dense_feature_array(
    timeseries_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    feature_params: List[str],
    soil_moisture_param: str = "HS_CV_AVG_-0.2m",
    missing_value: float = INVALID_MARKER_MISSING
) -> Tuple[np.ndarray, np.ndarray, List[int], pd.DatetimeIndex]:
    """
    Build dense feature array for all stations × all dates × all parameters

    This is MUCH faster than building sequences one-by-one with dict lookups.

    Args:
        timeseries_df: Raw timeseries data
        stations_df: Station metadata
        feature_params: List of parameter codes to include (WITHOUT soil moisture)
        soil_moisture_param: Soil moisture parameter to add separately
        missing_value: Value for missing data

    Returns:
        features_array: [num_stations, num_dates, num_features] array
        mask_array: [num_stations, num_dates, num_features] mask (1=valid, 0=missing)
        station_ids: List of station IDs (index matches array)
        date_index: DatetimeIndex of all dates
    """
    print("=" * 70)
    print("BUILDING DENSE FEATURE ARRAY (ONE-TIME PREPROCESSING)")
    print("=" * 70)

    # Get all unique dates and stations
    all_dates = sorted(timeseries_df['date'].unique())
    date_index = pd.DatetimeIndex(all_dates)
    # Only include stations with soil moisture (they are both targets AND neighbors)
    station_ids = sorted(stations_df[stations_df['has_soil_moisture']]['station_id'].unique())

    # Separate coordinate features from timeseries parameters
    coordinate_features = ['altitude', 'utmx', 'utmy']
    timeseries_params = [p for p in feature_params if p not in coordinate_features]

    # Build combined parameter list: features + soil moisture
    all_params = feature_params + [soil_moisture_param]

    num_stations = len(station_ids)
    num_dates = len(date_index)
    num_features = len(all_params)

    total_stations = len(stations_df)
    print(f"\nArray dimensions (optimized - only soil moisture stations):")
    print(f"  Stations: {num_stations} (out of {total_stations} total - {100 * (1 - num_stations/total_stations):.1f}% reduction)")
    print(f"  Dates: {num_dates}")
    print(f"  Features: {len(timeseries_params)} weather params + {len(coordinate_features)} coordinates + 1 soil moisture = {num_features} total")
    print(f"  Total elements: {num_stations * num_dates * num_features:,}")
    print(f"  Memory: ~{num_stations * num_dates * num_features * 4 / 1e6:.1f} MB")

    # Initialize arrays
    features_array = np.full((num_stations, num_dates, num_features), missing_value, dtype=np.float32)
    mask_array = np.zeros((num_stations, num_dates, num_features), dtype=bool)

    # Create mapping for fast indexing - ensure consistent types
    station_to_idx = {int(sid): idx for idx, sid in enumerate(station_ids)}
    date_to_idx = {date: idx for idx, date in enumerate(date_index)}
    param_to_idx = {str(param): idx for idx, param in enumerate(all_params)}

    print("\nFilling array with data...")

    # Fill coordinate features (static - same for all dates)
    if coordinate_features:
        print(f"  Filling {len(coordinate_features)} coordinate features (static)...")
        stations_with_coords = stations_df[stations_df['station_id'].isin(station_ids)].copy()

        for coord_feat in coordinate_features:
            if coord_feat in all_params and coord_feat in stations_with_coords.columns:
                coord_idx = all_params.index(coord_feat)

                # Fill for each station
                for _, station_row in stations_with_coords.iterrows():
                    sid = int(station_row['station_id'])
                    if sid in station_to_idx:
                        station_idx = station_to_idx[sid]
                        coord_value = station_row[coord_feat]

                        if pd.notna(coord_value):
                            # Fill for ALL dates (static feature)
                            features_array[station_idx, :, coord_idx] = float(coord_value)
                            mask_array[station_idx, :, coord_idx] = True

        print(f"    ✓ Filled coordinate features")
    # Vectorized approach - MUCH faster than iterrows()


    # Pre-filter to only include stations we care about (soil moisture stations)
    timeseries_filtered = timeseries_df[
        timeseries_df['station_id'].isin(station_ids)
    ].copy()

    print(f"  Filtered to {len(timeseries_filtered):,} rows for {num_stations} stations")

    # Map station_id, date, and parameter_code to indices using pandas .map() (fast!)
    # This is vectorized C code, much faster than Python loops
    timeseries_filtered['station_idx'] = timeseries_filtered['station_id'].map(station_to_idx)
    timeseries_filtered['date_idx'] = timeseries_filtered['date'].map(date_to_idx)
    timeseries_filtered['param_idx'] = timeseries_filtered['parameter_code'].map(param_to_idx)

    # Drop rows where mapping failed (station/date/param not in our arrays)
    before_drop = len(timeseries_filtered)
    timeseries_filtered = timeseries_filtered.dropna(subset=['station_idx', 'date_idx', 'param_idx'])
    after_drop = len(timeseries_filtered)
    dropped = before_drop - after_drop

    if dropped > 0:
        print(f"  Dropped {dropped:,} rows with missing indices (expected for non-soil-moisture stations)")

    # Convert index columns to integers
    timeseries_filtered['station_idx'] = timeseries_filtered['station_idx'].astype(np.int32)
    timeseries_filtered['date_idx'] = timeseries_filtered['date_idx'].astype(np.int32)
    timeseries_filtered['param_idx'] = timeseries_filtered['param_idx'].astype(np.int32)

    # Extract numpy arrays for vectorized assignment
    station_indices = timeseries_filtered['station_idx'].values
    date_indices = timeseries_filtered['date_idx'].values
    param_indices = timeseries_filtered['param_idx'].values
    values = timeseries_filtered['value'].values.astype(np.float32)

    print(f"  Assigning {len(values):,} values using vectorized numpy indexing...")

    # VECTORIZED ASSIGNMENT - this is the magic that makes it fast!
    # Instead of looping through millions of rows, we do one bulk assignment
    features_array[station_indices, date_indices, param_indices] = values
    mask_array[station_indices, date_indices, param_indices] = True

    print(f"\n✓ Dense array built!")
    print(f"  Filled {len(values):,} data points")
    print(f"  Valid data: {mask_array.sum():,.0f} ({mask_array.sum() / mask_array.size * 100:.1f}% coverage)")
    print(f"  Missing data: {(mask_array == 0).sum():,.0f}")

    return features_array, mask_array, station_ids, date_index


def regenerate_nearest_stations(n_nearest: Optional[int] = None, data_dir: str = "./meteogalicia_data"):
    """
    Regenerate nearest_stations.csv with a different number of neighbors.

    Use this when you need more neighbors than currently cached (e.g., for augmentation).

    Args:
        n_nearest: Number of nearest neighbors to compute. If None, computes ALL available
                  (n_soil_moisture_stations - 1).
        data_dir: Directory containing the dataset

    Returns:
        DataFrame with nearest stations
    """
    collector = MeteoGaliciaCollector(data_dir=data_dir)

    # Load existing stations (using cache)
    stations_df = collector.get_stations_df()
    if stations_df is None:
        raise FileNotFoundError(f"stations_metadata.csv not found in {data_dir}")

    # Calculate max possible neighbors
    n_soil_moisture = len(stations_df[stations_df['has_soil_moisture'] == True])
    max_neighbors = n_soil_moisture - 1

    if n_nearest is None:
        n_nearest = max_neighbors

    # Check current neighbors
    current_nearest = collector.get_nearest_df()
    if current_nearest is not None:
        n_current = len([c for c in current_nearest.columns if c.startswith('nearest_') and c.endswith('_id')])
        print(f"Current nearest_stations.csv has {n_current} neighbors (max possible: {max_neighbors})")
    else:
        n_current = 0

    if n_current >= n_nearest:
        print(f"Already have {n_current} neighbors, no regeneration needed")
        return current_nearest

    print(f"Regenerating nearest_stations.csv with {n_nearest} neighbors...")
    nearest_df = collector.calculate_nearest_stations(stations_df, n_nearest=n_nearest)
    print(f"✓ Saved to {collector.nearest_file}")

    return nearest_df


def buildDataset(seq_length: int = 64, days: int = 3705, end_date: Optional[datetime] = None,
                 force_refresh: bool = False, n_nearby: int = 4,
                 precompute_augmented: Union[bool, int] = False,
                 coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD):
    """
    Build the complete dataset with optimizations

    Args:
        seq_length: Number of days in each sequence (default: 64)
        days: Number of days of historical data to collect (default: 3705 = ~10 years,
              maximum range without losing stations with moisture data)
        end_date: End date for data collection (default: None = use current date)
        force_refresh: If True, force re-download even if cached data exists
        n_nearby: Number of nearby stations in the precomputed base dataset (default: 4)
        precompute_augmented: Whether to also precompute augmented dataset:
            - False: Only precompute the base dataset (default)
            - True: Precompute augmented using n_nearby as n_nearby_available (permutations only)
            - int: Precompute augmented using this value as n_nearby_available (must be >= n_nearby)
        coverage_threshold: Minimum data coverage for a parameter to be included

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    from SoilMoistureSequenceDataset import SoilMoistureSequenceDataset
    collector = MeteoGaliciaCollector()

    # Determine end date
    if end_date is None:
        actual_end_date = datetime.now()
    else:
        actual_end_date = end_date

    requested_start = (actual_end_date - timedelta(days=days)).strftime('%Y-%m-%d')
    requested_end = actual_end_date.strftime('%Y-%m-%d')

    # Check if we can use cached data by examining the CSV directly
    can_use_cache = False
    timeseries_df = None

    if not force_refresh and collector.timeseries_file.exists():
        try:
            # Load CSV and check date range (using cache)
            timeseries_df = collector.get_timeseries_df()
            cached_dates = pd.to_datetime(timeseries_df['date'])
            cached_start = cached_dates.min().strftime('%Y-%m-%d')
            cached_end = cached_dates.max().strftime('%Y-%m-%d')
            cached_days = (cached_dates.max() - cached_dates.min()).days

            # Cache is valid if:
            # - Number of days matches (within 1 day tolerance for edge cases)
            # - AND (end_date is None OR cached end_date matches requested)
            if abs(cached_days - days) <= 1:
                if end_date is None or cached_end == requested_end:
                    can_use_cache = True
                    print(f"✓ Found cached dataset matching parameters:")
                    print(f"  - Date range: {cached_start} to {cached_end} ({cached_days} days)")
        except Exception as e:
            print(f"  Warning: Could not validate cache: {e}")
            timeseries_df = None

    if can_use_cache:
        print("\n" + "=" * 60)
        print("USING CACHED DATA (skipping download)")
        print("=" * 60)
        # timeseries_df already loaded above (using cache)
        stations_df = collector.get_stations_df()
        nearest_df = collector.get_nearest_df()

        # Calculate max possible neighbors
        n_soil_moisture = len(stations_df[stations_df['has_soil_moisture'] == True])
        max_neighbors = n_soil_moisture - 1

        # Always ensure nearest_df has ALL possible neighbors (it's fast to compute)
        n_neighbors_in_file = len([c for c in nearest_df.columns if c.startswith('nearest_') and c.endswith('_id')])
        if n_neighbors_in_file < max_neighbors:
            print(f"\n⚠ Cached nearest_stations.csv has only {n_neighbors_in_file} neighbors, need {max_neighbors}")
            print(f"  Regenerating nearest_stations.csv with ALL neighbors...")
            nearest_df = collector.calculate_nearest_stations(stations_df, n_nearest=None)
            n_neighbors_in_file = len([c for c in nearest_df.columns if c.startswith('nearest_') and c.endswith('_id')])
            print(f"  ✓ Regenerated with {n_neighbors_in_file} neighbors")
    else:
        # Step 1: Discover stations with soil moisture
        print("=" * 60)
        print("STEP 1: Discovering stations")
        print("=" * 60)
        stations_df = collector.discover_stations_with_soil_moisture(force_refresh=force_refresh)

        # Step 2: Calculate nearest stations (always compute ALL neighbors - it's fast)
        print("\n" + "=" * 60)
        print("STEP 2: Calculating nearest stations")
        print("=" * 60)
        nearest_df = collector.calculate_nearest_stations(stations_df, n_nearest=None)

        # Step 3: Build historical dataset
        print("\n" + "=" * 60)
        print("STEP 3: Building historical dataset")
        print("=" * 60)

        # Get all stations (not just those with soil moisture)
        all_station_ids = stations_df['station_id'].tolist()

        # Use ALL sensors from MeteoGalicia API
        parameters = collector.ALL_SENSORS

        timeseries_df = collector.build_historical_dataset(
            station_ids=all_station_ids,
            parameter_ids=parameters,
            start_date=actual_end_date - timedelta(days=days),
            end_date=actual_end_date,
            chunk_days=30,
            force_refresh=force_refresh
        )

    # Report number of neighbors available in the file
    n_neighbors_in_file = len([c for c in nearest_df.columns if c.startswith('nearest_') and c.endswith('_id')])
    print(f"\n  nearest_stations.csv has {n_neighbors_in_file} neighbors (using {n_nearby} for base dataset)")

    # Step 4: Analyze parameter coverage from timeseries
    print("\n" + "=" * 60)
    print("STEP 4: Analyzing parameter coverage")
    print("=" * 60)

    # Use analyze_parameter_coverage() method to get filtered parameters
    coverage_dict, filtered_params = collector.analyze_parameter_coverage(
        timeseries_df=timeseries_df,
        stations_df=stations_df,
        coverage_threshold=coverage_threshold,
        soil_moisture_param="HS_CV_AVG_-0.2m",
        add_coordinate_features=True
    )

    if not filtered_params:
        print("\n✗ No parameters passed the threshold!")
        return

    print(f"\nUsing {len(filtered_params)} filtered parameters...")

    # Step 5: Build dense feature arrays (FAST!)
    print("\n" + "=" * 60)
    print("STEP 5: Building dense feature arrays")
    print("=" * 60)

    features_array, mask_array, station_ids_list, date_index = build_dense_feature_array(
        timeseries_df=timeseries_df,
        stations_df=stations_df,
        feature_params=filtered_params,
        soil_moisture_param="HS_CV_AVG_-0.2m",
        missing_value=INVALID_MARKER_MISSING
    )

    # Save dense arrays
    dense_array_path = collector.data_dir / "dense_features.npz"
    print(f"\nSaving dense arrays to {dense_array_path}...")
    all_params = filtered_params + ["HS_CV_AVG_-0.2m"]
    np.savez_compressed(
        dense_array_path,
        features=features_array,
        masks=mask_array,
        station_ids=np.array(station_ids_list, dtype=np.int32),
        dates=date_index.values.astype('datetime64[ns]').astype(np.int64),
        feature_params=np.array(all_params, dtype='U50')
    )
    print(f"✓ Saved! File size: {dense_array_path.stat().st_size / 1e6:.1f} MB")

    # Step 6: Create PyTorch Dataset with dense arrays
    print("\n" + "=" * 60)
    print("STEP 6: Creating PyTorch Dataset with dense arrays")
    print("=" * 60)

    dataset = SoilMoistureSequenceDataset(
        timeseries=str(collector.timeseries_file),
        stations=str(collector.stations_file),
        nearest=str(collector.nearest_file),
        seq_length=seq_length,
        n_nearest=n_nearby,  # Use n_nearby for base dataset
        feature_params=filtered_params,
        dense_array_path=str(dense_array_path)
    )

    print(f"\nFeature names: {dataset.get_feature_names()}")

    # Example sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nExample sample:")
        print(f"  Features shape: {sample['features'].shape}")
        print(f"  Target: {sample['target']}")
        print(f"  Mask shape: {sample['mask'].shape}")
        print(f"  Station ID: {sample['target_station_id']}")

    # Step 7: Precompute and save
    print("\n" + "=" * 60)
    print("STEP 7: Precomputing sequences")
    print("=" * 60)

    precomputed_path = collector.data_dir / "precomputed_sequences"  # Directory, not .npz
    norm_stats_path = collector.data_dir / "normalization_stats.npz"

    dataset.precompute_and_save(
        output_path=str(precomputed_path),
        norm_stats_path=str(norm_stats_path)
    )

    print(f"\n✓ Precomputed sequences saved to: {precomputed_path}")
    print(f"✓ Normalization stats saved to: {norm_stats_path}")
    print(f"\nYou can now use loadDataset() for fast loading!")

    # Step 7b (optional): Precompute augmented dataset
    if precompute_augmented:
        print("\n" + "=" * 60)
        print("STEP 7b: Precomputing augmented dataset")
        print("=" * 60)

        # Determine n_nearby_available for augmentation
        if precompute_augmented is True:
            n_nearby_available = n_nearby
            print(f"  Using n_nearby_available={n_nearby_available} (same as n_nearby, permutation-only augmentation)")
        else:
            n_nearby_available = int(precompute_augmented)
            if n_nearby_available < n_nearby:
                raise ValueError(
                    f"precompute_augmented={n_nearby_available} must be >= n_nearby={n_nearby}"
                )
            print(f"  Using n_nearby_available={n_nearby_available} (skip patterns + permutations)")

        # Import and run augmentation precompute
        from precompute_augmented import generate_all_augmentations_sequential

        generate_all_augmentations_sequential(
            data_dir=str(collector.data_dir),
            n_nearby_available=n_nearby_available,
            n_nearby_in_features=n_nearby,
            coverage_threshold=coverage_threshold,
            seq_length=seq_length,
        )

    # Step 8: Train/val/test split
    print("\n" + "=" * 60)
    print("STEP 8: Creating train/val/test splits")
    print("=" * 60)

    train_ds, val_ds, test_ds = SoilMoistureSequenceDataset.train_val_test_split(
        dataset,
        val_stations_ratio=0.15,
        test_stations_ratio=0.0
    )
    return train_ds, val_ds, test_ds

def loadDataset(use_precomputed=True, normalize=True, precomputed_path=None, norm_stats_path=None,
                coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
                n_nearby: int = 4, seq_length: int = 64,
                live_augment: Union[bool, int] = False):
    """
    Load PyTorch Dataset

    Args:
        use_precomputed: If True, load from precomputed file (much faster)
        normalize: If True, normalize data to [-1, 1] range
        precomputed_path: Path to precomputed sequences (optional)
        norm_stats_path: Path to normalization stats (optional)
        coverage_threshold: Minimum data coverage for a parameter to be included
        n_nearby: Number of nearby stations in features (default: 4)
        seq_length: Sequence length (default: 64)
        live_augment: Whether to use live augmentation:
            - False (default): Load regular dataset
            - True: Use live augmentation with n_nearby as n_nearby_available (permutations only)
            - int: Use live augmentation with this value as n_nearby_available (must be >= n_nearby)

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    from SoilMoistureSequenceDataset import SoilMoistureSequenceDataset
    collector = MeteoGaliciaCollector()  # Does nothing, just for the paths

    # Get filtered parameters
    _, filtered_params = collector.analyze_parameter_coverage(coverage_threshold=coverage_threshold)

    if not filtered_params:
        print("\n✗ No parameters passed the threshold!")
        return None, None, None

    print(f"\nUsing {len(filtered_params)} filtered parameters...")

    # Determine norm_stats_path
    if norm_stats_path is None:
        norm_stats_path = collector.data_dir / "normalization_stats.npz"

    # Handle live augmentation case
    if live_augment:
        print("\n" + "=" * 60)
        print("Loading Dataset with LIVE AUGMENTATION")
        print("=" * 60)

        # Lazy import to avoid circular dependency
        from augmented_live import AugmentedLiveDataset

        # Determine n_nearby_available
        if live_augment is True:
            n_nearby_available = n_nearby
            print(f"  n_nearby_available={n_nearby_available} (same as n_nearby, permutation-only)")
        else:
            n_nearby_available = int(live_augment)
            if n_nearby_available < n_nearby:
                raise ValueError(
                    f"live_augment={n_nearby_available} must be >= n_nearby={n_nearby}"
                )
            print(f"  n_nearby_available={n_nearby_available} (skip patterns + permutations)")

        dataset = AugmentedLiveDataset.from_base_dataset(
            timeseries=str(collector.timeseries_file),
            stations=str(collector.stations_file),
            nearest=str(collector.nearest_file),
            dense_array_path=str(collector.data_dir / "dense_features.npz"),
            feature_params=filtered_params,
            seq_length=seq_length,
            n_nearby_available=n_nearby_available,
            n_nearby_in_features=n_nearby,
            normalize=normalize,
            norm_stats_path=str(norm_stats_path) if normalize else None,
        )

        # Train/val/test split
        print("\n" + "=" * 60)
        print("Creating train/val/test splits")
        print("=" * 60)

        train_ds, val_ds, test_ds = AugmentedLiveDataset.train_val_test_split(
            dataset,
            val_stations_ratio=0.15,
            test_stations_ratio=0.0
        )
        return train_ds, val_ds, test_ds

    # Regular (non-augmented) dataset loading
    print("\n" + "=" * 60)
    print("Loading Dataset (no augmentation)")
    print("=" * 60)

    # Check for precomputed data
    if precomputed_path is None:
        precomputed_path = collector.data_dir / "precomputed_sequences"

    if use_precomputed and not Path(precomputed_path).exists():
        print(f"\n⚠ Precomputed data not found at {precomputed_path}")
        print("  Run buildDataset() first for much faster loading!")
        print("  Falling back to on-the-fly sequence building (SLOW)...")
        use_precomputed = False

    dataset = SoilMoistureSequenceDataset(
        timeseries=str(collector.timeseries_file),
        stations=str(collector.stations_file),
        nearest=str(collector.nearest_file),
        seq_length=seq_length,
        n_nearest=n_nearby,
        feature_params=filtered_params,
        precomputed_path=str(precomputed_path) if use_precomputed else None,
        normalize=normalize,
        norm_stats_path=str(norm_stats_path) if normalize else None
    )

    print(f"\nFeature names: {dataset.get_feature_names()}")

    # Example sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nExample sample:")
        print(f"  Features shape: {sample['features'].shape}")
        print(f"  Target: {sample['target']}")
        print(f"  Mask shape: {sample['mask'].shape}")
        print(f"  Station ID: {sample['target_station_id']}")

    # Train/val/test split
    print("\n" + "=" * 60)
    print("Creating train/val/test splits")
    print("=" * 60)

    train_ds, val_ds, test_ds = SoilMoistureSequenceDataset.train_val_test_split(
        dataset,
        val_stations_ratio=0.15,
        test_stations_ratio=0.0
    )
    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='MeteoGalicia Soil Moisture Dataset Builder and Trainer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Moisturizer.py --build                              # Build base dataset (4 nearby)
  python Moisturizer.py --build --precompute-augmented 5     # Build + precompute 120x augmented
  python Moisturizer.py --train                              # Train with no augmentation
  python Moisturizer.py --train --live-augment 5             # Train with 120x live augmentation
  python Moisturizer.py --build --n-nearby 4 --seq-length 64 # Explicit defaults
        """
    )
    parser.add_argument('--build', action='store_true',
                       help='Build/rebuild the dataset')
    parser.add_argument('--train', action='store_true',
                       help='Train the model')
    parser.add_argument('--coverage', type=float, default=DEFAULT_COVERAGE_THRESHOLD,
                       help=f'Minimum data coverage threshold for parameters (default: {DEFAULT_COVERAGE_THRESHOLD})')
    parser.add_argument('--n-nearby', type=int, default=4,
                       help='Number of nearby stations in features (default: 4)')
    parser.add_argument('--seq-length', type=int, default=64,
                       help='Sequence length in days (default: 64)')
    parser.add_argument('--precompute-augmented', type=int, default=0, metavar='N',
                       help='Precompute augmented dataset with N nearby available (0=disabled, use with --build)')
    parser.add_argument('--live-augment', type=int, default=5, metavar='N',
                       help='Use live augmentation with N nearby available (0=disabled, use with --train)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Training batch size (default: 512)')
    parser.add_argument('--lr', type=float, default=4.1e-4,
                       help='Initial learning rate (default: 4.1e-4)')

    args = parser.parse_args()

    # If no action specified, show help
    if not args.build and not args.train:
        parser.print_help()
        print("\nNo action specified. Use --build to build dataset or --train to train model.")
        exit(0)

    if args.build:
        print("\n" + "=" * 60)
        print("BUILDING DATASET")
        print("=" * 60)
        print(f"Coverage threshold: {args.coverage}")
        print(f"n_nearby: {args.n_nearby}")
        print(f"seq_length: {args.seq_length}")
        if args.precompute_augmented > 0:
            print(f"precompute_augmented: {args.precompute_augmented}")

        buildDataset(
            coverage_threshold=args.coverage,
            n_nearby=args.n_nearby,
            seq_length=args.seq_length,
            precompute_augmented=args.precompute_augmented if args.precompute_augmented > 0 else False,
        )

    if args.train:
        # Determine live_augment value
        live_augment_val = args.live_augment if args.live_augment > 0 else False

        train_ds, val_ds, _ = loadDataset(
            coverage_threshold=args.coverage,
            n_nearby=args.n_nearby,
            seq_length=args.seq_length,
            live_augment=live_augment_val,
        )

        # Data leakage check
        sample = train_ds[0]
        features = sample['features']
        target = sample['target']

        print(f"\n=== Data Leakage Check ===")
        print(f"Feature shape: {features.shape}")
        print(f"Target value: {target.item():.4f}")

        features_np = features.numpy()
        matches = (np.abs(features_np - target.item())[:, :26] < 0.001).sum()
        print(f"Features matching target value: {matches}")

        if matches > 0:
            print("⚠️  LEAKAGE DETECTED: Target value found in features!")
        else:
            print("✓ No obvious leakage detected")

        last_timestep = features_np[-1, :]
        last_matches = (np.abs(last_timestep - target.item()) < 0.001).sum()
        print(f"Last timestep matches: {last_matches}")

        # Train
        model = load_model()
        model.training_loop(
            train_data=train_ds,
            val_data=val_ds,
            lr=args.lr,
            lr_mid=args.lr * 0.975,  # Slightly lower mid-training LR
            lr_min=3e-5,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            transfer=0
        )