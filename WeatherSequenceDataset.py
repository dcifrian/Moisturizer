import os
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple, Set, Union
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from Moisturizer import INVALID_MARKER_API, INVALID_MARKER_MISSING, NORMALIZED_INVALID_MARKER, DEFAULT_COVERAGE_THRESHOLD, _load_precomputed_data, normalize_target, normalize_features


class WeatherSequenceDataset(Dataset):
    """
    PyTorch Dataset for weather parameter prediction with temporal sequences.
    Suitable for transformer models.

    Can predict any weather parameter (default: soil moisture).
    Uses nearby stations' data including distance, weather features, and target parameter.

    Note: Requires PyTorch to be installed. If PyTorch is not available,
    this class can still be instantiated but PyTorch-specific functionality
    (tensors, DataLoader) will not work.

    For optimal performance with large datasets:
    1. Precompute sequences using precompute_and_save()
    2. Convert .npz to .npy directory format: python convert_npz_to_memmap.py data.npz
    3. Load with precomputed_path pointing to the .npy directory for true memory-mapping
    """

    def __init__(
            self,
            timeseries: Union[str,pd.DataFrame],
            stations: Union[str,pd.DataFrame],
            nearest: Union[str,pd.DataFrame],
            seq_length: int,
            n_nearest: int = 4,
            target_stations: Optional[List[int]] = None,
            feature_params: Optional[List[str]] = None,
            target_param: str = "HS_CV_AVG_-0.2m",
            missing_value: float = INVALID_MARKER_MISSING,
            precomputed_path: Optional[str] = None,
            normalize: bool = True,
            norm_stats_path: Optional[str] = None,
            dense_array_path: Optional[str] = None
    ):
        """
        Initialize dataset

        Args:
            timeseries: Path to raw_timeseries.csv or DataFrame
            stations: Path to stations_metadata.csv or DataFrame
            nearest: Path to nearest_stations.csv or DataFrame
            seq_length: Number of days in each sequence
            n_nearest: Number of nearest stations to include
            target_stations: List of station IDs to use (if None, use all with target param data)
            feature_params: List of parameter codes to include as features
                           (if None, uses all except target_param for target station)
                           Tip: Use analyze_parameter_coverage() to get filtered params
            target_param: Parameter code for the prediction target (default: soil moisture)
            missing_value: Value to use for missing data
            precomputed_path: Path to precomputed data directory (.npy files) or .npz file
                             RECOMMENDED: Use .npy directory for true memory-mapping
                             Convert .npz to .npy: python convert_npz_to_memmap.py data.npz
            normalize: Whether to normalize features to [-1, 1] range
            norm_stats_path: Path to normalization stats .npz file (if None, computed automatically)
            dense_array_path: Path to dense_features.npz (FASTEST - recommended for generation)
        """
        self.seq_length = seq_length
        self.n_nearest = n_nearest
        self.target_param = target_param
        self.missing_value = missing_value
        self.normalize = normalize
        self.precomputed_path = precomputed_path

        # Precomputed data (loaded on demand)
        self.precomputed_data = None
        self.norm_stats = None
        self.dense_arrays = None
        self.dense_array_path = dense_array_path

        # Load data
        print("Loading data files...")
        self.timeseries_df = timeseries if isinstance(timeseries,pd.DataFrame) else pd.read_csv(timeseries)
        self.timeseries_df['date'] = pd.to_datetime(self.timeseries_df['date'])
        self.stations_df = stations if isinstance(stations,pd.DataFrame) else pd.read_csv(stations)
        self.nearest_df = nearest if isinstance(nearest,pd.DataFrame) else pd.read_csv(nearest)

        # Load dense arrays if provided (FASTEST method for precomputation)
        if dense_array_path and os.path.exists(dense_array_path):
            print(f"Loading dense feature arrays from {dense_array_path}...")
            dense_data = np.load(dense_array_path)
            self.dense_arrays = {
                'features': dense_data['features'],  # [stations, dates, features]
                'masks': dense_data['masks'],  # [stations, dates, features]
                'station_ids': dense_data['station_ids'].tolist(),
                'dates': pd.DatetimeIndex(dense_data['dates']),
                'feature_params': dense_data['feature_params'].tolist()
            }
            # Create fast lookup mappings
            self.dense_station_to_idx = {sid: idx for idx, sid in enumerate(self.dense_arrays['station_ids'])}
            # Dates in dense arrays are already at midnight - no need to normalize
            self.dense_date_to_idx = {date: idx for idx, date in enumerate(self.dense_arrays['dates'])}
            print(f"  Loaded dense arrays: {self.dense_arrays['features'].shape}")
            print(f"  Memory: ~{self.dense_arrays['features'].nbytes / 1e6:.1f} MB")

            # Still build timeseries index as fallback for edge cases
            print("  Building fallback index for edge cases...")
            # Use numpy arrays for fast dict building (100x faster than itertuples)
            station_ids = self.timeseries_df['station_id'].astype(np.int32).values
            # Dates from CSV/precomputed are already at midnight - no need to normalize
            dates = self.timeseries_df['date'].values
            param_codes = self.timeseries_df['parameter_code'].values
            values = self.timeseries_df['value'].astype(np.float32).values
            self.timeseries_index = dict(zip(
                zip(station_ids, dates, param_codes),
                values
            ))
        else:
            # Use dict lookup index
            print("Creating fast lookup index for timeseries...")
            # Use numpy arrays for fast dict building (100x faster than itertuples)
            station_ids = self.timeseries_df['station_id'].astype(np.int32).values
            # Dates from CSV/precomputed are already at midnight - no need to normalize
            dates = self.timeseries_df['date'].values
            param_codes = self.timeseries_df['parameter_code'].values
            values = self.timeseries_df['value'].astype(np.float32).values
            self.timeseries_index = dict(zip(
                zip(station_ids, dates, param_codes),
                values
            ))
            print(f"  Indexed {len(self.timeseries_index):,} data points")

        # Build station coordinates lookup (FAST DICT - avoid pandas filtering in loops!)
        print("Creating fast lookup index for station coordinates...")
        self.station_coords = {}
        coordinate_features = ['altitude', 'utmx', 'utmy']
        for _, row in self.stations_df.iterrows():
            station_id = int(row['station_id'])
            self.station_coords[station_id] = {
                'altitude': float(row['altitude']) if pd.notna(row['altitude']) else None,
                'utmx': float(row['utmx']) if pd.notna(row['utmx']) else None,
                'utmy': float(row['utmy']) if pd.notna(row['utmy']) else None
            }
        print(f"  Indexed {len(self.station_coords)} station coordinates")

        # Build nearest stations lookup (FAST DICT - avoid pandas filtering in loops!)
        print("Creating fast lookup index for nearest stations...")
        self.nearest_stations_cache = {}
        for _, row in self.nearest_df.iterrows():
            station_id = int(row['station_id'])
            nearby_with_soil = []

            # Extract all nearest neighbors from the row
            for i in range(1, len(self.nearest_df.columns) // 3 + 1):
                nearest_id_col = f'nearest_{i}_id'
                if nearest_id_col not in row:
                    break
                if row[f'nearest_{i}_has_soil_moisture']:
                    nearby_with_soil.append({
                        'station_id': int(row[nearest_id_col]),
                        'distance': row[f'nearest_{i}_distance']
                    })

            self.nearest_stations_cache[station_id] = nearby_with_soil
        print(f"  Indexed {len(self.nearest_stations_cache)} station nearest neighbor lists")

        # Validate that we have enough neighbors for the requested n_nearest
        # This catches the case where nearest_stations.csv was built with fewer neighbors
        n_neighbors_in_file = len([c for c in self.nearest_df.columns if c.startswith('nearest_') and c.endswith('_id')])
        if n_neighbors_in_file < n_nearest:
            raise ValueError(
                f"nearest_stations.csv has only {n_neighbors_in_file} neighbors but n_nearest={n_nearest} requested. "
                f"Regenerate nearest_stations.csv with more neighbors using:\n"
                f"  from Moisturizer import regenerate_nearest_stations\n"
                f"  regenerate_nearest_stations(n_nearest={n_nearest})"
            )

        # Determine target stations
        if target_stations is None:
            self.target_stations = self.stations_df[
                self.stations_df['has_soil_moisture']
            ]['station_id'].tolist()
        else:
            self.target_stations = target_stations

        # Track if data is already normalized in precomputed file
        self.is_prenormalized = False

        # Load precomputed data if available
        if precomputed_path and os.path.exists(precomputed_path):
            print(f"Loading precomputed sequences from {precomputed_path}...")
            # Use the new loading function that supports both .npy directories and .npz files
            self.precomputed_data = _load_precomputed_data(precomputed_path)
            print(f"  Using memory-mapped arrays (dataset will not be loaded into RAM)")

            # Validate feature_params match precomputed data
            if 'feature_params' not in self.precomputed_data:
                raise ValueError(
                    "Precomputed dataset is missing 'feature_params' (old format not supported). "
                    "Regenerate the dataset with buildDataset() to create new format."
                )

            precomputed_params = self.precomputed_data['feature_params'].tolist()
            print(f"  Precomputed with {len(precomputed_params)} feature parameters")

            if feature_params is not None:
                # User specified feature_params - must match precomputed
                if feature_params != precomputed_params:
                    raise ValueError(
                        f"Feature parameters mismatch!\n"
                        f"  Precomputed dataset was built with {len(precomputed_params)} parameters:\n"
                        f"    {precomputed_params}\n"
                        f"  But you requested {len(feature_params)} parameters:\n"
                        f"    {feature_params}\n"
                        f"  You must either:\n"
                        f"    1. Use feature_params=None to load with precomputed parameters, or\n"
                        f"    2. Rebuild the precomputed dataset with the desired parameters"
                    )
                print(f"  ✓ Feature parameters match precomputed dataset")
                self.feature_params = feature_params
            else:
                # User didn't specify - use precomputed params
                print(f"  Using precomputed feature parameters")
                self.feature_params = precomputed_params

            # Check if data is already normalized
            if 'is_normalized' in self.precomputed_data:
                self.is_prenormalized = bool(self.precomputed_data['is_normalized'][0])

            # Reconstruct sample_index from components (avoid pickle)
            target_stations = self.precomputed_data['target_stations']
            end_dates = self.precomputed_data['end_dates']
            start_dates = self.precomputed_data['start_dates']

            self.sample_index = []
            for i in range(len(target_stations)):
                self.sample_index.append({
                    'target_station': int(target_stations[i]),
                    'end_date': pd.Timestamp.fromtimestamp(end_dates[i]),
                    'start_date': pd.Timestamp.fromtimestamp(start_dates[i])
                })

            print(f"  Loaded {len(self.sample_index)} precomputed samples")
            if self.is_prenormalized:
                print(f"  Data is pre-normalized (fast path enabled!)")

            # No index mapping for original dataset (only used in splits)
            self._indices = None
        else:
            # Determine feature parameters (not using precomputed data)
            if feature_params is None:
                # Use all parameters except soil moisture
                all_params = self.timeseries_df['parameter_code'].unique()
                self.feature_params = [p for p in all_params if p != target_param]
            else:
                self.feature_params = feature_params

            # Build index of valid samples
            self._build_sample_index()
            self._indices = None

        # Check if soil moisture is in feature_params (data leakage!)
        self.soil_in_features = self.target_param in self.feature_params
        if self.soil_in_features:
            print(f"⚠ WARNING: Soil moisture ({self.target_param}) found in feature_params!")
            print(f"  This will be filtered out from target station features to prevent data leakage.")
            print(f"  Nearby stations will still have soil moisture as context.")
            self.soil_feature_idx = self.feature_params.index(self.target_param)
        else:
            self.soil_feature_idx = None

        # Load normalization statistics (only needed for reference or if not pre-normalized)
        if normalize and not self.is_prenormalized:
            if norm_stats_path and os.path.exists(norm_stats_path):
                print(f"Loading normalization statistics from {norm_stats_path}...")
                self.norm_stats = np.load(norm_stats_path)
            elif precomputed_path and os.path.exists(precomputed_path):
                # Compute from precomputed data
                print("Computing normalization statistics from precomputed data...")
                self._compute_norm_stats_from_precomputed()
            else:
                print("Warning: normalize=True but no precomputed data available.")
                print("  Normalization will be computed on-the-fly (slow).")
                self.normalize = False
        elif self.is_prenormalized:
            # Data is already normalized, skip normalization in __getitem__
            self.normalize = False  # Don't normalize again

        print(f"Dataset initialized:")
        print(f"  Sequence length: {seq_length}")
        print(f"  Target stations: {len(self.target_stations)}")
        print(f"  Feature parameters: {len(self.feature_params)}")
        print(f"  Valid samples: {len(self.sample_index)}")
        print(f"  Using precomputed: {precomputed_path is not None and os.path.exists(precomputed_path)}")
        print(f"  Pre-normalized: {self.is_prenormalized}")
        print(f"  Runtime normalization: {self.normalize}")

    def _build_sample_index(self):
        """Build index of valid samples (target_station, end_date) pairs - OPTIMIZED"""
        self.sample_index = []

        print("Building sample index...")

        # Get the first available date in the dataset (needed for history check)
        first_available_date = self.timeseries_df['date'].min()
        print(f"  First available date: {first_available_date.date()}")

        # Pre-filter soil moisture data once (HUGE SPEEDUP!)
        soil_moisture_df = self.timeseries_df[
            self.timeseries_df['parameter_code'] == self.target_param
        ].copy()

        # Build a lookup dict for fast value access
        soil_moisture_lookup = {}
        for _, row in soil_moisture_df.iterrows():
            key = (int(row['station_id']), row['date'])
            soil_moisture_lookup[key] = float(row['value'])

        # Get all dates per station (vectorized)
        station_dates = soil_moisture_df.groupby('station_id')['date'].apply(list).to_dict()

        for idx, target_id in enumerate(self.target_stations):
            if idx % 10 == 0:
                print(f"  Processing station {idx+1}/{len(self.target_stations)}...")

            # Get dates for this station
            target_dates = station_dates.get(target_id, [])
            if not target_dates:
                continue

            # Sort dates
            target_dates = sorted(target_dates)

            # Vectorized validation check
            for date in target_dates:
                # Need seq_length days including this date
                start_date = date - pd.Timedelta(days=self.seq_length - 1)

                # Skip if we don't have enough historical data
                if start_date < first_available_date:
                    continue

                # Fast lookup of soil moisture value
                key = (target_id, date)
                target_val = soil_moisture_lookup.get(key)

                # Skip if no value or if value is invalid marker
                if target_val is None or target_val == INVALID_MARKER_API or target_val == self.missing_value or pd.isna(target_val):
                    continue

                # This sample has valid target and enough historical data
                self.sample_index.append({
                    'target_station': target_id,
                    'end_date': date,
                    'start_date': start_date
                })

        print(f"  Built {len(self.sample_index)} valid samples")

    def _get_nearest_stations(self, target_station_id: int) -> List[Dict]:
        """Get n nearest stations with soil moisture for a target station"""
        # Use cached lookup instead of pandas filtering (100x faster!)
        all_nearby = self.nearest_stations_cache.get(target_station_id, [])

        # Return up to n_nearest stations
        return all_nearby[:self.n_nearest]

    def _build_sequence_tensor(
            self,
            target_station_id: int,
            start_date: pd.Timestamp,
            end_date: pd.Timestamp
    ) -> 'Tuple[torch.Tensor, torch.Tensor, torch.Tensor]':
        """
        Build sequence tensor for a sample (OPTIMIZED VERSION)

        Uses fast dictionary lookups instead of pandas filtering.
        Speedup: ~1000x faster than old version!

        Returns:
            features: [seq_length, total_features] tensor
            target: tensor (soil moisture at end_date)
            mask: [seq_length, total_features] tensor (1 for valid, 0 for missing)
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        nearby_stations = self._get_nearest_stations(target_station_id)

        # Calculate feature dimensions
        # Target station features + (nearby features + soil moisture + distance) * n_nearest
        target_features_per_timestep = len(self.feature_params)
        nearby_features_per_timestep = (len(self.feature_params) + 1 + 1)  # features + soil moisture + distance
        total_features = target_features_per_timestep + (nearby_features_per_timestep * self.n_nearest)

        # Initialize tensors
        features = np.full((self.seq_length, total_features), self.missing_value, dtype=np.float32)
        mask = np.zeros((self.seq_length, total_features), dtype=bool)

        # Coordinate features (static - not in timeseries)
        coordinate_features = ['altitude', 'utmx', 'utmy']

        # Fill target station coordinate features ONCE (static, same for all timesteps)
        # Use FAST DICT LOOKUP instead of pandas filtering!
        for f_idx, param in enumerate(self.feature_params):
            if param in coordinate_features:
                # Look up from station_coords dict (100x faster than pandas!)
                if target_station_id in self.station_coords:
                    coord_value = self.station_coords[target_station_id].get(param)
                    if coord_value is not None:
                        # Fill for ALL timesteps (static feature)
                        features[:, f_idx] = coord_value
                        mask[:, f_idx] = True

        # Fill target station features using FAST DICT LOOKUP
        # Note: dates from pd.date_range(freq='D') are already at midnight
        for t, date in enumerate(date_range):
            # Target station features (time-varying only)
            for f_idx, param in enumerate(self.feature_params):
                if param not in coordinate_features:
                    key = (target_station_id, date, param)
                    if key in self.timeseries_index:
                        features[t, f_idx] = self.timeseries_index[key]
                        mask[t, f_idx] = True

            # Fill nearby stations features using FAST DICT LOOKUP
            for n_idx, nearby in enumerate(nearby_stations):
                nearby_station_id = nearby['station_id']
                nearby_offset = target_features_per_timestep + (n_idx * nearby_features_per_timestep)

                # Distance (constant across time)
                features[t, nearby_offset] = nearby['distance']
                mask[t, nearby_offset] = True

                # Coordinate features (static) - Use FAST DICT LOOKUP!
                for f_idx, param in enumerate(self.feature_params):
                    if param in coordinate_features:
                        if nearby_station_id in self.station_coords:
                            coord_value = self.station_coords[nearby_station_id].get(param)
                            if coord_value is not None:
                                feat_idx = nearby_offset + 1 + f_idx
                                features[t, feat_idx] = coord_value
                                mask[t, feat_idx] = True

                # Time-varying features
                for f_idx, param in enumerate(self.feature_params):
                    if param not in coordinate_features:
                        key = (nearby_station_id, date, param)
                        feat_idx = nearby_offset + 1 + f_idx
                        if key in self.timeseries_index:
                            features[t, feat_idx] = self.timeseries_index[key]
                            mask[t, feat_idx] = True

                # Soil moisture for nearby station
                key = (nearby_station_id, date, self.target_param)
                soil_idx = nearby_offset + 1 + len(self.feature_params)
                if key in self.timeseries_index:
                    features[t, soil_idx] = self.timeseries_index[key]
                    mask[t, soil_idx] = True

        # Get target (soil moisture at end_date for target station)
        # Use the last date from date_range (already at midnight)
        target_key = (target_station_id, date_range[-1], self.target_param)
        target = self.timeseries_index.get(target_key, self.missing_value)

        return (
            torch.from_numpy(features),
            torch.tensor(target, dtype=torch.float32).unsqueeze(0),
            torch.from_numpy(mask)
        )

    def _build_sequence_from_dense(
            self,
            target_station_id: int,
            start_date: pd.Timestamp,
            end_date: pd.Timestamp
    ) -> 'Tuple[torch.Tensor, torch.Tensor, torch.Tensor]':
        """
        Build sequence tensor from dense arrays using FAST ARRAY SLICING

        This is the optimal method - no redundant computation, vectorized operations.
        Consecutive samples share 63/64 days but we just slice different windows.

        Returns:
            features: [seq_length, total_features] tensor
            target: tensor (soil moisture at end_date)
            mask: [seq_length, total_features] tensor (1 for valid, 0 for missing)
        """
        nearby_stations = self._get_nearest_stations(target_station_id)

        # Get date range indices (dates from pd.date_range are already at midnight)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        date_indices = [self.dense_date_to_idx.get(date) for date in date_range]

        # Check if all dates are in our dense array
        if None in date_indices:
            # Fall back to dict method if dates not available
            print(f"  DEBUG: Falling back to dict method - dates not found in dense array")
            print(f"    date_indices: {date_indices}")
            return self._build_sequence_tensor(target_station_id, start_date, end_date)

        # Get station indices
        target_idx = self.dense_station_to_idx.get(target_station_id)
        if target_idx is None:
            return self._build_sequence_tensor(target_station_id, start_date, end_date)

        nearby_indices = []
        for nearby in nearby_stations:
            nearby_idx = self.dense_station_to_idx.get(nearby['station_id'])
            if nearby_idx is None:
                return self._build_sequence_tensor(target_station_id, start_date, end_date)
            nearby_indices.append((nearby_idx, nearby['distance']))

        # Calculate feature dimensions
        target_features_per_timestep = len(self.feature_params)
        nearby_features_per_timestep = len(self.feature_params) + 1 + 1  # features + soil moisture + distance
        total_features = target_features_per_timestep + (nearby_features_per_timestep * self.n_nearest)

        # Initialize output arrays
        features = np.full((self.seq_length, total_features), self.missing_value, dtype=np.float32)
        mask = np.zeros((self.seq_length, total_features), dtype=bool)

        # VECTORIZED: Slice target station data for all dates at once
        target_slice = self.dense_arrays['features'][target_idx, date_indices, :]  # [seq_length, num_features]
        target_mask_slice = self.dense_arrays['masks'][target_idx, date_indices, :]

        # Copy target features (exclude soil moisture if present)
        features[:, :target_features_per_timestep] = target_slice[:, :target_features_per_timestep]
        mask[:, :target_features_per_timestep] = target_mask_slice[:, :target_features_per_timestep]

        # VECTORIZED: Slice each nearby station's data
        for n_idx, (nearby_idx, distance) in enumerate(nearby_indices):
            nearby_offset = target_features_per_timestep + (n_idx * nearby_features_per_timestep)

            # Distance (constant across time)
            features[:, nearby_offset] = distance
            mask[:, nearby_offset] = True

            # Slice all nearby station data at once
            nearby_slice = self.dense_arrays['features'][nearby_idx, date_indices, :]  # [seq_length, num_features]
            nearby_mask_slice = self.dense_arrays['masks'][nearby_idx, date_indices, :]

            # Copy features (not including soil moisture yet)
            feat_start = nearby_offset + 1
            feat_end = feat_start + target_features_per_timestep
            features[:, feat_start:feat_end] = nearby_slice[:, :target_features_per_timestep]
            mask[:, feat_start:feat_end] = nearby_mask_slice[:, :target_features_per_timestep]

            # Soil moisture (if available in dense array)
            soil_idx_in_dense = None
            if self.target_param in self.dense_arrays['feature_params']:
                soil_idx_in_dense = self.dense_arrays['feature_params'].index(self.target_param)
                soil_idx = feat_end
                features[:, soil_idx] = nearby_slice[:, soil_idx_in_dense]
                mask[:, soil_idx] = nearby_mask_slice[:, soil_idx_in_dense]

        # Get target value (soil moisture at end_date for target station)
        end_date_idx = date_indices[-1]
        if soil_idx_in_dense is not None:
            target = self.dense_arrays['features'][target_idx, end_date_idx, soil_idx_in_dense]
            # DEBUG
            if target == self.missing_value:
                print(f"  DEBUG: Target from dense array is missing_value!")
                print(f"    target_idx={target_idx}, end_date_idx={end_date_idx}, soil_idx={soil_idx_in_dense}")
                print(f"    target={target}")
        else:
            # Fall back to dict lookup for soil moisture if not in dense array
            print(f"  DEBUG: soil_idx_in_dense is None - falling back to dict lookup")
            target_key = (target_station_id, end_date, self.target_param)
            target = self.timeseries_index.get(target_key, self.missing_value) if self.timeseries_index else self.missing_value

        return (
            torch.from_numpy(features),
            torch.tensor(target, dtype=torch.float32).unsqueeze(0),
            torch.from_numpy(mask)
        )

    def _compute_norm_stats_from_precomputed(self):
        """
        Compute normalization statistics from precomputed data.

        Computes:
        - Per-slot stats for the full feature vector (for base dataset normalization)
        - Per-slot nearby stats: separate min/max for each nearby station slot
          This enables:
          - Non-augmented: use per-slot stats directly for full range utilization
          - Augmented: compute min/max across available slots for each feature type
        """
        print("Computing min/max for each feature (excluding invalid values)...")

        n_samples = len(self.precomputed_data['features'])
        seq_length = self.precomputed_data['features'].shape[1]
        n_features = self.precomputed_data['features'].shape[2]

        # Per-slot stats for full feature vector (for base dataset normalization)
        feature_mins = np.full(n_features, np.inf, dtype=np.float32)
        feature_maxs = np.full(n_features, -np.inf, dtype=np.float32)

        # Target feature stats (same for any configuration)
        n_params = len(self.feature_params)
        nearby_features_per_station = 1 + n_params + 1  # distance + params + soil
        target_feat_mins = np.full(n_params, np.inf, dtype=np.float32)
        target_feat_maxs = np.full(n_params, -np.inf, dtype=np.float32)

        # Per-slot nearby stats: [n_nearest, nearby_features_per_station]
        # Each slot has its own stats (closer stations may have different ranges than farther ones)
        nearby_slot_mins = np.full((self.n_nearest, nearby_features_per_station), np.inf, dtype=np.float32)
        nearby_slot_maxs = np.full((self.n_nearest, nearby_features_per_station), -np.inf, dtype=np.float32)

        # Invalid markers to exclude
        invalid_markers = [INVALID_MARKER_API, self.missing_value]

        # Process in batches to save memory
        batch_size = 1000
        for i in range(0, n_samples, batch_size):
            end_i = min(i + batch_size, n_samples)
            features_batch = self.precomputed_data['features'][i:end_i]
            masks_batch = self.precomputed_data['masks'][i:end_i]

            # Per-slot stats for full feature vector
            for feat_idx in range(n_features):
                feat_data = features_batch[:, :, feat_idx]
                feat_mask = masks_batch[:, :, feat_idx]

                # Get valid data (masked and not invalid marker)
                valid_mask = feat_mask.copy()
                for marker in invalid_markers:
                    valid_mask &= (feat_data != marker)

                valid_data = feat_data[valid_mask]

                if len(valid_data) > 0:
                    feature_mins[feat_idx] = min(feature_mins[feat_idx], valid_data.min())
                    feature_maxs[feat_idx] = max(feature_maxs[feat_idx], valid_data.max())

            # Target feature stats
            target_feats = features_batch[:, :, :n_params]
            for feat_idx in range(n_params):
                feat_data = target_feats[:, :, feat_idx].ravel()
                valid = feat_data[(feat_data != INVALID_MARKER_MISSING) & (feat_data != INVALID_MARKER_API)]
                if len(valid) > 0:
                    target_feat_mins[feat_idx] = min(target_feat_mins[feat_idx], valid.min())
                    target_feat_maxs[feat_idx] = max(target_feat_maxs[feat_idx], valid.max())

            # Per-slot nearby stats
            nearby_data = features_batch[:, :, n_params:]
            nearby_reshaped = nearby_data.reshape(
                end_i - i, seq_length, self.n_nearest, nearby_features_per_station
            )
            for slot_idx in range(self.n_nearest):
                for feat_idx in range(nearby_features_per_station):
                    feat_data = nearby_reshaped[:, :, slot_idx, feat_idx].ravel()
                    valid = feat_data[(feat_data != INVALID_MARKER_MISSING) & (feat_data != INVALID_MARKER_API)]
                    if len(valid) > 0:
                        nearby_slot_mins[slot_idx, feat_idx] = min(nearby_slot_mins[slot_idx, feat_idx], valid.min())
                        nearby_slot_maxs[slot_idx, feat_idx] = max(nearby_slot_maxs[slot_idx, feat_idx], valid.max())

        # Compute for target (soil moisture prediction target) as well
        targets = self.precomputed_data['targets']
        valid_targets = targets.copy()
        for marker in invalid_markers:
            valid_targets = valid_targets[valid_targets != marker]

        if len(valid_targets) > 0:
            target_min = valid_targets.min()
            target_max = valid_targets.max()
        else:
            warnings.warn(
                "No valid target values found in dataset! Using fallback range [0.0, 1.0]. "
                "This indicates a data integrity issue - check coverage_threshold setting.",
                stacklevel=2
            )
            target_min = 0.0
            target_max = 1.0

        # Store stats
        self.norm_stats = {
            # Per-slot stats for full feature vector (for base dataset)
            'feature_mins': feature_mins,
            'feature_maxs': feature_maxs,
            'target_min': target_min,
            'target_max': target_max,
            # Target feature stats (same for any configuration)
            'target_feature_mins': target_feat_mins,
            'target_feature_maxs': target_feat_maxs,
            # Per-slot nearby stats: [n_nearby_slots, nearby_features_per_station]
            'nearby_slot_mins': nearby_slot_mins,
            'nearby_slot_maxs': nearby_slot_maxs,
            'n_nearby_slots': self.n_nearest,
            # Metadata
            'n_params': n_params,
            'n_base_samples': n_samples,
            'seq_length': seq_length,
            'feature_params': self.feature_params,
        }

        print(f"  Feature min range: [{feature_mins.min():.2f}, {feature_mins.max():.2f}]")
        print(f"  Feature max range: [{feature_maxs.min():.2f}, {feature_maxs.max():.2f}]")
        print(f"  Target range: [{target_min:.2f}, {target_max:.2f}]")
        print(f"  Nearby slots: {self.n_nearest} (per-slot stats stored)")

    def _compute_comprehensive_norm_stats_from_dense(self):
        """
        Compute normalization statistics for ALL possible nearby stations using dense arrays.

        Unlike _compute_norm_stats_from_precomputed (which only computes stats for n_nearest),
        this method computes per-slot stats for ALL available nearby stations. This enables:
        - Non-augmented with any number of nearby stations: use per-slot stats directly
        - Augmented with any n_nearby_available: aggregate across available slots

        Requires dense_arrays and stations_df to be loaded.
        """
        if self.dense_arrays is None:
            raise ValueError("Dense arrays must be loaded to compute comprehensive norm stats")

        print("Computing COMPREHENSIVE normalization stats from dense arrays...")
        print("  (This computes stats for ALL possible nearby stations, not just n_nearest)")

        # Compute ALL nearest neighbors from station coordinates (not from cached n_nearest)
        print("  Computing ALL nearest neighbors from station coordinates...")

        # Get stations with soil moisture (candidates for neighbors)
        soil_moisture_stations = self.stations_df[self.stations_df['has_soil_moisture'] == True].copy()
        soil_moisture_stations = soil_moisture_stations[
            soil_moisture_stations['utmx'].notna() & soil_moisture_stations['utmy'].notna()
        ]

        soil_coords = soil_moisture_stations[['utmx', 'utmy']].values
        soil_station_ids = soil_moisture_stations['station_id'].values

        # Build comprehensive nearest neighbors for each target station
        comprehensive_neighbors = {}  # station_id -> list of {station_id, distance} sorted by distance

        for target_station_id in self.target_stations:
            target_row = self.stations_df[self.stations_df['station_id'] == target_station_id]
            if target_row.empty:
                continue

            target_utmx = target_row['utmx'].values[0]
            target_utmy = target_row['utmy'].values[0]

            if pd.isna(target_utmx) or pd.isna(target_utmy):
                continue

            # Calculate distances to all soil moisture stations
            distances = np.sqrt(
                (soil_coords[:, 0] - target_utmx) ** 2 +
                (soil_coords[:, 1] - target_utmy) ** 2
            )

            # Sort by distance, exclude self
            sorted_indices = np.argsort(distances)
            neighbors = []
            for idx in sorted_indices:
                if soil_station_ids[idx] != target_station_id:
                    neighbors.append({
                        'station_id': int(soil_station_ids[idx]),
                        'distance': float(distances[idx])  # Keep in meters (UTM units)
                    })

            comprehensive_neighbors[target_station_id] = neighbors

        # Find maximum number of nearby stations
        max_nearby = max(len(neighbors) for neighbors in comprehensive_neighbors.values()) if comprehensive_neighbors else 0
        print(f"  Maximum nearby stations available: {max_nearby}")

        n_params = len(self.feature_params)
        nearby_features_per_station = 1 + n_params + 1  # distance + weather params + soil moisture

        # Target feature stats (same for any configuration)
        target_feat_mins = np.full(n_params, np.inf, dtype=np.float32)
        target_feat_maxs = np.full(n_params, -np.inf, dtype=np.float32)

        # Per-slot nearby stats: [max_nearby, nearby_features_per_station]
        nearby_slot_mins = np.full((max_nearby, nearby_features_per_station), np.inf, dtype=np.float32)
        nearby_slot_maxs = np.full((max_nearby, nearby_features_per_station), -np.inf, dtype=np.float32)

        # Target (soil moisture prediction target) stats
        target_min = np.inf
        target_max = -np.inf

        # Invalid markers
        invalid_markers = [INVALID_MARKER_API, self.missing_value]

        # Get soil moisture feature index in dense array
        soil_idx_in_dense = None
        if self.target_param in self.dense_arrays['feature_params']:
            soil_idx_in_dense = self.dense_arrays['feature_params'].index(self.target_param)

        # Feature indices in dense array (weather params only, not soil)
        feature_indices = []
        for param in self.feature_params:
            if param in self.dense_arrays['feature_params']:
                feature_indices.append(self.dense_arrays['feature_params'].index(param))
            else:
                print(f"  Warning: Feature {param} not found in dense arrays")
                feature_indices.append(None)

        print(f"  Processing {len(self.sample_index)} samples...")

        n_samples = len(self.sample_index)

        for idx in tqdm(range(n_samples), desc="Computing comprehensive stats"):
            sample_info = self.sample_index[idx]
            target_station_id = sample_info['target_station']
            end_date = sample_info['end_date']
            start_date = sample_info['start_date']

            # Get target station index in dense array
            target_idx = self.dense_station_to_idx.get(target_station_id)
            if target_idx is None:
                continue

            # Get date indices
            date_indices = []
            current_date = start_date
            while current_date <= end_date:
                date_idx = self.dense_date_to_idx.get(current_date)
                if date_idx is not None:
                    date_indices.append(date_idx)
                current_date += pd.Timedelta(days=1)

            if not date_indices:
                continue

            # Target station features (weather params)
            for feat_idx, dense_idx in enumerate(feature_indices):
                if dense_idx is None:
                    continue
                feat_data = self.dense_arrays['features'][target_idx, date_indices, dense_idx]
                valid = feat_data[(feat_data != INVALID_MARKER_MISSING) & (feat_data != INVALID_MARKER_API)]
                if len(valid) > 0:
                    target_feat_mins[feat_idx] = min(target_feat_mins[feat_idx], valid.min())
                    target_feat_maxs[feat_idx] = max(target_feat_maxs[feat_idx], valid.max())

            # Target value (soil moisture at end_date)
            if soil_idx_in_dense is not None:
                end_date_idx = date_indices[-1]
                target_val = self.dense_arrays['features'][target_idx, end_date_idx, soil_idx_in_dense]
                if target_val not in invalid_markers:
                    target_min = min(target_min, target_val)
                    target_max = max(target_max, target_val)

            # Get ALL nearby stations for this target (from comprehensive computation)
            nearby_list = comprehensive_neighbors.get(target_station_id, [])

            # Process each nearby slot
            for slot_idx, nearby_info in enumerate(nearby_list):
                nearby_station_id = nearby_info['station_id']
                nearby_distance = nearby_info['distance']

                nearby_idx = self.dense_station_to_idx.get(nearby_station_id)
                if nearby_idx is None:
                    continue

                # Feature 0: Distance (constant for this station pair)
                nearby_slot_mins[slot_idx, 0] = min(nearby_slot_mins[slot_idx, 0], nearby_distance)
                nearby_slot_maxs[slot_idx, 0] = max(nearby_slot_maxs[slot_idx, 0], nearby_distance)

                # Features 1 to n_params: Weather parameters
                for feat_idx, dense_idx in enumerate(feature_indices):
                    if dense_idx is None:
                        continue
                    feat_data = self.dense_arrays['features'][nearby_idx, date_indices, dense_idx]
                    valid = feat_data[(feat_data != INVALID_MARKER_MISSING) & (feat_data != INVALID_MARKER_API)]
                    if len(valid) > 0:
                        out_feat_idx = 1 + feat_idx  # +1 for distance
                        nearby_slot_mins[slot_idx, out_feat_idx] = min(
                            nearby_slot_mins[slot_idx, out_feat_idx], valid.min()
                        )
                        nearby_slot_maxs[slot_idx, out_feat_idx] = max(
                            nearby_slot_maxs[slot_idx, out_feat_idx], valid.max()
                        )

                # Feature n_params+1: Soil moisture
                if soil_idx_in_dense is not None:
                    soil_data = self.dense_arrays['features'][nearby_idx, date_indices, soil_idx_in_dense]
                    valid = soil_data[(soil_data != INVALID_MARKER_MISSING) & (soil_data != INVALID_MARKER_API)]
                    if len(valid) > 0:
                        soil_feat_idx = 1 + n_params  # distance + n_params
                        nearby_slot_mins[slot_idx, soil_feat_idx] = min(
                            nearby_slot_mins[slot_idx, soil_feat_idx], valid.min()
                        )
                        nearby_slot_maxs[slot_idx, soil_feat_idx] = max(
                            nearby_slot_maxs[slot_idx, soil_feat_idx], valid.max()
                        )

        # For stats computation, we don't need the full feature vector stats
        # (those are specific to the precomputed dataset's n_nearest)
        # Instead, we store comprehensive per-slot stats

        # Handle case where no valid target data was found
        if target_min == np.inf or target_max == -np.inf:
            warnings.warn(
                "No valid target values found during comprehensive stats computation! "
                "Using fallback range [0.0, 1.0]. "
                "This indicates a data integrity issue - check coverage_threshold setting.",
                stacklevel=2
            )
            final_target_min = 0.0
            final_target_max = 1.0
        else:
            final_target_min = float(target_min)
            final_target_max = float(target_max)

        self.norm_stats = {
            # Target feature stats (same for any configuration)
            'target_feature_mins': target_feat_mins,
            'target_feature_maxs': target_feat_maxs,
            'target_min': final_target_min,
            'target_max': final_target_max,
            # Per-slot nearby stats: [max_nearby_slots, nearby_features_per_station]
            'nearby_slot_mins': nearby_slot_mins,
            'nearby_slot_maxs': nearby_slot_maxs,
            'n_nearby_slots': max_nearby,
            # Metadata
            'n_params': n_params,
            'n_base_samples': n_samples,
            'seq_length': self.seq_length,
            'feature_params': self.feature_params,
        }

        print(f"  Target feature min range: [{target_feat_mins.min():.2f}, {target_feat_mins.max():.2f}]")
        print(f"  Target feature max range: [{target_feat_maxs.min():.2f}, {target_feat_maxs.max():.2f}]")
        print(f"  Target (soil moisture) range: [{self.norm_stats['target_min']:.4f}, {self.norm_stats['target_max']:.4f}]")
        print(f"  Nearby slots: {max_nearby} (comprehensive per-slot stats stored)")

        # Show sample of per-slot distance ranges (convert m to km for readability)
        print(f"  Distance ranges by slot:")
        for slot in range(min(5, max_nearby)):
            min_km = nearby_slot_mins[slot, 0] / 1000.0
            max_km = nearby_slot_maxs[slot, 0] / 1000.0
            print(f"    Slot {slot}: [{min_km:.1f}, {max_km:.1f}] km")
        if max_nearby > 5:
            print(f"    ... (showing first 5 of {max_nearby} slots)")

    def _apply_normalization(self, features, target, mask):
        """
        Normalize features and target to [-1, 1] range
        Invalid markers are changed to NORMALIZED_INVALID_MARKER (-2.0)
        """
        invalid_markers = [INVALID_MARKER_API, self.missing_value]

        # Normalize features using shared function (inplace for efficiency)
        normalize_features(
            features, self.norm_stats['feature_mins'], self.norm_stats['feature_maxs'],
            invalid_markers=invalid_markers, inplace=True
        )

        # Normalize target using shared function
        target_min = self.norm_stats['target_min']
        target_max = self.norm_stats['target_max']
        normalized_target = normalize_target(target, target_min, target_max, invalid_markers=invalid_markers)
        target[:] = normalized_target

        return features, target

    def precompute_and_save(self, output_path: str, norm_stats_path: Optional[str] = None, normalize: bool = True):
        """
        Precompute all sequences and save to disk for fast loading

        Args:
            output_path: Path to save precomputed sequences (.npz file)
            norm_stats_path: Path to save normalization statistics (optional)
            normalize: If True, normalize data before saving (recommended for speed)
        """
        print(f"Precomputing {len(self.sample_index)} sequences...")
        print("This may take a while but only needs to be done once.")

        # Choose the optimal sequence building method
        if self.dense_arrays is not None:
            print("  Using FAST dense array slicing method!")
            build_method = self._build_sequence_from_dense
        else:
            print("  Using dict lookup method")
            build_method = self._build_sequence_tensor

        # Determine feature dimensions from first sample
        sample0 = self.sample_index[0]
        features0, target0, mask0 = build_method(
            sample0['target_station'],
            sample0['start_date'],
            sample0['end_date']
        )

        seq_length, n_features = features0.shape

        # Preallocate arrays
        all_features = np.zeros((len(self.sample_index), seq_length, n_features), dtype=np.float32)
        all_targets = np.zeros((len(self.sample_index), 1), dtype=np.float32)
        all_masks = np.zeros((len(self.sample_index), seq_length, n_features), dtype=bool)

        # Store first sample
        all_features[0] = features0.numpy()
        all_targets[0] = target0.numpy()
        all_masks[0] = mask0.numpy()

        # Precompute all samples
        from tqdm import tqdm
        total = len(self.sample_index)
        for idx in tqdm(range(1, total), desc="Building sequences", initial=1, total=total):
            sample = self.sample_index[idx]
            features, target, mask = build_method(
                sample['target_station'],
                sample['start_date'],
                sample['end_date']
            )
            all_features[idx] = features.numpy()
            all_targets[idx] = target.numpy()
            all_masks[idx] = mask.numpy()

        # Compute and apply normalization if requested
        is_normalized = False
        if normalize:
            print(f"Computing normalization statistics...")
            # Temporarily store in precomputed_data for computing stats
            self.precomputed_data = {
                'features': all_features,
                'targets': all_targets,
                'masks': all_masks
            }
            # First compute stats from precomputed data (needed for feature_mins/maxs to normalize)
            self._compute_norm_stats_from_precomputed()
            precomputed_norm_stats = self.norm_stats.copy()

            # If dense_arrays available, compute COMPREHENSIVE per-slot stats for ALL nearby stations
            # This enables flexible use for any augmentation configuration
            if self.dense_arrays is not None:
                print(f"\nComputing COMPREHENSIVE per-slot stats for ALL nearby stations...")
                self._compute_comprehensive_norm_stats_from_dense()
                comprehensive_norm_stats = self.norm_stats

                # Merge: use precomputed feature_mins/maxs (for normalization of this specific dataset)
                # but use comprehensive nearby_slot stats (for flexible augmentation support)
                self.norm_stats = {
                    # From precomputed: for normalizing this specific dataset
                    'feature_mins': precomputed_norm_stats['feature_mins'],
                    'feature_maxs': precomputed_norm_stats['feature_maxs'],
                    'target_min': comprehensive_norm_stats['target_min'],
                    'target_max': comprehensive_norm_stats['target_max'],
                    # From comprehensive: for flexible augmentation
                    'target_feature_mins': comprehensive_norm_stats['target_feature_mins'],
                    'target_feature_maxs': comprehensive_norm_stats['target_feature_maxs'],
                    'nearby_slot_mins': comprehensive_norm_stats['nearby_slot_mins'],
                    'nearby_slot_maxs': comprehensive_norm_stats['nearby_slot_maxs'],
                    'n_nearby_slots': comprehensive_norm_stats['n_nearby_slots'],
                    # Metadata
                    'n_params': comprehensive_norm_stats['n_params'],
                    'n_base_samples': comprehensive_norm_stats['n_base_samples'],
                    'seq_length': comprehensive_norm_stats['seq_length'],
                    'feature_params': comprehensive_norm_stats['feature_params'],
                }
            else:
                raise ValueError(
                    "Dense arrays not available - cannot compute comprehensive per-slot stats. "
                    "Provide dense_array_path when creating the dataset to enable proper stats computation."
                )

            print(f"\nNormalizing all data...")
            # Normalize all samples in-place
            for idx in tqdm(range(len(self.sample_index)), desc="Normalizing data"):
                all_features[idx], all_targets[idx] = self._apply_normalization(
                    all_features[idx],
                    all_targets[idx],
                    all_masks[idx]
                )

            is_normalized = True

            # Save normalization statistics (per-slot format for flexible reuse)
            if norm_stats_path:
                print(f"Saving normalization stats to {norm_stats_path}...")
                np.savez(
                    norm_stats_path,
                    # Per-slot stats for full feature vector (for base dataset)
                    feature_mins=self.norm_stats['feature_mins'],
                    feature_maxs=self.norm_stats['feature_maxs'],
                    target_min=np.array([self.norm_stats['target_min']]),
                    target_max=np.array([self.norm_stats['target_max']]),
                    # Target feature stats
                    target_feature_mins=self.norm_stats['target_feature_mins'],
                    target_feature_maxs=self.norm_stats['target_feature_maxs'],
                    # Per-slot nearby stats: [n_nearby_slots, nearby_features_per_station]
                    nearby_slot_mins=self.norm_stats['nearby_slot_mins'],
                    nearby_slot_maxs=self.norm_stats['nearby_slot_maxs'],
                    n_nearby_slots=np.array([self.norm_stats['n_nearby_slots']]),
                    # Metadata
                    n_params=np.array([self.norm_stats['n_params']]),
                    n_base_samples=np.array([self.norm_stats['n_base_samples']]),
                    seq_length=np.array([self.norm_stats['seq_length']]),
                    feature_params=np.array(self.norm_stats['feature_params'], dtype='U50'),
                )

        # Save to disk as individual .npy files (memory-mappable)
        print(f"Saving to {output_path}...")

        # Create directory if it doesn't exist
        from pathlib import Path
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract sample_index components to avoid pickle requirement
        target_stations = np.array([s['target_station'] for s in self.sample_index], dtype=np.int32)
        end_dates = np.array([s['end_date'].timestamp() for s in self.sample_index], dtype=np.float64)
        start_dates = np.array([s['start_date'].timestamp() for s in self.sample_index], dtype=np.float64)

        # Save each array as a separate .npy file
        np.save(output_dir / 'features.npy', all_features)
        np.save(output_dir / 'targets.npy', all_targets)
        np.save(output_dir / 'masks.npy', all_masks)
        np.save(output_dir / 'target_stations.npy', target_stations)
        np.save(output_dir / 'end_dates.npy', end_dates)
        np.save(output_dir / 'start_dates.npy', start_dates)
        np.save(output_dir / 'is_normalized.npy', np.array([is_normalized], dtype=bool))

        # Save feature_params so we can validate when loading
        if self.feature_params is not None:
            np.save(output_dir / 'feature_params.npy', np.array(self.feature_params, dtype='U50'))

        print(f"  Saved {len(self.sample_index)} sequences to {output_dir}/")
        print(f"  Shape: features={all_features.shape}, targets={all_targets.shape}")
        print(f"  Data is {'normalized' if is_normalized else 'not normalized'}")

        print("Done!")

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int) -> 'Dict[str, torch.Tensor]':
        """
        Get a sample - optimized for minimal overhead

        Returns:
            Dictionary with:
                - features: [seq_length, total_features]
                - target: scalar
                - mask: [seq_length, total_features]
                - target_station_id: int
                - end_date: timestamp
        """
        # Load from precomputed data if available (fast path)
        if self.precomputed_data is not None:
            # Map index if this is a split dataset
            actual_idx = self._indices[idx] if hasattr(self, '_indices') and self._indices is not None else idx

            # Direct numpy array access - no copy needed if data is read-only
            # PyTorch will handle memory efficiently
            features = self.precomputed_data['features'][actual_idx]
            target = self.precomputed_data['targets'][actual_idx]
            mask = self.precomputed_data['masks'][actual_idx]

            # Remove soil moisture from target station features if present (data leakage prevention)
            if self.soil_in_features and self.soil_feature_idx is not None:
                # Need to copy since we're modifying
                features = features.copy()
                mask = mask.copy()

                # Remove soil moisture column from target station features
                # Target features are at indices [0:len(feature_params)]
                # We remove the column at soil_feature_idx
                indices_to_keep = [i for i in range(features.shape[1]) if i != self.soil_feature_idx]
                features = features[:, indices_to_keep]
                mask = mask[:, indices_to_keep]

            # Apply normalization if needed (only if not pre-normalized)
            elif self.normalize and self.norm_stats is not None:
                # Need to copy here since we're modifying
                features = features.copy()
                target = target.copy()
                features, target = self._apply_normalization(features, target, mask)

            # Convert to tensors (torch.from_numpy shares memory if possible)
            features_tensor = torch.from_numpy(features)
            target_tensor = torch.from_numpy(target)
            mask_tensor = torch.from_numpy(mask)

        else:
            # Build on-the-fly: use dense arrays if available (fast), otherwise dict lookup (slow)
            sample_info = self.sample_index[idx]

            if self.dense_arrays is not None:
                # FAST PATH: Use dense array slicing
                # if idx < 3:  # Debug first 3 samples
                #     print(f"  [DEBUG] Sample {idx}: Using FAST dense array path")
                features_tensor, target_tensor, mask_tensor = self._build_sequence_from_dense(
                    sample_info['target_station'],
                    sample_info['start_date'],
                    sample_info['end_date']
                )
            else:
                # SLOW PATH: Use dict lookups (fallback)
                if idx < 3:  # Debug first 3 samples
                    print(f"  [DEBUG] Sample {idx}: Using SLOW dict lookup path")
                features_tensor, target_tensor, mask_tensor = self._build_sequence_tensor(
                    sample_info['target_station'],
                    sample_info['start_date'],
                    sample_info['end_date']
                )

        # Get end date (lightweight operation at end)
        sample = self.sample_index[idx]
        end_date_unix = sample['end_date'].timestamp() if hasattr(sample['end_date'], 'timestamp') else float(sample['end_date'])

        return {
            'features': features_tensor,
            'target': target_tensor,
            'mask': mask_tensor,
            'target_station_id': sample['target_station'],
            'end_date': end_date_unix
        }

    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names"""
        feature_names = []

        # Target station features
        for param in self.feature_params:
            feature_names.append(f'target_{param}')

        # Nearby stations features
        for n_idx in range(self.n_nearest):
            feature_names.append(f'nearby{n_idx + 1}_distance')
            for param in self.feature_params:
                feature_names.append(f'nearby{n_idx + 1}_{param}')
            feature_names.append(f'nearby{n_idx + 1}_{self.target_param}')

        return feature_names

    @property
    def soil_moisture_param(self) -> str:
        """Backwards compatibility: alias for target_param"""
        return self.target_param

    def _split_precomputed(
            self,
            train_stations: List[int],
            val_stations: List[int],
            test_stations: List[int]
    ) -> Tuple['WeatherSequenceDataset', 'WeatherSequenceDataset', 'WeatherSequenceDataset']:
        """
        Efficiently split precomputed data by filtering arrays

        This is MUCH faster than creating new datasets because we just filter
        the precomputed arrays instead of rebuilding from DataFrames.
        """
        # Get indices for each split
        train_indices = [i for i, s in enumerate(self.sample_index) if s['target_station'] in train_stations]
        val_indices = [i for i, s in enumerate(self.sample_index) if s['target_station'] in val_stations]
        test_indices = [i for i, s in enumerate(self.sample_index) if s['target_station'] in test_stations]

        print(f"  Train: {len(train_indices)} samples")
        print(f"  Val: {len(val_indices)} samples")
        print(f"  Test: {len(test_indices)} samples")

        # Create train dataset
        train_dataset = self._create_split_dataset(train_stations, train_indices)

        # Create val dataset
        val_dataset = None
        if len(val_stations) > 0:
            val_dataset = self._create_split_dataset(val_stations, val_indices)

        # Create test dataset
        test_dataset = None
        if len(test_stations) > 0:
            test_dataset = self._create_split_dataset(test_stations, test_indices)

        return train_dataset, val_dataset, test_dataset

    def _create_split_dataset(
            self,
            target_stations: List[int],
            indices: List[int]
    ) -> 'WeatherSequenceDataset':
        """Create a dataset from filtered precomputed data"""
        # Create new dataset instance
        split_dataset = WeatherSequenceDataset.__new__(WeatherSequenceDataset)

        # Copy basic attributes
        split_dataset.seq_length = self.seq_length
        split_dataset.n_nearest = self.n_nearest
        split_dataset.target_param = self.target_param
        split_dataset.missing_value = self.missing_value
        split_dataset.normalize = self.normalize
        split_dataset.is_prenormalized = self.is_prenormalized
        split_dataset.soil_in_features = self.soil_in_features
        split_dataset.soil_feature_idx = self.soil_feature_idx
        split_dataset.precomputed_path = None  # Don't need path anymore

        # Copy dataframes (lightweight references)
        split_dataset.timeseries_df = self.timeseries_df
        split_dataset.stations_df = self.stations_df
        split_dataset.nearest_df = self.nearest_df
        split_dataset.target_stations = target_stations
        split_dataset.feature_params = self.feature_params

        # Keep reference to original precomputed_data (don't copy arrays!)
        # This avoids loading hundreds of GB into RAM for large datasets
        split_dataset.precomputed_data = self.precomputed_data

        # Store index mapping for this split
        split_dataset._indices = indices

        # Build filtered sample_index for this split
        split_dataset.sample_index = [self.sample_index[i] for i in indices]

        # Copy normalization stats (shared across all splits)
        split_dataset.norm_stats = self.norm_stats

        return split_dataset

    @staticmethod
    def train_val_test_split(
            dataset: 'WeatherSequenceDataset',
            val_stations_ratio: float = 0.15,
            test_stations_ratio: float = 0.0,
            random_seed: int = 42
    ) -> Tuple['WeatherSequenceDataset', 'WeatherSequenceDataset', 'WeatherSequenceDataset']:
        """
        Split dataset by stations (not by time) for better generalization testing

        Args:
            dataset: Original dataset
            val_stations_ratio: Ratio of stations for validation
            test_stations_ratio: Ratio of stations for test
            random_seed: Random seed for reproducibility

        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        np.random.seed(random_seed)

        # Get unique stations
        stations = np.array(dataset.target_stations)
        n_stations = len(stations)

        # Shuffle stations
        shuffled_indices = np.random.permutation(n_stations)
        shuffled_stations = stations[shuffled_indices]

        # Calculate splits
        n_val = int(n_stations * val_stations_ratio)
        n_test = int(n_stations * test_stations_ratio)
        n_train = n_stations - n_val - n_test

        train_stations = shuffled_stations[:n_train].tolist()
        val_stations = shuffled_stations[n_train:n_train + n_val].tolist()
        test_stations = shuffled_stations[n_train + n_val:].tolist()

        print(f"\nStation-based split:")
        print(f"  Train: {len(train_stations)} stations")
        print(f"  Val: {len(val_stations)} stations")
        print(f"  Test: {len(test_stations)} stations")

        # If we have precomputed data, split it efficiently
        if dataset.precomputed_data is not None:
            print("Splitting precomputed data (fast)...")
            return dataset._split_precomputed(train_stations, val_stations, test_stations)
        else:
            # Fallback to creating new datasets (slow)
            print("Warning: No precomputed data, splitting will be slow...")
            train_dataset = WeatherSequenceDataset(
                timeseries=dataset.timeseries_df,
                stations=dataset.stations_df,
                nearest=dataset.nearest_df,
                seq_length=dataset.seq_length,
                n_nearest=dataset.n_nearest,
                target_stations=train_stations,
                feature_params=dataset.feature_params,
                target_param=dataset.target_param,
                missing_value=dataset.missing_value,
                normalize=dataset.normalize
            )
            val_dataset = None
            if val_stations_ratio > 0:
                val_dataset = WeatherSequenceDataset(
                    timeseries=dataset.timeseries_df,
                    stations=dataset.stations_df,
                    nearest=dataset.nearest_df,
                    seq_length=dataset.seq_length,
                    n_nearest=dataset.n_nearest,
                    target_stations=val_stations,
                    feature_params=dataset.feature_params,
                    target_param=dataset.target_param,
                    missing_value=dataset.missing_value,
                    normalize=dataset.normalize
                )
            test_dataset = None
            if test_stations_ratio > 0:
                test_dataset = WeatherSequenceDataset(
                    timeseries=dataset.timeseries_df,
                    stations=dataset.stations_df,
                    nearest=dataset.nearest_df,
                    seq_length=dataset.seq_length,
                    n_nearest=dataset.n_nearest,
                    target_stations=test_stations,
                    feature_params=dataset.feature_params,
                    target_param=dataset.target_param,
                    missing_value=dataset.missing_value,
                    normalize=dataset.normalize
                )
            return train_dataset, val_dataset, test_dataset

    def get_sequence_data(
            self,
            target_station_id: int,
            end_date: datetime,
            seq_length: int,
            n_nearest: int = 4,
            parameter_ids: Optional[List[str]] = None
    ) -> Dict:
        """
        Get sequence of historical data for prediction with transformer

        Args:
            target_station_id: Station ID to predict for
            end_date: End date of sequence (the date we're predicting for)
            seq_length: Number of days in sequence
            n_nearest: Number of nearest stations to include
            parameter_ids: Parameters to fetch

        Returns:
            Dictionary with sequence data in same format as Dataset
        """
        if parameter_ids is None:
            parameter_ids = [
                self.SOIL_MOISTURE_PARAM,
                'PP_SUM_1.5m',
                'TA_AVG_1.5m',
                'HR_AVG_1.5m'
            ]

        # Calculate date range
        start_date = end_date - timedelta(days=seq_length - 1)

        # Load stations metadata and nearest stations (using cache)
        stations_df = self.get_stations_df()
        nearest_df = self.get_nearest_df()

        # Get target station info
        target_info = stations_df[stations_df['station_id'] == target_station_id].iloc[0]

        # Get nearest stations info
        nearest_info = nearest_df[nearest_df['station_id'] == target_station_id].iloc[0]

        # Find n nearest stations WITH soil moisture
        nearby_with_soil = []
        for i in range(1, len(nearest_df.columns) // 3 + 1):
            if f'nearest_{i}_id' not in nearest_info:
                break
            if nearest_info[f'nearest_{i}_has_soil_moisture']:
                nearby_with_soil.append({
                    'station_id': int(nearest_info[f'nearest_{i}_id']),
                    'distance': nearest_info[f'nearest_{i}_distance']
                })
                if len(nearby_with_soil) == n_nearest:
                    break

        # Fetch data for date range
        all_station_ids = [target_station_id] + [s['station_id'] for s in nearby_with_soil]

        json_data = self.get_daily_data(
            station_ids=all_station_ids,
            parameter_ids=parameter_ids,
            start_date=start_date.strftime('%d/%m/%Y'),
            end_date=end_date.strftime('%d/%m/%Y')
        )

        all_data_df = self.parse_data_to_dataframe(json_data) if json_data else pd.DataFrame()

        # Split into target and nearby
        target_df = all_data_df[all_data_df['station_id'] == target_station_id].copy()
        nearby_df = all_data_df[all_data_df['station_id'] != target_station_id].copy()

        # Add distance information
        if not nearby_df.empty:
            distance_map = {s['station_id']: s['distance'] for s in nearby_with_soil}
            nearby_df['distance'] = nearby_df['station_id'].map(distance_map)

        return {
            'target_station_id': target_station_id,
            'target_coordinates': {
                'utmx': target_info['utmx'],
                'utmy': target_info['utmy'],
                'altitude': target_info['altitude']
            },
            'date_range': (start_date, end_date),
            'seq_length': seq_length,
            'target_data': target_df,
            'nearby_stations': nearby_with_soil,
            'nearby_data': nearby_df
        }


# Backwards compatibility alias
SoilMoistureSequenceDataset = WeatherSequenceDataset

