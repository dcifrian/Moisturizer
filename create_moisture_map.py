#!/usr/bin/env python3
"""
Create a soil moisture map of all Galicia

Shows:
- Real soil moisture data where sensors exist (solid colors)
- Predicted soil moisture for stations without sensors (hatched/different style)
- Spatial interpolation across the entire region

Usage:
    python create_moisture_map.py --model path/to/model.pth --date 2024-01-15
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
from datetime import datetime, timedelta
import torch
from pathlib import Path

from Moisturizer import (
    expand_canonical_to_augmented_stats,
    normalize_features,
    denormalize_target,
    FeatureLayout,
    INVALID_MARKER_API,
    INVALID_MARKER_MISSING,
    NORMALIZED_INVALID_MARKER,
)
from MeteoGaliciaCollector import MeteoGaliciaCollector
from WeatherSequenceDataset import WeatherSequenceDataset
from model_loader import load_model

# Constants for distance calculations
DEG_TO_KM = 111.0  # Approximate km per degree latitude
IMPUTE_DISTANCE_THRESHOLD = 1.05  # 5% tolerance for including additional nearby stations
OFFENDER_DISTANCE_KM = 1.0  # Max km distance to consider when debugging prediction offenders


def load_coastline_data(lon_min, lon_max, lat_min, lat_max, padding=0.15, cache_dir=None):
    """
    Load and prepare coastline data early to fail fast if there are issues.
    Uses Galicia's administrative boundary when available, falls back to land data.
    
    Downloads are cached to avoid re-downloading on subsequent runs.

    Returns:
        tuple: (coastline_points, galicia_land)
    """
    import geopandas as gpd
    from shapely.geometry import box, Point, LineString
    import os
    import hashlib
    
    # Set up cache directory
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(__file__), '.geo_cache')
    os.makedirs(cache_dir, exist_ok=True)

    print("  Loading boundary data...")
    
    galicia_land = None
    
    def cached_read_file(url):
        """Read a file from URL, caching locally to avoid re-downloads."""
        # Create a filename from the URL
        url_hash = hashlib.md5(url.encode()).hexdigest()[:10]
        filename = os.path.basename(url).replace('.zip', '')
        cache_path = os.path.join(cache_dir, f"{filename}_{url_hash}")
        
        if os.path.exists(cache_path):
            print(f"    Using cached: {os.path.basename(cache_path)}")
            return gpd.read_file(cache_path)
        else:
            print(f"    Downloading: {os.path.basename(url)}...")
            gdf = gpd.read_file(url)
            # Save to cache
            gdf.to_file(cache_path, driver='GPKG')
            print(f"    Cached to: {os.path.basename(cache_path)}")
            return gdf
    
    # Try to load Galicia's administrative boundary (more accurate than land data)
    try:
        print("    Attempting to load Galicia administrative boundary...")
        admin_url = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip"
        admin = cached_read_file(admin_url)
        
        # Natural Earth uses 'admin' for country name and 'name' for region name
        # Try multiple approaches to find Galicia
        galicia_rows = None
        
        # Approach 1: Look for Spain entries first, then find Galicia
        if 'admin' in admin.columns:
            spain = admin[admin['admin'].str.contains('Spain', case=False, na=False)]
            print(f"    Found {len(spain)} Spain regions")
            if len(spain) > 0:
                # Look for Galicia in the name column
                galicia_rows = spain[spain['name'].str.contains('Galicia', case=False, na=False)]
                if len(galicia_rows) == 0 and 'name_en' in spain.columns:
                    galicia_rows = spain[spain['name_en'].str.contains('Galicia', case=False, na=False)]
        
        # Approach 2: Direct search in all data
        if galicia_rows is None or len(galicia_rows) == 0:
            for col in ['name', 'name_en', 'name_local', 'gn_name', 'woe_name']:
                if col in admin.columns:
                    matches = admin[admin[col].str.contains('Galicia', case=False, na=False)]
                    if len(matches) > 0:
                        galicia_rows = matches
                        print(f"    Found Galicia via column '{col}'")
                        break
        
        if galicia_rows is not None and len(galicia_rows) > 0:
            galicia_land = galicia_rows.geometry.union_all()
            print(f"    ✓ Found Galicia administrative boundary")
        else:
            # Print some Spain region names to help debug
            if 'admin' in admin.columns:
                spain = admin[admin['admin'].str.contains('Spain', case=False, na=False)]
                if len(spain) > 0:
                    print(f"    Spain region names: {list(spain['name'].head(10))}")
            print(f"    Could not find Galicia in admin boundaries, falling back to land data")
    except Exception as e:
        print(f"    Could not load admin boundaries ({e}), falling back to land data")
    
    # Fall back to Natural Earth land data if admin boundary not found
    if galicia_land is None:
        print("    Loading Natural Earth land data...")
        url = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip"
        world = cached_read_file(url)

        # Add padding to bounding box
        lon_pad = (lon_max - lon_min) * padding
        lat_pad = (lat_max - lat_min) * padding

        # Clip to our region of interest
        bbox = box(lon_min - lon_pad, lat_min - lat_pad, lon_max + lon_pad, lat_max + lat_pad)

        # Get land areas in our bbox
        galicia_land = world.geometry.intersection(bbox)
        galicia_land = galicia_land[~galicia_land.is_empty].union_all()

    print("  Sampling coastline/boundary points...")
    # Extract exterior boundary
    coastlines = []
    if hasattr(galicia_land, 'geoms'):
        # MultiPolygon - get all exteriors
        for geom in galicia_land.geoms:
            if hasattr(geom, 'exterior'):
                coastlines.append(geom.exterior)
    elif hasattr(galicia_land, 'exterior'):
        # Single Polygon
        coastlines = [galicia_land.exterior]

    # Sample points along the coastline(s)
    coastline_points = []
    n_points_per_coastline = max(10000 // len(coastlines), 100) if coastlines else 0

    for coastline in coastlines:
        # Sample points along this coastline
        coords = list(coastline.coords)
        if len(coords) < 2:
            continue

        # Interpolate points evenly along the line
        line = LineString(coords)
        total_length = line.length

        for i in range(n_points_per_coastline):
            distance = (i / n_points_per_coastline) * total_length
            point = line.interpolate(distance)
            coastline_points.append([point.x, point.y])

    coastline_points = np.array(coastline_points)
    print(f"  ✓ Loaded boundary with {len(coastline_points)} sampled points")

    return coastline_points, galicia_land



def predict_for_station(model, dataset, station_id, end_date, device='cuda'):
    """
    Get soil moisture prediction for a specific station and date

    Args:
        model: Trained TROLOLO model
        dataset: WeatherSequenceDataset
        station_id: Station ID to predict for
        end_date: Date to predict (as pandas Timestamp)
        device: 'cuda' or 'cpu'

    Returns:
        Predicted soil moisture value (denormalized)
    """
    # Find sample with matching station and date
    matching_samples = [
        i for i, s in enumerate(dataset.sample_index)
        if s['target_station'] == station_id and s['end_date'] == end_date
    ]

    if not matching_samples:
        return None

    # Get the sample
    sample_idx = matching_samples[0]
    sample = dataset[sample_idx]

    x_gpu = torch.zeros([1, model.embed_dim - 2, model.seq_length - model.n_class_tokens], dtype=torch.float16, device="cuda")
    with (torch.inference_mode(), torch.autocast(device_type='cuda', enabled=True, cache_enabled=True, dtype=torch.bfloat16)):
            data = sample["features"]
            X_batch = data
            x_gpu[:1, :X_batch.shape[2], :].copy_(X_batch.permute(0, 2, 1), non_blocking=True)
            x = x_gpu[:1, :, :]
            pred_value = model(x).cpu().item()

    return pred_value


def denormalize_soil_moisture(normalized_value, norm_stats_path):
    """Convert from [-1, 1] back to original soil moisture range"""
    norm_stats = np.load(norm_stats_path)
    # Handle scalar, 0-d array, and 1-d array formats
    target_min_val = norm_stats['target_min']
    target_max_val = norm_stats['target_max']
    # Use .item() for 0-d arrays, [0] for 1-d arrays, direct float for scalars
    if hasattr(target_min_val, 'ndim'):
        target_min = float(target_min_val.item()) if target_min_val.ndim == 0 else float(target_min_val[0])
        target_max = float(target_max_val.item()) if target_max_val.ndim == 0 else float(target_max_val[0])
    else:
        target_min = float(target_min_val)
        target_max = float(target_max_val)

    return denormalize_target(normalized_value, target_min, target_max)


def get_real_soil_moisture_from_lookup(timeseries_lookup, station_id, date):
    """
    Get actual soil moisture from pre-built lookup dict.
    
    OPTIMIZED: Uses pre-built dict instead of re-reading CSV every time.
    """
    date_str = str(date.date()) if hasattr(date, 'date') else str(date)[:10]
    key = (int(station_id), date_str, 'HS_CV_AVG_-0.2m')
    return timeseries_lookup.get(key)


def build_fast_timeseries_lookup(timeseries_df, start_date, end_date, station_ids, feature_params):
    """
    Pre-process timeseries into fast numpy lookup structure.
    
    OPTIMIZED: Uses vectorized numpy operations instead of iterrows.

    Returns dict: {(station_id, date_str, parameter_code): value}
    """
    # Convert station_ids to set for fast lookup
    station_ids_set = set(station_ids)
    
    # Filter using numpy boolean indexing (much faster than pandas)
    dates = timeseries_df['date'].values
    sids = timeseries_df['station_id'].values
    
    # Create boolean mask vectorized
    date_mask = (dates >= np.datetime64(start_date)) & (dates <= np.datetime64(end_date))
    station_mask = np.isin(sids, list(station_ids_set))
    mask = date_mask & station_mask
    
    # Extract filtered arrays directly (no DataFrame overhead)
    filtered_stations = sids[mask].astype(np.int32)
    filtered_dates = dates[mask]
    filtered_params = timeseries_df['parameter_code'].values[mask]
    filtered_values = timeseries_df['value'].values[mask].astype(np.float32)
    
    # Convert dates to strings vectorized
    # Use pandas for efficient datetime->string conversion
    date_strings = pd.to_datetime(filtered_dates).strftime('%Y-%m-%d').values
    
    # Build lookup dict using zip (much faster than iterrows)
    lookup = dict(zip(
        zip(filtered_stations, date_strings, filtered_params),
        filtered_values
    ))

    return lookup


# =============================================================================
# Shared helper functions for building sequences
# =============================================================================

def _init_sequence_arrays(end_date, seq_length, feature_params, n_nearest, missing_value=-1000.0):
    """
    Initialize arrays and compute layout for sequence building.

    Returns:
        dict with keys: date_strings, layout, features, mask, coordinate_features,
                        target_features_per_timestep, nearby_features_per_timestep, total_features
    """
    start_date = end_date - timedelta(days=seq_length - 1)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    date_strings = [str(d.date()) for d in date_range]

    coordinate_features = {'altitude', 'utmx', 'utmy'}

    layout = FeatureLayout(n_params=len(feature_params), n_nearby=n_nearest)
    target_features_per_timestep = layout.n_target_features
    nearby_features_per_timestep = layout.nearby_features_per_station
    total_features = layout.n_total_features

    features = np.full((seq_length, total_features), missing_value, dtype=np.float32)
    mask = np.zeros((seq_length, total_features), dtype=bool)

    return {
        'date_strings': date_strings,
        'layout': layout,
        'features': features,
        'mask': mask,
        'coordinate_features': coordinate_features,
        'target_features_per_timestep': target_features_per_timestep,
        'nearby_features_per_timestep': nearby_features_per_timestep,
        'total_features': total_features,
    }


def _fill_target_coordinate_features(features, mask, target_coords, feature_params, coordinate_features):
    """
    Fill coordinate features for the target station (constant across all timesteps).
    Modifies features and mask in-place.
    """
    for f_idx, param in enumerate(feature_params):
        if param in coordinate_features:
            coord_value = target_coords.get(param)
            if coord_value is not None:
                features[:, f_idx] = coord_value
                mask[:, f_idx] = True


def _fill_nearby_coordinates_and_distances(features, mask, nearby_stations, stations_lookup,
                                           feature_params, coordinate_features,
                                           target_features_per_timestep, nearby_features_per_timestep,
                                           n_nearest):
    """
    Fill coordinate features and distances for nearby stations (constant across all timesteps).
    Modifies features and mask in-place.
    """
    for n_idx, nearby in enumerate(nearby_stations[:n_nearest]):
        nearby_offset = target_features_per_timestep + (n_idx * nearby_features_per_timestep)

        # Distance (constant across time)
        features[:, nearby_offset] = nearby['distance']
        mask[:, nearby_offset] = True

        # Coordinate features for nearby station
        nid = nearby['station_id']
        if nid in stations_lookup:
            ncoords = stations_lookup[nid]
            for f_idx_nearby, param in enumerate(feature_params):
                if param in coordinate_features:
                    feat_idx = nearby_offset + 1 + f_idx_nearby
                    coord_value = ncoords.get(param)
                    if coord_value is not None:
                        features[:, feat_idx] = coord_value
                        mask[:, feat_idx] = True


def _fill_nearby_timeseries_features(features, mask, date_strings, nearby_stations,
                                      timeseries_lookup, feature_params, coordinate_features,
                                      target_features_per_timestep, nearby_features_per_timestep,
                                      n_nearest):
    """
    Fill timeseries features (weather + soil moisture) for nearby stations.
    Modifies features and mask in-place.
    """
    for t, date_str in enumerate(date_strings):
        for n_idx, nearby in enumerate(nearby_stations[:n_nearest]):
            nearby_offset = target_features_per_timestep + (n_idx * nearby_features_per_timestep)
            nid = nearby['station_id']

            # Weather features for this nearby station
            for f_idx_nearby, param in enumerate(feature_params):
                if param not in coordinate_features:
                    feat_idx = nearby_offset + 1 + f_idx_nearby
                    key = (nid, date_str, param)
                    if key in timeseries_lookup:
                        features[t, feat_idx] = timeseries_lookup[key]
                        mask[t, feat_idx] = True

            # Soil moisture for this nearby station
            key = (nid, date_str, 'HS_CV_AVG_-0.2m')
            soil_idx = nearby_offset + 1 + len(feature_params)
            if key in timeseries_lookup:
                features[t, soil_idx] = timeseries_lookup[key]
                mask[t, soil_idx] = True


def _fill_target_timeseries_direct(features, mask, date_strings, station_id,
                                   timeseries_lookup, feature_params, coordinate_features):
    """
    Fill target station timeseries features using direct lookup (for real stations).
    Modifies features and mask in-place.
    """
    for t, date_str in enumerate(date_strings):
        for f_idx, param in enumerate(feature_params):
            if param not in coordinate_features:
                key = (station_id, date_str, param)
                if key in timeseries_lookup:
                    features[t, f_idx] = timeseries_lookup[key]
                    mask[t, f_idx] = True


def _fill_target_timeseries_interpolated(features, mask, date_strings, interpolation_stations,
                                          timeseries_lookup, feature_params, coordinate_features):
    """
    Fill target station timeseries features using interpolation (for virtual stations).
    Uses inverse distance weighting from selected interpolation stations.
    Modifies features and mask in-place.
    """
    # Compute interpolation weights
    interp_distances = np.array([ns['distance'] for ns in interpolation_stations])
    interp_weights = 1.0 / (interp_distances + 1.0)  # +1 meter epsilon
    interp_weights = interp_weights / interp_weights.sum()

    nearest_distance = interpolation_stations[0]['distance']

    for t, date_str in enumerate(date_strings):
        for f_idx, param in enumerate(feature_params):
            if param not in coordinate_features:
                # Check nearest station first
                nearest_key = (interpolation_stations[0]['station_id'], date_str, param)
                nearest_val = timeseries_lookup.get(nearest_key)

                if nearest_val is not None and nearest_val > -9000:
                    # Nearest has valid data - interpolate from all stations
                    values = [nearest_val]
                    valid_weights = [interp_weights[0]]

                    for i in range(1, len(interpolation_stations)):
                        key = (interpolation_stations[i]['station_id'], date_str, param)
                        val = timeseries_lookup.get(key)
                        if val is not None and val > -9000:
                            values.append(val)
                            valid_weights.append(interp_weights[i])

                    # Interpolate
                    valid_weights = np.array(valid_weights)
                    valid_weights = valid_weights / valid_weights.sum()
                    features[t, f_idx] = float(np.sum(np.array(values) * valid_weights))
                    mask[t, f_idx] = True
                else:
                    # Nearest has missing data - check if we should impute from others
                    allow_impute = False
                    for i in range(1, len(interpolation_stations)):
                        if interpolation_stations[i]['distance'] <= nearest_distance * IMPUTE_DISTANCE_THRESHOLD:
                            allow_impute = True
                            break

                    if allow_impute:
                        # Try to get value from other stations
                        values = []
                        valid_weights = []
                        for i in range(1, len(interpolation_stations)):
                            key = (interpolation_stations[i]['station_id'], date_str, param)
                            val = timeseries_lookup.get(key)
                            if val is not None and val > -9000:
                                values.append(val)
                                valid_weights.append(interp_weights[i])

                        if values:
                            valid_weights = np.array(valid_weights)
                            valid_weights = valid_weights / valid_weights.sum()
                            features[t, f_idx] = float(np.sum(np.array(values) * valid_weights))
                            mask[t, f_idx] = True
                    # else: keep as missing_value (-1000) -> -2 after normalization


def build_sequence_for_any_station(
    station_id,
    end_date,
    timeseries_lookup,
    nearest_lookup,  # Pre-built dict: station_id -> list of nearby stations
    stations_lookup,  # Pre-built dict: station_id -> station row dict
    feature_params,
    norm_stats,
    seq_length=64,
    n_nearest=4,
    missing_value=-1000.0
):
    """
    Build a sequence for ANY station (even without soil moisture sensor)

    This allows us to predict for stations without sensors by using their
    weather data + nearby stations with sensors as context.

    Args:
        timeseries_lookup: Pre-built dict from build_fast_timeseries_lookup
        nearest_lookup: Pre-built dict: station_id -> [{'station_id': X, 'distance': Y}, ...]
        stations_lookup: Pre-built dict: station_id -> {'altitude': X, 'utmx': Y, ...}

    OPTIMIZED: Uses pre-built dict lookups instead of DataFrame filtering.
    """
    # Get nearest stations from pre-built lookup
    if station_id not in nearest_lookup:
        return None

    nearby_stations = nearest_lookup[station_id][:n_nearest]

    if len(nearby_stations) < n_nearest:
        return None

    # Get target station metadata from pre-built lookup
    if station_id not in stations_lookup:
        return None
    target_station_coords = stations_lookup[station_id]

    # Initialize arrays and layout
    init = _init_sequence_arrays(end_date, seq_length, feature_params, n_nearest, missing_value)
    features = init['features']
    mask = init['mask']
    date_strings = init['date_strings']
    coordinate_features = init['coordinate_features']

    # Fill coordinate features for target station (constant across time)
    _fill_target_coordinate_features(features, mask, target_station_coords,
                                     feature_params, coordinate_features)

    # Fill nearby station coordinate features and distances (constant across time)
    _fill_nearby_coordinates_and_distances(
        features, mask, nearby_stations, stations_lookup,
        feature_params, coordinate_features,
        init['target_features_per_timestep'], init['nearby_features_per_timestep'],
        n_nearest
    )

    # Fill target station timeseries features (direct lookup)
    _fill_target_timeseries_direct(features, mask, date_strings, station_id,
                                   timeseries_lookup, feature_params, coordinate_features)

    # Fill nearby stations timeseries features (weather + soil moisture)
    _fill_nearby_timeseries_features(
        features, mask, date_strings, nearby_stations,
        timeseries_lookup, feature_params, coordinate_features,
        init['target_features_per_timestep'], init['nearby_features_per_timestep'],
        n_nearest
    )

    # Apply normalization
    features_normalized = apply_normalization_to_features(features, mask, norm_stats, missing_value)

    return features_normalized, mask

def apply_normalization_to_features(features, mask, norm_stats, missing_value=-1000.0):
    """Apply normalization to features (same as in Dataset)"""
    feature_mins = norm_stats['feature_mins']
    feature_maxs = norm_stats['feature_maxs']

    # Verify dimensions match
    if features.shape[1] != len(feature_mins):
        raise ValueError(
            f"Feature dimension mismatch! "
            f"Built sequence has {features.shape[1]} features, "
            f"but normalization stats have {len(feature_mins)} features. "
            f"This means the on-the-fly sequence structure doesn't match training data structure."
        )

    # Use shared normalization function
    invalid_markers = [INVALID_MARKER_API, missing_value]
    return normalize_features(features, feature_mins, feature_maxs, invalid_markers=invalid_markers)


# =============================================================================
# Ensemble prediction with live augmentation
# =============================================================================

def build_augmentation_column_indices(n_target_features, nearby_features_per_station,
                                       n_nearby_available, n_nearby_output,
                                       use_skip_patterns=True):
    """
    Build column indices for all augmentation patterns.

    Args:
        n_target_features: Number of target station features
        nearby_features_per_station: Features per nearby station (including distance, soil moisture)
        n_nearby_available: How many nearby stations are in the source data
        n_nearby_output: How many nearby stations should be in the output
        use_skip_patterns: If True, also generate skip patterns (drop one station at a time)
                          If False, only generate permutations (n_nearby_available must equal n_nearby_output)

    Returns:
        List of numpy arrays, each containing column indices for one augmentation
    """
    from itertools import permutations

    # Build skip patterns
    if use_skip_patterns and n_nearby_available > n_nearby_output:
        skip_patterns = []
        for skip_idx in range(n_nearby_available):
            keep_indices = [i for i in range(n_nearby_available) if i != skip_idx][:n_nearby_output]
            skip_patterns.append(keep_indices)
    else:
        # No skipping - use all stations
        skip_patterns = [list(range(n_nearby_output))]

    # Generate all permutations
    all_perms = list(permutations(range(n_nearby_output)))

    # Build column indices for each skip pattern + permutation combo
    column_indices = []
    target_cols = list(range(n_target_features))

    for skip_pattern in skip_patterns:
        for perm in all_perms:
            cols = target_cols.copy()
            # Apply skip pattern then permutation
            permuted_stations = [skip_pattern[p] for p in perm]

            for source_station in permuted_stations:
                source_start = n_target_features + (source_station * nearby_features_per_station)
                source_end = source_start + nearby_features_per_station
                cols.extend(range(source_start, source_end))

            column_indices.append(np.array(cols, dtype=np.int64))

    return column_indices


def build_sequence_for_ensemble(
    station_id,
    end_date,
    timeseries_lookup,
    nearest_lookup,
    stations_lookup,
    feature_params,
    norm_stats,
    seq_length=64,
    n_nearby_output=4,
    n_nearby_available=5,
    missing_value=-1000.0
):
    """
    Build a sequence with extra nearby stations for ensemble augmentation.

    Returns (features_norm, mask) with n_nearby_available stations,
    or None if insufficient data.
    """
    if station_id not in nearest_lookup:
        return None

    nearby_stations = nearest_lookup[station_id][:n_nearby_available]
    if len(nearby_stations) < n_nearby_available:
        return None

    if station_id not in stations_lookup:
        return None
    target_station_coords = stations_lookup[station_id]

    # Initialize with n_nearby_available stations (not n_nearby_output)
    init = _init_sequence_arrays(end_date, seq_length, feature_params, n_nearby_available, missing_value)
    features = init['features']
    mask = init['mask']
    date_strings = init['date_strings']
    coordinate_features = init['coordinate_features']

    _fill_target_coordinate_features(features, mask, target_station_coords,
                                     feature_params, coordinate_features)

    _fill_nearby_coordinates_and_distances(
        features, mask, nearby_stations, stations_lookup,
        feature_params, coordinate_features,
        init['target_features_per_timestep'], init['nearby_features_per_timestep'],
        n_nearby_available
    )

    _fill_target_timeseries_direct(features, mask, date_strings, station_id,
                                   timeseries_lookup, feature_params, coordinate_features)

    _fill_nearby_timeseries_features(
        features, mask, date_strings, nearby_stations,
        timeseries_lookup, feature_params, coordinate_features,
        init['target_features_per_timestep'], init['nearby_features_per_timestep'],
        n_nearby_available
    )

    # Apply normalization using expanded stats for n_nearby_available
    expanded_stats = expand_canonical_to_augmented_stats(
        norm_stats,
        n_slots_needed=n_nearby_available
    )
    features_normalized = apply_normalization_to_features(features, mask, expanded_stats, missing_value)

    return features_normalized, mask


def _run_ensemble_inference(sequences_to_predict, model, device, collector,
                            n_nearby_output=4, n_nearby_available=5,
                            use_skip_patterns=True):
    """
    Run ensemble inference using augmentation patterns.

    Each station gets multiple predictions (one per augmentation pattern),
    which are then averaged. Processes one augmentation pattern per batch
    to avoid memory blowup.

    Args:
        sequences_to_predict: List of (station_info, features_norm, mask)
                             features_norm has n_nearby_available nearby stations
        model: Trained model
        device: 'cuda' or 'cpu'
        collector: MeteoGaliciaCollector instance
        n_nearby_output: Number of nearby stations in model input (4)
        n_nearby_available: Number of nearby stations in source data (5)
        use_skip_patterns: If True, use skip patterns + permutations
                          If False, use permutations only

    Returns:
        list: predicted_results with averaged predictions
    """
    if not sequences_to_predict:
        return []

    # Get feature layout
    feature_params = np.load(collector.data_dir / "normalization_stats.npz", allow_pickle=True)['feature_params']
    n_params = len(feature_params)
    layout = FeatureLayout(n_params=n_params, n_nearby=n_nearby_output)

    # Build column indices for all augmentation patterns
    aug_col_indices = build_augmentation_column_indices(
        n_target_features=layout.n_target_features,
        nearby_features_per_station=layout.nearby_features_per_station,
        n_nearby_available=n_nearby_available,
        n_nearby_output=n_nearby_output,
        use_skip_patterns=use_skip_patterns
    )

    n_augmentations = len(aug_col_indices)
    n_stations = len(sequences_to_predict)

    print(f"\nPhase 2: Running ensemble inference...")
    print(f"  Stations: {n_stations}")
    print(f"  Augmentations per station: {n_augmentations}")
    print(f"  Skip patterns: {'enabled' if use_skip_patterns else 'disabled (permutations only)'}")

    # Accumulate predictions for each station
    prediction_sums = np.zeros(n_stations, dtype=np.float64)
    prediction_counts = np.zeros(n_stations, dtype=np.int32)

    # Process one augmentation pattern at a time to avoid memory blowup
    for aug_idx, col_indices in enumerate(aug_col_indices):
        if aug_idx % 24 == 0:  # Report every 24 augmentations (one skip pattern worth)
            print(f"  Processing augmentation {aug_idx + 1}/{n_augmentations}...")

        # Build batch for this augmentation pattern
        batch_features = []
        for _, features_norm, _ in sequences_to_predict:
            # Select columns for this augmentation
            aug_features = features_norm[:, col_indices]
            batch_features.append(torch.from_numpy(aug_features))

        X_batch = torch.stack(batch_features)

        # Run inference
        x_gpu = torch.zeros([n_stations, model.embed_dim - 2, model.seq_length - model.n_class_tokens],
                           dtype=torch.float16, device=device)

        with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=True,
                                                     cache_enabled=True, dtype=torch.bfloat16):
            x_gpu[:n_stations, :X_batch.shape[2], :].copy_(X_batch.permute(0, 2, 1), non_blocking=True)
            predictions = model(x_gpu[:n_stations]).cpu().numpy().flatten()

        # Accumulate predictions
        prediction_sums += predictions
        prediction_counts += 1

    print(f"✓ Ensemble inference complete ({n_augmentations} augmentations per station)")

    # Average predictions and denormalize
    print(f"\nPhase 3: Averaging and denormalizing predictions...")
    predicted_results = []

    for i, (station_info, _, _) in enumerate(sequences_to_predict):
        avg_pred_normalized = prediction_sums[i] / prediction_counts[i]
        pred_denorm = denormalize_soil_moisture(
            avg_pred_normalized,
            str(collector.data_dir / "normalization_stats.npz")
        )

        predicted_results.append({
            'station_id': station_info['station_id'],
            'latitude': station_info['latitude'],
            'longitude': station_info['longitude'],
            'moisture': pred_denorm,
            'type': 'predicted',
            'name': station_info['name'],
            'n_augmentations': int(prediction_counts[i])
        })

    return predicted_results


def create_virtual_grid_stations(
    lon_min, lon_max, lat_min, lat_max,
    grid_size=100,
    galicia_land=None
):
    """
    Create a grid of virtual weather stations covering the region.
    
    Args:
        lon_min, lon_max, lat_min, lat_max: Bounding box
        grid_size: Number of points in each dimension (grid_size x grid_size)
        galicia_land: Optional shapely geometry to filter land-only points
    
    Returns:
        List of dicts with virtual station info: [{lon, lat, grid_id}, ...]
    """
    from shapely.geometry import Point
    
    # Create regular grid
    lons = np.linspace(lon_min, lon_max, grid_size)
    lats = np.linspace(lat_min, lat_max, grid_size)
    
    virtual_stations = []
    grid_id = 0
    
    for lon in lons:
        for lat in lats:
            # Filter to land only if geometry provided
            if galicia_land is not None:
                point = Point(lon, lat)
                if not galicia_land.contains(point):
                    continue
            
            virtual_stations.append({
                'grid_id': grid_id,
                'longitude': lon,
                'latitude': lat
            })
            grid_id += 1
    
    return virtual_stations


def find_nearest_real_stations(virtual_station, stations_df, stations_lookup, n_nearest=4):
    """
    Find the n nearest real stations to a virtual station.
    Uses UTM coordinates for distance calculation (meters) to match the precomputed distances.
    
    Returns list of dicts: [{station_id, distance, lon, lat}, ...]
    """
    # First, interpolate UTM coordinates for the virtual station from nearby stations
    # Use a simple approach: find closest station by lat/lon and use similar UTM
    virtual_lon = virtual_station['longitude']
    virtual_lat = virtual_station['latitude']
    
    # Get stations with valid UTM coordinates
    valid_utm = []
    for _, row in stations_df.iterrows():
        sid = int(row['station_id'])
        if sid in stations_lookup:
            coords = stations_lookup[sid]
            if coords.get('utmx') is not None and coords.get('utmy') is not None:
                valid_utm.append({
                    'station_id': sid,
                    'longitude': row['longitude'],
                    'latitude': row['latitude'],
                    'utmx': coords['utmx'],
                    'utmy': coords['utmy'],
                    'has_soil_moisture': row['has_soil_moisture']
                })
    
    if not valid_utm:
        raise ValueError(
            "No stations found with valid UTM coordinates. "
            "This indicates a data integrity issue - all stations should have coordinates."
        )

    # Compute approximate UTM for virtual station using inverse distance weighted interpolation
    # from nearest stations (by lat/lon)
    lons = np.array([s['longitude'] for s in valid_utm])
    lats = np.array([s['latitude'] for s in valid_utm])

    ll_distances = np.sqrt((lons - virtual_lon)**2 + (lats - virtual_lat)**2)
    nearest_idx = np.argsort(ll_distances)[:6]  # Use 6 nearest for UTM interpolation
    
    weights = 1.0 / (ll_distances[nearest_idx] + 1e-9)
    weights = weights / weights.sum()
    
    virtual_utmx = np.sum(weights * np.array([valid_utm[i]['utmx'] for i in nearest_idx]))
    virtual_utmy = np.sum(weights * np.array([valid_utm[i]['utmy'] for i in nearest_idx]))
    
    # Now compute distances in meters using UTM
    result = []
    for s in valid_utm:
        dist_m = np.sqrt((s['utmx'] - virtual_utmx)**2 + (s['utmy'] - virtual_utmy)**2)
        result.append({
            'station_id': s['station_id'],
            'distance': dist_m,  # Distance in meters
            'longitude': s['longitude'],
            'latitude': s['latitude'],
            'has_soil_moisture': s['has_soil_moisture']
        })
    
    # Sort by distance and return n nearest
    result.sort(key=lambda x: x['distance'])
    return result[:n_nearest]


def interpolate_coordinate_features(virtual_station, nearest_stations, stations_lookup, use_real_elevation=True):
    """
    Interpolate coordinate features (altitude, utmx, utmy) from nearest stations.
    Uses inverse distance weighting based on the distances already computed (in meters).
    
    For altitude, attempts to use real SRTM elevation data if available.
    Falls back to interpolation from nearby stations if SRTM data not available.
    
    Returns dict: {altitude, utmx, utmy}
    """
    result = {'altitude': None, 'utmx': None, 'utmy': None}
    
    # Try to get real elevation from SRTM data first
    if use_real_elevation:
        real_elevation = get_elevation_from_srtm(
            virtual_station['longitude'], 
            virtual_station['latitude']
        )
        if real_elevation is not None:
            result['altitude'] = real_elevation
    
    for coord in ['altitude', 'utmx', 'utmy']:
        # Skip altitude if we already got it from SRTM
        if coord == 'altitude' and result['altitude'] is not None:
            continue
            
        values = []
        weights = []
        
        for ns in nearest_stations:
            sid = ns['station_id']
            if sid in stations_lookup and stations_lookup[sid].get(coord) is not None:
                values.append(stations_lookup[sid][coord])
                # Use the distance already computed (now in meters)
                weights.append(1.0 / (ns['distance'] + 1.0))  # +1m to avoid div by zero
        
        if values:
            weights = np.array(weights)
            weights = weights / weights.sum()
            result[coord] = float(np.sum(np.array(values) * weights))
    
    return result


# Global elevation data cache
_elevation_data = None
_elevation_transform = None
_elevation_bounds = None


def load_srtm_elevation_data(cache_dir=None):
    """
    Load SRTM elevation data for Galicia region.
    Downloads and caches the data locally.
    
    Uses the 'elevation' package to download SRTM 30m data.
    
    Returns (data, transform, bounds) or (None, None, None) if unavailable.
    """
    global _elevation_data, _elevation_transform, _elevation_bounds
    
    # Return cached data if already loaded in memory
    if _elevation_data is not None:
        return _elevation_data, _elevation_transform, _elevation_bounds
    
    import os
    
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(__file__), '.geo_cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    elevation_cache = os.path.join(cache_dir, 'galicia_elevation.tif')
    
    try:
        import rasterio
        
        if os.path.exists(elevation_cache):
            print("    Using cached elevation data...")
            with rasterio.open(elevation_cache) as src:
                _elevation_data = src.read(1)
                _elevation_transform = src.transform
                _elevation_bounds = src.bounds
            return _elevation_data, _elevation_transform, _elevation_bounds
        
        # Try to download SRTM data using the 'elevation' package
        print("    Downloading SRTM elevation data (first run only)...")
        
        try:
            import elevation
            
            # Define bounds for Galicia (west, south, east, north)
            bounds = (-9.5, 41.7, -6.5, 44.0)
            
            # Download and clip SRTM data
            elevation.clip(bounds=bounds, output=elevation_cache, product='SRTM3')
            
            with rasterio.open(elevation_cache) as src:
                _elevation_data = src.read(1)
                _elevation_transform = src.transform
                _elevation_bounds = src.bounds
            
            print(f"    ✓ Downloaded and cached SRTM elevation data")
            return _elevation_data, _elevation_transform, _elevation_bounds
            
        except ImportError:
            print("    'elevation' package not installed.")
            print("    To enable real elevation data, install: pip install elevation")
            return None, None, None
            
    except ImportError:
        print("    'rasterio' not installed.")
        print("    To enable real elevation data, install: pip install rasterio elevation")
        return None, None, None
    except Exception as e:
        print(f"    Could not load elevation data: {e}")
        return None, None, None


def get_elevation_from_srtm(lon, lat, cache_dir=None):
    """
    Get elevation for a point from cached SRTM data.
    
    Returns elevation in meters, or None if data unavailable.
    """
    data, transform, bounds = load_srtm_elevation_data(cache_dir)
    
    if data is None:
        return None

    from rasterio.transform import rowcol

    # Check if point is within bounds
    if not (bounds.left <= lon <= bounds.right and bounds.bottom <= lat <= bounds.top):
        return None

    # Convert lon/lat to row/col
    row, col = rowcol(transform, lon, lat)

    # Check bounds
    if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
        elev = data[row, col]
        # SRTM uses -32768 as nodata
        if elev > -1000:
            return float(elev)

    return None


def point_in_triangle(px, py, ax, ay, bx, by, cx, cy):
    """Check if point (px, py) is inside triangle ABC using barycentric coordinates."""
    def sign(p1x, p1y, p2x, p2y, p3x, p3y):
        return (p1x - p3x) * (p2y - p3y) - (p2x - p3x) * (p1y - p3y)
    
    d1 = sign(px, py, ax, ay, bx, by)
    d2 = sign(px, py, bx, by, cx, cy)
    d3 = sign(px, py, cx, cy, ax, ay)
    
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    
    return not (has_neg and has_pos)


def triangle_area(ax, ay, bx, by, cx, cy):
    """Compute area of triangle ABC."""
    return abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)) / 2.0


def select_triangle_stations(virtual_station, nearest_stations, stations_lookup):
    """
    Select 3 stations for interpolation:
    - First station is always the nearest
    - Other two are chosen from the provided nearest stations to form the smallest triangle
      that contains the virtual station
    - If no containing triangle exists, fall back to 3 nearest
    
    Returns list of 3 station dicts with 'station_id' and 'distance'
    """
    if len(nearest_stations) < 3:
        return nearest_stations
    
    # Get virtual station UTM coordinates (approximate from nearest)
    v_lon = virtual_station['longitude']
    v_lat = virtual_station['latitude']
    
    # Get UTM coordinates for candidate stations (use all provided)
    candidates = []
    for ns in nearest_stations:
        sid = ns['station_id']
        if sid in stations_lookup:
            coords = stations_lookup[sid]
            if coords.get('utmx') is not None and coords.get('utmy') is not None:
                candidates.append({
                    'station_id': sid,
                    'distance': ns['distance'],
                    'utmx': coords['utmx'],
                    'utmy': coords['utmy']
                })
    
    if len(candidates) < 3:
        return nearest_stations[:3]
    
    # Compute approximate UTM for virtual station using weighted average of nearest
    weights = np.array([1.0 / (c['distance'] + 1.0) for c in candidates[:3]])
    weights = weights / weights.sum()
    v_utmx = sum(w * c['utmx'] for w, c in zip(weights, candidates[:3]))
    v_utmy = sum(w * c['utmy'] for w, c in zip(weights, candidates[:3]))
    
    # First station is always nearest
    first = candidates[0]
    
    # Find smallest triangle containing the virtual station
    best_triangle = None
    best_area = float('inf')
    
    for i in range(1, len(candidates)):
        for j in range(i + 1, len(candidates)):
            # Triangle: first, candidates[i], candidates[j]
            ax, ay = first['utmx'], first['utmy']
            bx, by = candidates[i]['utmx'], candidates[i]['utmy']
            cx, cy = candidates[j]['utmx'], candidates[j]['utmy']
            
            if point_in_triangle(v_utmx, v_utmy, ax, ay, bx, by, cx, cy):
                area = triangle_area(ax, ay, bx, by, cx, cy)
                if area < best_area:
                    best_area = area
                    best_triangle = [first, candidates[i], candidates[j]]
    
    if best_triangle is not None:
        return best_triangle
    else:
        # No containing triangle found - use 3 nearest
        return [{'station_id': c['station_id'], 'distance': c['distance']} 
                for c in candidates[:3]]


def build_sequence_for_virtual_station(
    virtual_station,
    end_date,
    timeseries_lookup,
    nearest_real_stations,  # List from find_nearest_real_stations
    nearest_with_soil,  # List of nearest stations WITH soil moisture sensors
    virtual_coords,  # Interpolated {altitude, utmx, utmy}
    stations_lookup,
    feature_params,
    norm_stats,
    seq_length=64,
    n_nearest=4,
    missing_value=-1000.0
):
    """
    Build a sequence for a virtual grid station by interpolating features from real stations.

    For each feature at each timestep:
    - Use inverse distance weighted interpolation from nearest real stations
    - Missing values (-9999, -1000) are excluded from interpolation
    """

    # Initialize arrays and layout
    init = _init_sequence_arrays(end_date, seq_length, feature_params, n_nearest, missing_value)
    features = init['features']
    mask = init['mask']
    date_strings = init['date_strings']
    coordinate_features = init['coordinate_features']

    # Fill coordinate features for target (virtual) station - constant across time
    _fill_target_coordinate_features(features, mask, virtual_coords,
                                     feature_params, coordinate_features)

    # Fill nearby station (with soil moisture) coordinate features and distances
    _fill_nearby_coordinates_and_distances(
        features, mask, nearest_with_soil, stations_lookup,
        feature_params, coordinate_features,
        init['target_features_per_timestep'], init['nearby_features_per_timestep'],
        n_nearest
    )

    # Select which stations to use for interpolation (triangle selection)
    # Uses 3 stations forming the smallest triangle containing the virtual station
    interpolation_stations = select_triangle_stations(
        virtual_station, nearest_real_stations, stations_lookup
    )

    # Fill target station timeseries by interpolation from selected stations
    _fill_target_timeseries_interpolated(
        features, mask, date_strings, interpolation_stations,
        timeseries_lookup, feature_params, coordinate_features
    )

    # Fill nearby stations (with soil moisture) timeseries features
    # Context stations use their ACTUAL data (not interpolated)
    _fill_nearby_timeseries_features(
        features, mask, date_strings, nearest_with_soil,
        timeseries_lookup, feature_params, coordinate_features,
        init['target_features_per_timestep'], init['nearby_features_per_timestep'],
        n_nearest
    )

    # Apply normalization
    features_normalized = apply_normalization_to_features(features, mask, norm_stats, missing_value)

    return features_normalized, mask


def find_nearest_stations_with_soil_moisture(virtual_station, stations_df, stations_lookup, n_max=10):
    """
    Find the nearest stations that have soil moisture sensors.
    Uses UTM coordinates for distance calculation (meters) to match the precomputed distances.
    """
    virtual_lon = virtual_station['longitude']
    virtual_lat = virtual_station['latitude']
    
    # Get soil moisture stations with valid UTM coordinates
    valid_utm = []
    for _, row in stations_df.iterrows():
        if not row['has_soil_moisture']:
            continue
        sid = int(row['station_id'])
        if sid in stations_lookup:
            coords = stations_lookup[sid]
            if coords.get('utmx') is not None and coords.get('utmy') is not None:
                valid_utm.append({
                    'station_id': sid,
                    'longitude': row['longitude'],
                    'latitude': row['latitude'],
                    'utmx': coords['utmx'],
                    'utmy': coords['utmy']
                })

    if not valid_utm:
        raise ValueError(
            "No soil moisture stations found with valid UTM coordinates. "
            "This indicates a data integrity issue - soil moisture stations should have coordinates."
        )

    # Compute approximate UTM for virtual station using inverse distance weighted interpolation
    lons = np.array([s['longitude'] for s in valid_utm])
    lats = np.array([s['latitude'] for s in valid_utm])
    
    ll_distances = np.sqrt((lons - virtual_lon)**2 + (lats - virtual_lat)**2)
    nearest_idx = np.argsort(ll_distances)[:6]
    
    weights = 1.0 / (ll_distances[nearest_idx] + 1e-9)
    weights = weights / weights.sum()
    
    virtual_utmx = np.sum(weights * np.array([valid_utm[i]['utmx'] for i in nearest_idx]))
    virtual_utmy = np.sum(weights * np.array([valid_utm[i]['utmy'] for i in nearest_idx]))
    
    # Compute distances in meters using UTM
    result = []
    for s in valid_utm:
        dist_m = np.sqrt((s['utmx'] - virtual_utmx)**2 + (s['utmy'] - virtual_utmy)**2)
        result.append({
            'station_id': s['station_id'],
            'distance': dist_m,  # Distance in meters
            'longitude': s['longitude'],
            'latitude': s['latitude'],
            'has_soil_moisture': True
        })
    
    # Sort by distance and return n nearest
    result.sort(key=lambda x: x['distance'])
    return result[:n_max]


def debug_find_worst_offenders(
    virtual_sequences,  # List of (station_info, features_norm, mask) for virtual stations
    predicted_sequences,  # List of (station_info, features_norm, mask) for predicted stations
    virtual_results,  # Results with moisture predictions
    predicted_results,  # Results with moisture predictions
    stations_df,
    stations_lookup,
    nearest_lookup,
    feature_params,
    norm_stats,
    n_nearby=4,
    top_n=5
):
    """
    Find virtual stations that are close to predicted (triangle) stations but have
    very different moisture predictions. Compare the ACTUAL NORMALIZED FEATURES
    that were sent to the model.
    """
    if not virtual_results or not predicted_results:
        print("No virtual or predicted results to compare")
        return
    
    print("\n" + "=" * 80)
    print("DEBUG: Finding worst offenders (virtual vs predicted station discrepancies)")
    print("=" * 80)
    
    # Build lookup from station_id/grid_id to sequence data
    virtual_seq_lookup = {}
    for station_info, features_norm, mask in virtual_sequences:
        key = (station_info['longitude'], station_info['latitude'])
        virtual_seq_lookup[key] = (features_norm, mask, station_info)
    
    pred_seq_lookup = {}
    for station_info, features_norm, mask in predicted_sequences:
        sid = station_info['station_id']
        pred_seq_lookup[sid] = (features_norm, mask, station_info)
    
    # Build arrays for fast distance computation
    virtual_coords = np.array([[v['longitude'], v['latitude']] for v in virtual_results])
    virtual_moisture = np.array([v['moisture'] for v in virtual_results])
    
    pred_coords = np.array([[p['longitude'], p['latitude']] for p in predicted_results])
    pred_moisture = np.array([p['moisture'] for p in predicted_results])
    pred_ids = [p['station_id'] for p in predicted_results]

    offenders = []
    for i, vr in enumerate(virtual_results):
        # Find nearest predicted station
        dists_deg = np.sqrt(np.sum((pred_coords - virtual_coords[i])**2, axis=1))
        nearest_idx = np.argmin(dists_deg)
        dist_km = dists_deg[nearest_idx] * DEG_TO_KM

        # Only consider if within threshold distance
        if dist_km > OFFENDER_DISTANCE_KM:
            continue
        
        moisture_diff = abs(virtual_moisture[i] - pred_moisture[nearest_idx])
        score = moisture_diff / (dist_km + 0.1)
        
        offenders.append({
            'virtual_idx': i,
            'virtual_result': vr,
            'pred_idx': nearest_idx,
            'pred_result': predicted_results[nearest_idx],
            'pred_station_id': pred_ids[nearest_idx],
            'dist_km': dist_km,
            'moisture_diff': moisture_diff,
            'score': score
        })
    
    offenders.sort(key=lambda x: -x['score'])
    
    print(f"\nFound {len(offenders)} virtual stations within 1km of a predicted station")
    if not offenders:
        print("No offenders found!")
        return
        
    print(f"Showing top {min(top_n, len(offenders))} worst offenders:\n")

    # Feature structure using FeatureLayout
    layout = FeatureLayout(n_params=len(feature_params), n_nearby=n_nearby)
    n_target_features = layout.n_target_features
    nearby_features_per_station = layout.nearby_features_per_station
    
    for rank, off in enumerate(offenders[:top_n]):
        vr = off['virtual_result']
        pr = off['pred_result']
        pred_sid = off['pred_station_id']
        
        print(f"\n{'='*70}")
        print(f"OFFENDER #{rank+1} (score: {off['score']:.3f})")
        print(f"{'='*70}")
        print(f"Virtual station: ({vr['longitude']:.4f}, {vr['latitude']:.4f})")
        print(f"Predicted station {pred_sid}: ({pr['longitude']:.4f}, {pr['latitude']:.4f})")
        print(f"Distance between them: {off['dist_km']*1000:.1f}m")
        print(f"\nMOISTURE PREDICTION:")
        print(f"  Virtual:   {vr['moisture']:.4f}")
        print(f"  Predicted: {pr['moisture']:.4f}")
        print(f"  Difference: {off['moisture_diff']:.4f}")
        
        # Get actual normalized sequences
        vkey = (vr['longitude'], vr['latitude'])
        if vkey not in virtual_seq_lookup:
            print(f"  WARNING: Could not find virtual sequence for {vkey}")
            continue
        v_features, v_mask, _ = virtual_seq_lookup[vkey]
        
        if pred_sid not in pred_seq_lookup:
            print(f"  WARNING: Could not find predicted sequence for station {pred_sid}")
            continue
        p_features, p_mask, _ = pred_seq_lookup[pred_sid]
        
        # Compare the ACTUAL normalized features (last timestep, index -1)
        print(f"\n--- ACTUAL NORMALIZED FEATURES SENT TO MODEL (last timestep) ---")
        print(f"  Comparing what the model actually saw for each station")
        
        # Compute differences for target station features
        print(f"\n  TARGET STATION FEATURES (first {n_target_features} features):")
        print(f"  {'Idx':>4s}  {'Parameter':25s}  {'Pred':>10s}  {'Virtual':>10s}  {'Diff':>10s}")
        print(f"  {'-'*4}  {'-'*25}  {'-'*10}  {'-'*10}  {'-'*10}")
        
        target_diffs = []
        for f_idx in range(n_target_features):
            param = feature_params[f_idx] if f_idx < len(feature_params) else f"feat_{f_idx}"
            p_val = p_features[-1, f_idx]  # Last timestep
            v_val = v_features[-1, f_idx]
            diff = abs(p_val - v_val)
            target_diffs.append((f_idx, param, p_val, v_val, diff))
        
        # Sort by diff and show top 10
        target_diffs.sort(key=lambda x: -x[4])
        for f_idx, param, p_val, v_val, diff in target_diffs[:10]:
            print(f"  {f_idx:4d}  {param:25s}  {p_val:10.4f}  {v_val:10.4f}  {diff:10.4f}")
        
        # Compare context station features (nearby stations with soil moisture)
        print(f"\n  CONTEXT STATION FEATURES ({n_nearby} nearest soil moisture stations):")

        for n_idx in range(n_nearby):
            offset = n_target_features + n_idx * nearby_features_per_station
            
            # Distance feature
            dist_idx = offset
            p_dist = p_features[-1, dist_idx]
            v_dist = v_features[-1, dist_idx]
            
            print(f"\n  Context station {n_idx+1}:")
            print(f"    Distance (normalized): Pred={p_dist:.4f}, Virtual={v_dist:.4f}, Diff={abs(p_dist-v_dist):.4f}")
            
            # Soil moisture feature (last in the group)
            soil_idx = offset + 1 + len(feature_params)
            p_soil = p_features[-1, soil_idx]
            v_soil = v_features[-1, soil_idx]
            print(f"    Soil moisture (norm):  Pred={p_soil:.4f}, Virtual={v_soil:.4f}, Diff={abs(p_soil-v_soil):.4f}")
            
            # Check a few weather features for this context station
            context_diffs = []
            for f_idx_rel, param in enumerate(feature_params[:5]):  # First 5 params
                feat_idx = offset + 1 + f_idx_rel
                p_val = p_features[-1, feat_idx]
                v_val = v_features[-1, feat_idx]
                context_diffs.append((param, p_val, v_val, abs(p_val - v_val)))
            
            context_diffs.sort(key=lambda x: -x[3])
            if context_diffs[0][3] > 0.01:  # Only show if there's meaningful difference
                print(f"    Largest feature diffs:")
                for param, p_val, v_val, diff in context_diffs[:3]:
                    print(f"      {param}: Pred={p_val:.4f}, Virt={v_val:.4f}, Diff={diff:.4f}")
        
        # Overall statistics
        all_diffs = np.abs(p_features[-1, :] - v_features[-1, :])
        print(f"\n  SUMMARY:")
        print(f"    Total features: {len(all_diffs)}")
        print(f"    Max diff: {np.max(all_diffs):.4f}")
        print(f"    Mean diff: {np.mean(all_diffs):.4f}")
        print(f"    Features with diff > 0.1: {np.sum(all_diffs > 0.1)}")
        print(f"    Features with diff > 0.5: {np.sum(all_diffs > 0.5)}")
    
    print("\n" + "=" * 80)
    print("END DEBUG")
    print("=" * 80)


def _load_map_data(collector, model_path, target_date, device, n_nearby, n_nearby_available, augmented):
    """
    Load all data files, model, and build lookup dictionaries.

    Returns:
        dict with keys: stations_df, timeseries_df, nearest_df, stations_lookup,
                        nearest_lookup, timeseries_lookup, norm_stats, model,
                        filtered_params, coastline_points, galicia_land, target_date
    """
    # Load all data files ONCE at the start
    print("\nLoading data files...")
    stations_df = pd.read_csv(collector.stations_file)
    nearest_df = pd.read_csv(collector.nearest_file)

    # Load timeseries ONCE with optimized dtypes
    print("  Loading timeseries (this may take a moment)...")
    timeseries_df = pd.read_csv(
        collector.timeseries_file,
        dtype={'station_id': np.int32, 'value': np.float32}
    )
    timeseries_df['date'] = pd.to_datetime(timeseries_df['date'])
    print(f"  ✓ Loaded {len(timeseries_df):,} timeseries records")

    print(f"\nFound {len(stations_df)} stations total")
    print(f"  - {stations_df['has_soil_moisture'].sum()} with soil moisture sensors")
    print(f"  - {(~stations_df['has_soil_moisture']).sum()} without sensors (will predict)")

    # Load coastline data early (fail fast before expensive model inference)
    print("\nPreparing coastline data...")
    lon_min, lon_max = stations_df['longitude'].min(), stations_df['longitude'].max()
    lat_min, lat_max = stations_df['latitude'].min(), stations_df['latitude'].max()
    coastline_points, galicia_land = load_coastline_data(lon_min, lon_max, lat_min, lat_max)

    # Determine target date using already-loaded timeseries
    if target_date is None:
        target_date = timeseries_df['date'].max()
    else:
        target_date = pd.to_datetime(target_date)

    print(f"\nTarget date: {target_date.strftime('%Y-%m-%d')}")

    # Load model
    model = load_model(model_path, device, compilation=False)

    # Analyze parameter coverage
    print("\nAnalyzing parameter coverage...")
    _, filtered_params = collector.analyze_parameter_coverage(
        timeseries_df=timeseries_df,
        stations_df=stations_df,
        coverage_threshold=0.25
    )

    # Load canonical normalization stats and expand to augmented layout
    n_params = len(filtered_params)
    canonical_stats_path = str(collector.data_dir / "normalization_stats.npz")
    canonical_stats = np.load(canonical_stats_path)

    if 'target_feature_mins' in canonical_stats:
        aug_str = f"augmented (n_nearby_available={n_nearby_available})" if augmented else "non-augmented"
        print(f"  Using canonical stats, expanding to {n_nearby}-nearby {aug_str} layout...")
        norm_stats = expand_canonical_to_augmented_stats(
            canonical_stats, n_params, n_nearby,
            n_nearby_available=n_nearby_available, augmented=augmented
        )
    else:
        print(f"  Using legacy augmented stats format...")
        norm_stats = canonical_stats

    # Build lookup dictionaries
    print("\nBuilding fast timeseries lookup...")
    start_date = target_date - timedelta(days=64 - 1)

    # Build nearest lookup dict once
    nearest_lookup = {}
    for _, row in nearest_df.iterrows():
        station_id = int(row['station_id'])
        nearby_with_soil = []
        for i in range(1, 50):
            if f'nearest_{i}_id' not in row:
                break
            if row.get(f'nearest_{i}_has_soil_moisture', False):
                nearby_with_soil.append({
                    'station_id': int(row[f'nearest_{i}_id']),
                    'distance': float(row[f'nearest_{i}_distance'])
                })
        nearest_lookup[station_id] = nearby_with_soil

    # Build stations lookup dict once
    stations_lookup = {}
    for _, row in stations_df.iterrows():
        sid = int(row['station_id'])
        stations_lookup[sid] = {
            'altitude': float(row['altitude']) if pd.notna(row['altitude']) else None,
            'utmx': float(row['utmx']) if pd.notna(row['utmx']) else None,
            'utmy': float(row['utmy']) if pd.notna(row['utmy']) else None
        }

    # Collect all needed stations
    all_needed_stations = set(stations_df['station_id'].tolist())
    for station_id in stations_df[~stations_df['has_soil_moisture']]['station_id']:
        if station_id in nearest_lookup:
            for nearby in nearest_lookup[station_id]:
                all_needed_stations.add(nearby['station_id'])

    timeseries_lookup = build_fast_timeseries_lookup(
        timeseries_df, start_date, target_date,
        list(all_needed_stations), filtered_params
    )
    print(f"  ✓ Built lookup with {len(timeseries_lookup)} entries for {len(all_needed_stations)} stations")

    return {
        'stations_df': stations_df,
        'timeseries_df': timeseries_df,
        'nearest_df': nearest_df,
        'stations_lookup': stations_lookup,
        'nearest_lookup': nearest_lookup,
        'timeseries_lookup': timeseries_lookup,
        'norm_stats': norm_stats,
        'model': model,
        'filtered_params': filtered_params,
        'coastline_points': coastline_points,
        'galicia_land': galicia_land,
        'target_date': target_date,
    }


def _collect_real_and_build_sequences(stations_df, timeseries_lookup, nearest_lookup,
                                       stations_lookup, filtered_params, norm_stats,
                                       target_date, n_nearby, real_moisture_only,
                                       ensemble_mode=False, n_nearby_available=5):
    """
    Phase 1: Collect real moisture data and build sequences for stations needing prediction.

    Args:
        ensemble_mode: If True, build sequences with extra nearby stations for ensemble prediction
        n_nearby_available: Number of nearby stations to include when ensemble_mode=True

    Returns:
        tuple: (real_results, sequences_to_predict)
    """
    print("\nPhase 1: Gathering real data and building sequences...")
    if ensemble_mode:
        print(f"  Ensemble mode: building with {n_nearby_available} nearby stations")
    real_results = []
    sequences_to_predict = []

    for idx, station in stations_df.iterrows():
        if idx % 20 == 0:
            print(f"  Processing station {idx+1}/{len(stations_df)}...")

        station_id = station['station_id']
        lat = station['latitude']
        lon = station['longitude']
        has_sensor = station['has_soil_moisture']

        if has_sensor:
            moisture = get_real_soil_moisture_from_lookup(timeseries_lookup, station_id, target_date)
            if moisture is not None:
                real_results.append({
                    'station_id': station_id,
                    'latitude': lat,
                    'longitude': lon,
                    'moisture': moisture,
                    'type': 'real',
                    'name': station.get('name', f'Station {station_id}')
                })
        elif not real_moisture_only:
            if ensemble_mode:
                sequence_data = build_sequence_for_ensemble(
                    station_id=station_id,
                    end_date=target_date,
                    timeseries_lookup=timeseries_lookup,
                    nearest_lookup=nearest_lookup,
                    stations_lookup=stations_lookup,
                    feature_params=filtered_params,
                    norm_stats=norm_stats,
                    seq_length=64,
                    n_nearby_output=n_nearby,
                    n_nearby_available=n_nearby_available
                )
            else:
                sequence_data = build_sequence_for_any_station(
                    station_id=station_id,
                    end_date=target_date,
                    timeseries_lookup=timeseries_lookup,
                    nearest_lookup=nearest_lookup,
                    stations_lookup=stations_lookup,
                    feature_params=filtered_params,
                    norm_stats=norm_stats,
                    seq_length=64,
                    n_nearest=n_nearby
                )

            if sequence_data is not None:
                features_norm, mask = sequence_data
                station_info = {
                    'station_id': station_id,
                    'latitude': lat,
                    'longitude': lon,
                    'name': station.get('name', f'Station {station_id}')
                }
                sequences_to_predict.append((station_info, features_norm, mask))

    print(f"✓ Phase 1 complete: {len(real_results)} real, {len(sequences_to_predict)} to predict")
    return real_results, sequences_to_predict


def _run_batch_inference(sequences_to_predict, model, device, collector):
    """
    Phase 2-3: Run batched inference and denormalize predictions.

    Returns:
        list: predicted_results
    """
    predicted_results = []
    if not sequences_to_predict:
        return predicted_results

    print(f"\nPhase 2: Running batched inference for {len(sequences_to_predict)} stations...")

    batch_size = len(sequences_to_predict)

    X_batch = torch.stack([
        torch.from_numpy(features_norm) for _, features_norm, _ in sequences_to_predict
    ])

    print(f"  Batch shape: {X_batch.shape}")

    x_gpu = torch.zeros([batch_size, model.embed_dim - 2, model.seq_length - model.n_class_tokens],
                       dtype=torch.float16, device=device)
    torch._dynamo.config.disable = True
    with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=True, cache_enabled=True, dtype=torch.bfloat16):
        x_gpu[:batch_size, :X_batch.shape[2], :].copy_(X_batch.permute(0, 2, 1), non_blocking=True)
        x = x_gpu[:batch_size, :, :]
        predictions_normalized = model(x).cpu()

    print(f"✓ Inference complete")

    # Phase 3: Denormalize and assemble results
    print(f"\nPhase 3: Denormalizing predictions...")
    for i, (station_info, _, _) in enumerate(sequences_to_predict):
        pred_normalized = predictions_normalized[i].item()
        pred_denorm = denormalize_soil_moisture(
            pred_normalized,
            str(collector.data_dir / "normalization_stats.npz")
        )

        predicted_results.append({
            'station_id': station_info['station_id'],
            'latitude': station_info['latitude'],
            'longitude': station_info['longitude'],
            'moisture': pred_denorm,
            'type': 'predicted',
            'name': station_info['name']
        })

    return predicted_results


def _create_virtual_grid_predictions(virtual_grid_size, galicia_land, stations_df, stations_lookup,
                                      timeseries_lookup, filtered_params, norm_stats, model,
                                      device, collector, target_date, n_nearby,
                                      sequences_to_predict, predicted_results):
    """
    Phase 4: Create virtual grid and run predictions.

    Returns:
        tuple: (virtual_results, virtual_sequences)
    """
    virtual_results = []
    virtual_sequences = []

    if virtual_grid_size is None or virtual_grid_size <= 0:
        return virtual_results, virtual_sequences

    lon_min, lon_max = stations_df['longitude'].min(), stations_df['longitude'].max()
    lat_min, lat_max = stations_df['latitude'].min(), stations_df['latitude'].max()

    print(f"\n" + "=" * 60)
    print(f"Phase 4: Creating {virtual_grid_size}x{virtual_grid_size} virtual grid...")
    print("=" * 60)

    virtual_stations = create_virtual_grid_stations(
        lon_min, lon_max, lat_min, lat_max,
        grid_size=virtual_grid_size,
        galicia_land=galicia_land
    )
    print(f"  Created {len(virtual_stations)} virtual stations (land only)")

    print(f"\n  Building sequences for virtual stations...")

    for i, vs in enumerate(virtual_stations):
        if i % 500 == 0:
            print(f"    Processing virtual station {i+1}/{len(virtual_stations)}...")

        nearest_real = find_nearest_real_stations(vs, stations_df, stations_lookup, n_nearest=5)
        nearest_soil = find_nearest_stations_with_soil_moisture(vs, stations_df, stations_lookup, n_max=10)

        if len(nearest_soil) < 4:
            continue

        virtual_coords_interp = interpolate_coordinate_features(vs, nearest_real, stations_lookup)

        sequence_data = build_sequence_for_virtual_station(
            virtual_station=vs,
            end_date=target_date,
            timeseries_lookup=timeseries_lookup,
            nearest_real_stations=nearest_real,
            nearest_with_soil=nearest_soil,
            virtual_coords=virtual_coords_interp,
            stations_lookup=stations_lookup,
            feature_params=filtered_params,
            norm_stats=norm_stats,
            seq_length=64,
            n_nearest=n_nearby
        )

        if sequence_data is not None:
            features_norm, mask = sequence_data
            station_info = {
                'grid_id': vs['grid_id'],
                'latitude': vs['latitude'],
                'longitude': vs['longitude']
            }
            virtual_sequences.append((station_info, features_norm, mask))

    print(f"  ✓ Built {len(virtual_sequences)} valid virtual station sequences")

    # Run batch inference for virtual stations
    if virtual_sequences:
        print(f"\n  Running batched inference for virtual grid...")

        batch_size = len(virtual_sequences)
        X_batch = torch.stack([
            torch.from_numpy(features_norm) for _, features_norm, _ in virtual_sequences
        ])

        print(f"    Batch shape: {X_batch.shape}")

        x_gpu = torch.zeros([batch_size, model.embed_dim - 2, model.seq_length - model.n_class_tokens],
                           dtype=torch.float16, device=device)

        with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=True, cache_enabled=True, dtype=torch.bfloat16):
            x_gpu[:batch_size, :X_batch.shape[2], :].copy_(X_batch.permute(0, 2, 1), non_blocking=True)
            x = x_gpu[:batch_size, :, :]
            predictions_normalized = model(x).cpu()

        print(f"  ✓ Virtual grid inference complete")

        for i, (station_info, _, _) in enumerate(virtual_sequences):
            pred_normalized = predictions_normalized[i].item()
            pred_denorm = denormalize_soil_moisture(
                pred_normalized,
                str(collector.data_dir / "normalization_stats.npz")
            )

            virtual_results.append({
                'station_id': f"grid_{station_info['grid_id']}",
                'latitude': station_info['latitude'],
                'longitude': station_info['longitude'],
                'moisture': pred_denorm,
                'type': 'virtual',
                'name': f"Virtual {station_info['grid_id']}"
            })

        print(f"  ✓ {len(virtual_results)} virtual grid predictions complete")

    # Debug: find worst offenders
    debug_find_worst_offenders(
        virtual_sequences=virtual_sequences,
        predicted_sequences=sequences_to_predict,
        virtual_results=virtual_results,
        predicted_results=predicted_results,
        stations_df=stations_df,
        stations_lookup=stations_lookup,
        nearest_lookup={},  # Not used in debug
        feature_params=filtered_params,
        norm_stats=norm_stats,
        n_nearby=n_nearby,
        top_n=5
    )

    return virtual_results, virtual_sequences


def _generate_output_maps(real_results, predicted_results, virtual_results,
                          coastline_points, galicia_land, target_date, output_file,
                          moisture_range, hide_markers, all_maps, include_weather_maps,
                          stations_df, timeseries_lookup):
    """
    Generate all output map files.

    Returns:
        pd.DataFrame: Combined results
    """
    all_results = real_results + predicted_results + virtual_results
    results_df = pd.DataFrame(all_results)
    print(f"\n✓ Collected data for {len(results_df)} stations")
    print(f"  - Real: {(results_df['type'] == 'real').sum()}")
    print(f"  - Predicted: {(results_df['type'] == 'predicted').sum()}")
    if virtual_results:
        print(f"  - Virtual grid: {(results_df['type'] == 'virtual').sum()}")

    # Compute output filenames
    if output_file.endswith('_moisture_map.png'):
        base_name = output_file[:-len('_moisture_map.png')]
    else:
        base_name = output_file.replace('.png', '')

    if all_maps:
        print("\n" + "=" * 60)
        print("CREATING ALL MOISTURE MAP VARIANTS")
        print("=" * 60)

        # 1. Full map
        full_file = f"{base_name}_moisture_map.png"
        print(f"\nCreating full moisture map (real + predicted + virtual)...")
        create_visualization(
            results_df, target_date, full_file,
            coastline_points, galicia_land,
            moisture_range=moisture_range,
            hide_markers=hide_markers
        )
        print(f"✓ Full map saved to {full_file}")

        # 2. No virtual map
        novirtual_file = f"{base_name}_moisture_map_novirtual.png"
        print(f"\nCreating no-virtual moisture map (real + predicted)...")
        novirtual_results = real_results + predicted_results
        novirtual_df = pd.DataFrame(novirtual_results)
        create_visualization(
            novirtual_df, target_date, novirtual_file,
            coastline_points, galicia_land,
            moisture_range=moisture_range,
            hide_markers=hide_markers
        )
        print(f"✓ No-virtual map saved to {novirtual_file}")

        # 3. Real only map
        realonly_file = f"{base_name}_moisture_map_realonly.png"
        print(f"\nCreating real-only moisture map (sensors only)...")
        realonly_df = pd.DataFrame(real_results)
        create_visualization(
            realonly_df, target_date, realonly_file,
            coastline_points, galicia_land,
            moisture_range=moisture_range,
            hide_markers=hide_markers
        )
        print(f"✓ Real-only map saved to {realonly_file}")

        # Weather maps
        print("\n" + "=" * 60)
        print("CREATING CUMULATIVE WEATHER MAPS")
        print("=" * 60)

        start_date = target_date - timedelta(days=63)
        weather_results = compute_cumulative_weather(
            stations_df, timeseries_lookup, start_date, target_date
        )

        if not weather_results.empty:
            precip_file = f"{base_name}_precipitation.png"
            print(f"\nCreating cumulative precipitation map...")
            create_weather_visualization(
                weather_results[weather_results['precipitation'].notna()],
                'precipitation',
                target_date,
                precip_file,
                coastline_points,
                galicia_land,
                title=f"Cumulative Precipitation (64 days)\n{start_date.date()} to {target_date.date()}",
                unit="mm"
            )
            print(f"✓ Precipitation map saved to {precip_file}")

            balance_file = f"{base_name}_water_balance.png"
            print(f"\nCreating cumulative water balance map...")
            create_weather_visualization(
                weather_results[weather_results['water_balance'].notna()],
                'water_balance',
                target_date,
                balance_file,
                coastline_points,
                galicia_land,
                title=f"Cumulative Water Balance (64 days)\n{start_date.date()} to {target_date.date()}",
                unit="mm"
            )
            print(f"✓ Water balance map saved to {balance_file}")
        else:
            print("⚠ No weather data available for the selected date range")

    else:
        # Original single-map behavior
        print(f"\nCreating visualization...")
        create_visualization(
            results_df, target_date, output_file,
            coastline_points, galicia_land,
            moisture_range=moisture_range,
            hide_markers=hide_markers
        )
        print(f"\n✓ Map saved to {output_file}")

        if include_weather_maps:
            print("\n" + "=" * 60)
            print("CREATING CUMULATIVE WEATHER MAPS")
            print("=" * 60)

            start_date = target_date - timedelta(days=63)
            weather_results = compute_cumulative_weather(
                stations_df, timeseries_lookup, start_date, target_date
            )

            if not weather_results.empty:
                precip_file = output_file.replace('.png', '_precipitation.png')
                print(f"\nCreating cumulative precipitation map...")
                create_weather_visualization(
                    weather_results[weather_results['precipitation'].notna()],
                    'precipitation',
                    target_date,
                    precip_file,
                    coastline_points,
                    galicia_land,
                    title=f"Cumulative Precipitation (64 days)\n{start_date.date()} to {target_date.date()}",
                    unit="mm"
                )
                print(f"✓ Precipitation map saved to {precip_file}")

                balance_file = output_file.replace('.png', '_water_balance.png')
                print(f"\nCreating cumulative water balance map...")
                create_weather_visualization(
                    weather_results[weather_results['water_balance'].notna()],
                    'water_balance',
                    target_date,
                    balance_file,
                    coastline_points,
                    galicia_land,
                    title=f"Cumulative Water Balance (64 days)\n{start_date.date()} to {target_date.date()}",
                    unit="mm"
                )
                print(f"✓ Water balance map saved to {balance_file}")
            else:
                print("⚠ No weather data available for the selected date range")

    print("\n" + "=" * 60)
    return results_df


def create_moisture_map(
    model_path,
    target_date=None,
    output_file='galicia_moisture_map.png',
    device='cuda',
    include_weather_maps=False,
    virtual_grid_size=100,  # Default to 100x100 grid
    moisture_range=(0.07, 0.40),  # Fixed range tuple or 'auto' for data-based range
    hide_markers=None,  # Set of markers to hide: {'real', 'predicted', 'virtual'}
    real_moisture_only=False,  # If True, only use real moisture stations for the map
    all_maps=False,  # If True, create all map variants efficiently (reuses data)
    n_nearby=4,  # Number of nearby stations used in the model's input
    n_nearby_available=None,  # For augmented: how many nearby stations were available for permutations
    ensemble_mode=False,  # If True, use ensemble prediction with augmentations
    ensemble_skip_patterns=True  # If True, use skip patterns + permutations; if False, permutations only
):
    """
    Create a beautiful moisture map of all Galicia

    Args:
        model_path: Path to trained model checkpoint
        target_date: Date to predict (default: most recent available)
        output_file: Path to save the visualization
        device: 'cuda' or 'cpu'
        include_weather_maps: If True, also create cumulative precipitation and water balance maps
        virtual_grid_size: If set, create a NxN grid of virtual stations with interpolated
                          features and model-predicted soil moisture (e.g. 100 for 100 by 100 grid)
        moisture_range: Tuple (min, max) for colorbar range, or 'auto' for data-based range
        hide_markers: Set of marker types to hide: 'real', 'predicted', 'virtual'
        real_moisture_only: If True, only use real moisture stations (skip predictions)
        all_maps: If True, create all map variants efficiently by reusing data:
                  - {base}_moisture_map.png (full: real + predicted + virtual)
                  - {base}_moisture_map_novirtual.png (real + predicted only)
                  - {base}_moisture_map_realonly.png (real sensor data only)
                  - {base}_precipitation.png (cumulative precipitation)
                  - {base}_water_balance.png (cumulative water balance)
        n_nearby_available: For augmented models, how many nearby stations were available
                           for permutations during training
        ensemble_mode: If True, run multiple predictions with augmented inputs and average them.
                      This uses live augmentation (permutations and optionally skip patterns)
                      to create a "single model ensemble" effect.
        ensemble_skip_patterns: When ensemble_mode=True, if True use skip patterns + permutations
                               (120 augmentations with 5 available / 4 output), if False use
                               permutations only (24 augmentations)
    """
    augmented = n_nearby_available is not None and n_nearby_available > n_nearby

    # Ensemble mode requires n_nearby_available to be set
    if ensemble_mode and n_nearby_available is None:
        n_nearby_available = n_nearby + 1  # Default: one extra station for skip patterns
        print(f"  Ensemble mode: defaulting to n_nearby_available={n_nearby_available}")

    if hide_markers is None:
        hide_markers = set()

    # When all_maps is enabled, ensure we collect all data types
    if all_maps:
        real_moisture_only = False  # Need predictions
        if virtual_grid_size is None:
            virtual_grid_size = 100  # Need virtual grid
        include_weather_maps = True  # Will create weather maps too

    print("=" * 60)
    print("CREATING GALICIA SOIL MOISTURE MAP")
    print("=" * 60)

    # Initialize collector
    collector = MeteoGaliciaCollector()

    # Step 1: Load all data
    data = _load_map_data(
        collector, model_path, target_date, device,
        n_nearby, n_nearby_available, augmented
    )

    # Step 2: Collect real data and build sequences for prediction
    real_results, sequences_to_predict = _collect_real_and_build_sequences(
        data['stations_df'], data['timeseries_lookup'], data['nearest_lookup'],
        data['stations_lookup'], data['filtered_params'], data['norm_stats'],
        data['target_date'], n_nearby, real_moisture_only,
        ensemble_mode=ensemble_mode,
        n_nearby_available=n_nearby_available if ensemble_mode else n_nearby
    )

    # Step 3: Run batch inference for predictions
    if ensemble_mode:
        predicted_results = _run_ensemble_inference(
            sequences_to_predict, data['model'], device, collector,
            n_nearby_output=n_nearby,
            n_nearby_available=n_nearby_available,
            use_skip_patterns=ensemble_skip_patterns
        )
    else:
        predicted_results = _run_batch_inference(
            sequences_to_predict, data['model'], device, collector
        )

    # Step 4: Create virtual grid predictions (if enabled)
    virtual_results, virtual_sequences = _create_virtual_grid_predictions(
        virtual_grid_size if not real_moisture_only else None,
        data['galicia_land'], data['stations_df'], data['stations_lookup'],
        data['timeseries_lookup'], data['filtered_params'], data['norm_stats'],
        data['model'], device, collector, data['target_date'], n_nearby,
        sequences_to_predict, predicted_results
    )

    # Step 5: Generate output maps
    results_df = _generate_output_maps(
        real_results, predicted_results, virtual_results,
        data['coastline_points'], data['galicia_land'], data['target_date'],
        output_file, moisture_range, hide_markers, all_maps, include_weather_maps,
        data['stations_df'], data['timeseries_lookup']
    )

    return results_df


def compute_cumulative_weather(stations_df, timeseries_lookup, start_date, end_date):
    """
    Compute cumulative precipitation and water balance for all stations over a date range.

    Args:
        stations_df: DataFrame of stations
        timeseries_lookup: Pre-built dict {(station_id, date_str, parameter_code): value}
        start_date: Start date for accumulation
        end_date: End date for accumulation

    Returns:
        DataFrame with columns: station_id, latitude, longitude, precipitation, water_balance
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    results = []

    for _, station in stations_df.iterrows():
        station_id = station['station_id']
        lat = station['latitude']
        lon = station['longitude']

        # Sum up precipitation over all days (skip invalid -9999 values)
        precip_sum = 0
        precip_count = 0
        for date in date_range:
            date_str = str(date.date())
            key = (station_id, date_str, 'PP_SUM_1.5m')
            if key in timeseries_lookup:
                val = timeseries_lookup[key]
                if val > -9000:  # Valid value
                    precip_sum += val
                    precip_count += 1

        # Sum up water balance over all days (skip invalid -9999 values)
        balance_sum = 0
        balance_count = 0
        for date in date_range:
            date_str = str(date.date())
            key = (station_id, date_str, 'BH_SUM_1.5m')
            if key in timeseries_lookup:
                val = timeseries_lookup[key]
                if val > -9000:  # Valid value
                    balance_sum += val
                    balance_count += 1

        # Only include if we have some data
        if precip_count > 0 or balance_count > 0:
            results.append({
                'station_id': station_id,
                'latitude': lat,
                'longitude': lon,
                'name': station.get('name', f'Station {station_id}'),
                'precipitation': precip_sum if precip_count > 0 else np.nan,
                'water_balance': balance_sum if balance_count > 0 else np.nan,
                'precip_days': precip_count,
                'balance_days': balance_count
            })

    return pd.DataFrame(results)


def create_weather_visualization(results_df, value_column, target_date, output_file,
                                 coastline_points=None, galicia_land=None, title="Weather Map", unit="mm"):
    """Create weather map visualization (similar to moisture map but for precipitation/water balance)"""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree
    import numpy as np
    import contextily as ctx
    from shapely.geometry import Point

    fig, ax = plt.subplots(figsize=(16, 12))


    # Green/blue for water balance (can be negative)
    colors = ['#8B4513', '#D2691E', '#FFFFFF', '#90EE90', '#32CD32', '#228B22', '#4682B4']
    cmap = LinearSegmentedColormap.from_list('water_balance', colors, N=100)

    # Get coordinate bounds from Galicia boundary (consistent across all maps)
    if galicia_land is not None:
        bounds = galicia_land.bounds  # (minx, miny, maxx, maxy)
        lon_min, lat_min, lon_max, lat_max = bounds
    else:
        lon_min, lon_max = results_df['longitude'].min(), results_df['longitude'].max()
        lat_min, lat_max = results_df['latitude'].min(), results_df['latitude'].max()

    # Add 2% padding
    lon_pad = (lon_max - lon_min) * 0.02
    lat_pad = (lat_max - lat_min) * 0.02

    ax.set_xlim(lon_min - lon_pad, lon_max + lon_pad)
    ax.set_ylim(lat_min - lat_pad, lat_max + lat_pad)

    # Add OpenStreetMap basemap
    ctx.add_basemap(
        ax,
        crs="EPSG:4326",
        source=ctx.providers.OpenStreetMap.Mapnik,
        alpha=0.5,
        zoom=10
    )

    # Create interpolation grid
    grid_lon = np.linspace(lon_min - lon_pad, lon_max + lon_pad, 400)
    grid_lat = np.linspace(lat_min - lat_pad, lat_max + lat_pad, 400)
    grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)

    # Prepare station points and values
    station_points = results_df[['longitude', 'latitude']].values
    station_values = results_df[value_column].values

    # Add virtual stations along coastline if available
    if coastline_points is not None and galicia_land is not None:
        tree = cKDTree(station_points)
        distances, indices = tree.query(coastline_points, k=2)

        coastline_values = np.zeros(len(coastline_points))
        for i in range(len(coastline_points)):
            dists = distances[i]
            idxs = indices[i]
            if dists[0] < 1e-9:
                coastline_values[i] = station_values[idxs[0]]
            else:
                weights = 1.0 / dists
                weights = weights / weights.sum()
                coastline_values[i] = np.sum(weights * station_values[idxs])

        all_points = np.vstack([station_points, coastline_points])
        all_values = np.concatenate([station_values, coastline_values])

        # Create land mask
        land_mask = np.zeros_like(grid_lon_mesh, dtype=bool)
        for i in range(grid_lon_mesh.shape[0]):
            for j in range(grid_lon_mesh.shape[1]):
                point = Point(grid_lon_mesh[i, j], grid_lat_mesh[i, j])
                if galicia_land.contains(point):
                    land_mask[i, j] = True
    else:
        all_points = station_points
        all_values = station_values
        land_mask = None

    # Interpolate
    grid_values = griddata(all_points, all_values, (grid_lon_mesh, grid_lat_mesh), method='linear')

    # Fill NaN at edges
    mask_nan = np.isnan(grid_values)
    if mask_nan.any():
        grid_values_nearest = griddata(all_points, all_values, (grid_lon_mesh, grid_lat_mesh), method='nearest')
        grid_values[mask_nan] = grid_values_nearest[mask_nan]

    # Apply land mask
    if land_mask is not None:
        grid_values[~land_mask] = np.nan

    # Plot contour
    values_plot = ax.contourf(
        grid_lon_mesh, grid_lat_mesh, grid_values,
        levels=20, cmap=cmap, alpha=0.5, extend='both', zorder=2
    )

    # Plot station points
    scatter = ax.scatter(
        results_df['longitude'], results_df['latitude'],
        c=results_df[value_column], s=150, cmap=cmap,
        edgecolors='black', linewidths=2.5,
        marker='o', label='Station data',
        vmin=station_values.min(), vmax=station_values.max(), zorder=10
    )

    # Add colorbar
    cbar = plt.colorbar(values_plot, ax=ax, pad=0.02, shrink=0.8)
    cbar.set_label(f'{value_column.replace("_", " ").title()} ({unit})', fontsize=14, weight='bold')
    cbar.ax.tick_params(labelsize=11)

    # Labels and title
    ax.set_xlabel('Longitude', fontsize=13, weight='bold')
    ax.set_ylabel('Latitude', fontsize=13, weight='bold')
    ax.set_title(title, fontsize=18, weight='bold', pad=20)

    # Legend
    legend = ax.legend(loc='upper right', fontsize=12, framealpha=0.95,
                      edgecolor='black', fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Stats text box
    stats_text = (
        f"Stations: {len(results_df)} with data\n"
        f"Range: {station_values.min():.1f} - {station_values.max():.1f} {unit}\n"
        f"Mean: {station_values.mean():.1f} {unit}"
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    # Save
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def create_visualization(results_df, target_date, output_file, coastline_points=None, galicia_land=None,
                         moisture_range=(0.07, 0.40), hide_markers=None):
    """Create beautiful moisture map visualization overlaid on Galicia map

    Args:
        moisture_range: Tuple (min, max) for colorbar range, or 'auto' for data-based range
        hide_markers: Set of marker types to hide: 'real', 'predicted', 'virtual'
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree
    import numpy as np
    import contextily as ctx
    from shapely.geometry import Point

    if hide_markers is None:
        hide_markers = set()

    fig, ax = plt.subplots(figsize=(16, 12))

    # Custom colormap: dry (brown) -> moist (green) -> wet (blue)
    colors = ['#8B4513', '#D2691E', '#F4A460', '#90EE90', '#32CD32', '#228B22', '#4682B4', '#1E90FF']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('moisture', colors, N=n_bins)

    # Determine value range for colorbar
    if moisture_range == 'auto':
        vmin = results_df['moisture'].min()
        vmax = results_df['moisture'].max()
    else:
        vmin, vmax = moisture_range

    # Get coordinate bounds from Galicia boundary (consistent across all maps)
    # Fall back to station data if boundary not available
    if galicia_land is not None:
        bounds = galicia_land.bounds  # (minx, miny, maxx, maxy)
        lon_min, lat_min, lon_max, lat_max = bounds
    else:
        lon_min, lon_max = results_df['longitude'].min(), results_df['longitude'].max()
        lat_min, lat_max = results_df['latitude'].min(), results_df['latitude'].max()

    # Add 10% padding
    lon_pad = (lon_max - lon_min) * 0.02
    lat_pad = (lat_max - lat_min) * 0.02

    # Set axis limits
    ax.set_xlim(lon_min - lon_pad, lon_max + lon_pad)
    ax.set_ylim(lat_min - lat_pad, lat_max + lat_pad)

    # Add OpenStreetMap basemap
    ctx.add_basemap(
        ax,
        crs="EPSG:4326",  # WGS84 lat/lon
        source=ctx.providers.OpenStreetMap.Mapnik,
        alpha=0.5,
        zoom=10
    )
    print("  ✓ Added OpenStreetMap basemap")

    # Create interpolation grid (higher resolution for smoother contours)
    grid_lon = np.linspace(lon_min - lon_pad, lon_max + lon_pad, 400)
    grid_lat = np.linspace(lat_min - lat_pad, lat_max + lat_pad, 400)
    grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)

    # Prepare station points and values
    station_points = results_df[['longitude', 'latitude']].values
    station_values = results_df['moisture'].values

    # Add virtual stations along coastline to constrain interpolation
    print("  Creating virtual coastline stations with distance-weighted averaging...")
    tree = cKDTree(station_points)

    # Query for 2 nearest neighbors for each coastline point
    distances, indices = tree.query(coastline_points, k=2)

    # Compute distance-weighted average for each coastline point
    coastline_values = np.zeros(len(coastline_points))
    for i in range(len(coastline_points)):
        # Get distances and indices for the 2 nearest stations
        dists = distances[i]  # shape: (2,)
        idxs = indices[i]     # shape: (2,)

        # Handle edge case: if a coastline point is exactly on a station (distance = 0)
        if dists[0] < 1e-9:  # essentially zero distance
            coastline_values[i] = station_values[idxs[0]]
        else:
            # Inverse distance weighting: weight_i = 1/distance_i
            weights = 1.0 / dists
            # Normalize weights so they sum to 1
            weights = weights / weights.sum()
            # Weighted average
            coastline_values[i] = np.sum(weights * station_values[idxs])

    # Add coastline virtual stations to interpolation data
    all_points = np.vstack([station_points, coastline_points])
    all_values = np.concatenate([station_values, coastline_values])

    print(f"  ✓ Added {len(coastline_points)} virtual coastline stations (2-nearest distance-weighted)")

    # Create land mask for grid
    land_mask = np.zeros_like(grid_lon_mesh, dtype=bool)
    for i in range(grid_lon_mesh.shape[0]):
        for j in range(grid_lon_mesh.shape[1]):
            point = Point(grid_lon_mesh[i, j], grid_lat_mesh[i, j])
            if galicia_land.contains(point):
                land_mask[i, j] = True

    # Interpolate moisture across the region using all points (real + virtual)
    grid_moisture = griddata(all_points, all_values, (grid_lon_mesh, grid_lat_mesh), method='linear')

    # Fill NaN values at edges using nearest neighbor
    mask_nan = np.isnan(grid_moisture)
    if mask_nan.any():
        grid_moisture_nearest = griddata(all_points, all_values, (grid_lon_mesh, grid_lat_mesh), method='nearest')
        grid_moisture[mask_nan] = grid_moisture_nearest[mask_nan]

    # Apply land mask to exclude sea areas
    grid_moisture[~land_mask] = np.nan
    print("  ✓ Applied land mask to exclude sea areas")

    # Plot interpolated moisture as semi-transparent overlay (50% opacity)
    moisture_plot = ax.contourf(
        grid_lon_mesh, grid_lat_mesh, grid_moisture,
        levels=np.linspace(vmin, vmax, 21), cmap=cmap, alpha=0.5, extend='both', zorder=2
    )

    # Plot station points
    real_data = results_df[results_df['type'] == 'real']
    pred_data = results_df[results_df['type'] == 'predicted']
    virtual_data = results_df[results_df['type'] == 'virtual'] if 'type' in results_df.columns else pd.DataFrame()

    # Virtual data: small black dots (plot first so they're behind real stations)
    if len(virtual_data) > 0 and 'virtual' not in hide_markers:
        scatter_virtual = ax.scatter(
            virtual_data['longitude'], virtual_data['latitude'],
            c='black', s=3,  # Small black dots
            marker='.', label=f'Virtual grid ({len(virtual_data)} pts)',
            zorder=7, alpha=0.6
        )

    # Real data: solid circles with black outline
    if 'real' not in hide_markers:
        scatter_real = ax.scatter(
            real_data['longitude'], real_data['latitude'],
            c=real_data['moisture'], s=150, cmap=cmap,
            edgecolors='black', linewidths=2.5,
            marker='o', label='Real data (sensors)',
            vmin=vmin, vmax=vmax, zorder=10
        )

    # Predicted data: triangles with gray outline
    if len(pred_data) > 0 and 'predicted' not in hide_markers:
        scatter_pred = ax.scatter(
            pred_data['longitude'], pred_data['latitude'],
            c=pred_data['moisture'], s=130, cmap=cmap,
            edgecolors='white', linewidths=2,
            marker='^', label='Predicted (no sensor)',
            vmin=vmin, vmax=vmax, zorder=9
        )

    # Add colorbar with percentage formatting
    cbar = plt.colorbar(moisture_plot, ax=ax, pad=0.02, shrink=0.8)
    cbar.set_label('Soil Moisture (%)', fontsize=14, weight='bold')
    cbar.ax.tick_params(labelsize=11)
    # Format tick labels as percentages
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.1f}'))

    # Labels and title
    ax.set_xlabel('Longitude', fontsize=13, weight='bold')
    ax.set_ylabel('Latitude', fontsize=13, weight='bold')
    ax.set_title(
        f'Galicia Soil Moisture Map\n{target_date.strftime("%Y-%m-%d")}',
        fontsize=18, weight='bold', pad=20
    )

    # Legend with better styling (only if there are visible items)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        legend = ax.legend(loc='upper right', fontsize=12, framealpha=0.95,
                           edgecolor='black', fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('white')

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Stats text box
    stats_lines = [
        f"Stations: {len(real_data) + len(pred_data)} total",
        f"Real: {len(real_data)} | Predicted: {len(pred_data)}"
    ]
    if len(virtual_data) > 0:
        stats_lines.append(f"Virtual grid: {len(virtual_data)}")
    stats_lines.extend([
        f"Range: {vmin:.1%} - {vmax:.1%}",
        f"Mean: {station_values.mean():.1%}"
    ])
    stats_text = "\n".join(stats_lines)
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create Galicia soil moisture map')
    parser.add_argument('--model', type=str, default="trololo.weight", help='Path to trained model checkpoint')
    parser.add_argument('--date', type=str, default="2025-10-25", help='Target date (YYYY-MM-DD), default: most recent')
    parser.add_argument('--output', type=str, default='galicia_moisture_map.png', help='Output file path')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--include-weather-maps', action='store_true',
                       help='Also create cumulative precipitation and water balance maps')
    parser.add_argument('--virtual-grid', type=int, default=None,
                       help='Create NxN grid of virtual stations (e.g., 100 for 100x100 grid)')
    parser.add_argument('--range', type=str, default='0.07,0.40',
                       help='Moisture colorbar range: "MIN,MAX" (default: 0.07,0.40) or "auto" for data-based range')
    parser.add_argument('--hide-markers', type=str, default=None,
                       help='Comma-separated list of markers to hide: real,predicted,virtual (e.g., --hide-markers predicted,virtual)')
    parser.add_argument('--real-moisture-only', action='store_true',
                       help='Draw map using only real moisture stations (no predictions)')
    parser.add_argument('--all-maps', action='store_true',
                       help='Create all map variants efficiently (moisture, novirtual, realonly, precipitation, water_balance)')
    parser.add_argument('--n-nearby', type=int, default=4,
                       help='Number of nearby stations used in model input (default: 4)')
    parser.add_argument('--n-nearby-available', type=int, default=5,
                       help='For augmented models: how many nearby stations were available for permutations')
    parser.add_argument('--ensemble', action='store_true',
                       help='Use ensemble prediction: average multiple predictions with augmented inputs')
    parser.add_argument('--ensemble-permutations-only', action='store_true',
                       help='When using --ensemble, only use permutations (24) instead of skip patterns + permutations (120)')

    args = parser.parse_args()

    # Parse hide-markers into a set
    hide_markers = set()
    if args.hide_markers:
        hide_markers = set(m.strip().lower() for m in args.hide_markers.split(','))

    # Parse --range argument: either "auto" or "min,max"
    if args.range.lower() == 'auto':
        moisture_range = 'auto'
    else:
        try:
            parts = args.range.split(',')
            moisture_range = (float(parts[0]), float(parts[1]))
        except (ValueError, IndexError):
            print(f"Error: Invalid --range format '{args.range}'. Use 'auto' or 'MIN,MAX' (e.g., '0.07,0.40')")
            sys.exit(1)

    create_moisture_map(
        model_path=args.model,
        target_date=args.date,
        output_file=args.output,
        device=args.device,
        include_weather_maps=args.include_weather_maps,
        virtual_grid_size=args.virtual_grid,
        moisture_range=moisture_range,
        hide_markers=hide_markers,
        real_moisture_only=args.real_moisture_only,
        all_maps=args.all_maps,
        n_nearby=args.n_nearby,
        n_nearby_available=args.n_nearby_available,
        ensemble_mode=args.ensemble,
        ensemble_skip_patterns=not args.ensemble_permutations_only
    )
