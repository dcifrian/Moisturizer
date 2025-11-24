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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
from datetime import datetime, timedelta
import torch
from pathlib import Path

from Moisturizer import MeteoGaliciaCollector, SoilMoistureSequenceDataset


def load_coastline_data(lon_min, lon_max, lat_min, lat_max, padding=0.15):
    """
    Load and prepare coastline data early to fail fast if there are issues.

    Returns:
        tuple: (coastline_points, galicia_land)
    """
    import geopandas as gpd
    from shapely.geometry import box, Point, LineString

    print("  Loading Natural Earth coastline data...")
    # Download 10m resolution land data from Natural Earth CDN
    url = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip"
    world = gpd.read_file(url)

    # Add padding to bounding box
    lon_pad = (lon_max - lon_min) * padding
    lat_pad = (lat_max - lat_min) * padding

    # Clip to our region of interest
    bbox = box(lon_min - lon_pad, lat_min - lat_pad, lon_max + lon_pad, lat_max + lat_pad)

    # Get land areas in our bbox
    galicia_land = world.geometry.intersection(bbox)
    galicia_land = galicia_land[~galicia_land.is_empty].unary_union

    print("  Sampling coastline points...")
    # Extract exterior boundary (coastline)
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
    print(f"  ✓ Loaded coastline with {len(coastline_points)} sampled points")

    return coastline_points, galicia_land


def load_model(model_path, device='cuda'):
    """Load trained TROLOLO model"""
    print(f"Loading model from {model_path}...")

    # Import TROLOLO (adjust import based on your structure)
    try:
        from TROLOLO.TROLOLO_pyramid import TROLOLO
    except ImportError:
        print("Error: Could not import TROLOLO. Make sure it's in your path.")
        return None

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    quantize = False
    """trololo = TROLOLO(seq_length=64,
                      num_layers=8,
                      num_heads=48,
                      embed_dim=192,
                      mlp_dim=512,
                      n_class_tokens=2,
                      num_classes=1,
                      mlp_rank=0.1,
                      qkv_rank=0.2,
                      attnproj_rank=0.1,
                      sequence_pyramid=[],
                      attn_rank_pyramid=[],
                      rank_pyramid_begin=2,
                      rank_pyramid_factor=1.0,
                      head_constriction="ONE_CLASS_TOKEN",
                      dropout=0.05,
                      attention_dropout=0.01,
                      quantize_bits=None if not quantize else 8
                      )
    """
    trololo = TROLOLO(seq_length=64,
                      num_layers=16,
                      num_heads=48,
                      embed_dim=192,
                      mlp_dim=512,
                      n_class_tokens=2,
                      num_classes=1,
                      mlp_rank=0.1,
                      qkv_rank=0.2,
                      attnproj_rank=0.1,
                      sequence_pyramid=[],
                      attn_rank_pyramid=[],
                      rank_pyramid_begin=2,
                      rank_pyramid_factor=1.0,
                      head_constriction="ONE_CLASS_TOKEN",
                      dropout=0.05,
                      attention_dropout=0.01,
                      quantize_bits= None if not quantize else 8
                      )
    trololo.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    trololo.to(device)
    trololo.eval()

    print(f"✓ Model loaded successfully")
    return trololo


def predict_for_station(model, dataset, station_id, end_date, device='cuda'):
    """
    Get soil moisture prediction for a specific station and date

    Args:
        model: Trained TROLOLO model
        dataset: SoilMoistureSequenceDataset
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
    target_min = float(norm_stats['target_min'])
    target_max = float(norm_stats['target_max'])

    # Denormalize: value in [-1, 1] -> original range
    original = (normalized_value + 1.0) / 2.0 * (target_max - target_min) + target_min
    return original


def get_real_soil_moisture(collector, station_id, date):
    """Get actual soil moisture from timeseries data if available"""
    timeseries_df = pd.read_csv(collector.timeseries_file)
    timeseries_df['date'] = pd.to_datetime(timeseries_df['date'])

    data = timeseries_df[
        (timeseries_df['station_id'] == station_id) &
        (timeseries_df['date'] == date) &
        (timeseries_df['parameter_code'] == 'HS_CV_AVG_-0.2m')
    ]

    if not data.empty:
        return data.iloc[0]['value']
    return None


def build_fast_timeseries_lookup(timeseries_df, start_date, end_date, station_ids, feature_params):
    """
    Pre-process timeseries into fast numpy lookup structure.

    Returns dict: {(station_id, date_str, parameter_code): value}
    """
    # Filter to relevant date range and stations
    mask = (
        (timeseries_df['date'] >= start_date) &
        (timeseries_df['date'] <= end_date) &
        (timeseries_df['station_id'].isin(station_ids))
    )
    filtered = timeseries_df[mask]

    # Create lookup dict for O(1) access
    lookup = {}
    for _, row in filtered.iterrows():
        key = (int(row['station_id']), str(row['date'].date()), row['parameter_code'])
        lookup[key] = float(row['value'])

    return lookup


def build_sequence_for_any_station(
    station_id,
    end_date,
    timeseries_lookup,
    nearest_df,
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
    """
    import numpy as np

    start_date = end_date - timedelta(days=seq_length - 1)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Get nearest stations for this station
    nearest_info = nearest_df[nearest_df['station_id'] == station_id]
    if nearest_info.empty:
        return None

    nearest_info = nearest_info.iloc[0]

    # Find n nearest stations WITH soil moisture
    nearby_stations = []
    for i in range(1, 50):  # Check up to 50 nearest
        if f'nearest_{i}_id' not in nearest_info:
            break
        if nearest_info.get(f'nearest_{i}_has_soil_moisture', False):
            nearby_stations.append({
                'station_id': int(nearest_info[f'nearest_{i}_id']),
                'distance': nearest_info[f'nearest_{i}_distance']
            })
            if len(nearby_stations) == n_nearest:
                break

    if len(nearby_stations) < n_nearest:
        return None

    # Calculate feature dimensions
    target_features_per_timestep = len(feature_params)
    nearby_features_per_timestep = (len(feature_params) + 1 + 1)  # features + soil moisture + distance
    total_features = target_features_per_timestep + (nearby_features_per_timestep * n_nearest)

    # Initialize arrays
    features = np.full((seq_length, total_features), missing_value, dtype=np.float32)
    mask = np.zeros((seq_length, total_features), dtype=np.float32)

    # Fill target station features using fast lookup
    for t, date in enumerate(date_range):
        date_str = str(date.date())

        for f_idx, param in enumerate(feature_params):
            key = (station_id, date_str, param)
            if key in timeseries_lookup:
                features[t, f_idx] = timeseries_lookup[key]
                mask[t, f_idx] = 1.0

        # Fill nearby stations features
        for n_idx, nearby in enumerate(nearby_stations):
            nearby_offset = target_features_per_timestep + (n_idx * nearby_features_per_timestep)

            # Distance (constant across time)
            features[t, nearby_offset] = nearby['distance']
            mask[t, nearby_offset] = 1.0

            # Features
            for f_idx, param in enumerate(feature_params):
                key = (nearby['station_id'], date_str, param)
                feat_idx = nearby_offset + 1 + f_idx
                if key in timeseries_lookup:
                    features[t, feat_idx] = timeseries_lookup[key]
                    mask[t, feat_idx] = 1.0

            # Soil moisture for nearby station
            key = (nearby['station_id'], date_str, 'HS_CV_AVG_-0.2m')
            soil_idx = nearby_offset + 1 + len(feature_params)
            if key in timeseries_lookup:
                features[t, soil_idx] = timeseries_lookup[key]
                mask[t, soil_idx] = 1.0

    # Apply normalization
    features_normalized = apply_normalization_to_features(features, mask, norm_stats, missing_value)

    return features_normalized, mask


def build_sequence_for_any_station_OLD(
    station_id,
    end_date,
    timeseries_df,
    nearest_df,
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
    """
    import numpy as np

    start_date = end_date - timedelta(days=seq_length - 1)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Get nearest stations for this station
    nearest_info = nearest_df[nearest_df['station_id'] == station_id]
    if nearest_info.empty:
        return None

    nearest_info = nearest_info.iloc[0]

    # Find n nearest stations WITH soil moisture
    nearby_stations = []
    for i in range(1, 50):  # Check up to 50 nearest
        if f'nearest_{i}_id' not in nearest_info:
            break
        if nearest_info.get(f'nearest_{i}_has_soil_moisture', False):
            nearby_stations.append({
                'station_id': int(nearest_info[f'nearest_{i}_id']),
                'distance': nearest_info[f'nearest_{i}_distance']
            })
            if len(nearby_stations) == n_nearest:
                break

    if len(nearby_stations) < n_nearest:
        print(f"  Warning: Station {station_id} only has {len(nearby_stations)}/{n_nearest} nearby stations with soil moisture")
        return None

    # Calculate feature dimensions
    target_features_per_timestep = len(feature_params)
    nearby_features_per_timestep = (len(feature_params) + 1 + 1)  # features + soil moisture + distance
    total_features = target_features_per_timestep + (nearby_features_per_timestep * n_nearest)

    # Initialize arrays
    features = np.full((seq_length, total_features), missing_value, dtype=np.float32)
    mask = np.zeros((seq_length, total_features), dtype=np.float32)

    # Fill target station features (this station, even if no soil moisture sensor)
    for t, date in enumerate(date_range):
        target_data = timeseries_df[
            (timeseries_df['station_id'] == station_id) &
            (timeseries_df['date'] == date)
        ]

        for f_idx, param in enumerate(feature_params):
            param_data = target_data[target_data['parameter_code'] == param]
            if not param_data.empty:
                features[t, f_idx] = param_data.iloc[0]['value']
                mask[t, f_idx] = 1.0

        # Fill nearby stations features (these have soil moisture - context!)
        for n_idx, nearby in enumerate(nearby_stations):
            nearby_data = timeseries_df[
                (timeseries_df['station_id'] == nearby['station_id']) &
                (timeseries_df['date'] == date)
            ]

            nearby_offset = target_features_per_timestep + (n_idx * nearby_features_per_timestep)

            # Distance (constant across time)
            features[t, nearby_offset] = nearby['distance']
            mask[t, nearby_offset] = 1.0

            # Features
            for f_idx, param in enumerate(feature_params):
                param_data = nearby_data[nearby_data['parameter_code'] == param]
                feat_idx = nearby_offset + 1 + f_idx
                if not param_data.empty:
                    features[t, feat_idx] = param_data.iloc[0]['value']
                    mask[t, feat_idx] = 1.0

            # Soil moisture for nearby station (the key context!)
            soil_data = nearby_data[nearby_data['parameter_code'] == 'HS_CV_AVG_-0.2m']
            soil_idx = nearby_offset + 1 + len(feature_params)
            if not soil_data.empty:
                features[t, soil_idx] = soil_data.iloc[0]['value']
                mask[t, soil_idx] = 1.0

    # Apply normalization
    features_normalized = apply_normalization_to_features(features, mask, norm_stats, missing_value)

    return features_normalized, mask


def apply_normalization_to_features(features, mask, norm_stats, missing_value=-1000.0):
    """Apply normalization to features (same as in Dataset)"""
    import numpy as np

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

    invalid_markers = [-9999.0, missing_value]
    normalized_invalid_marker = -2.0

    features_norm = features.copy()

    # Normalize ALL feature columns (target station + nearby stations)
    for feat_idx in range(features.shape[1]):
        feat_min = feature_mins[feat_idx]
        feat_max = feature_maxs[feat_idx]

        # Handle invalid markers
        invalid_mask = np.zeros(features.shape[0], dtype=bool)
        for marker in invalid_markers:
            invalid_mask |= (features[:, feat_idx] == marker)

        # Normalize valid values to [-1, 1]
        if feat_max > feat_min:
            features_norm[:, feat_idx] = 2.0 * (features[:, feat_idx] - feat_min) / (feat_max - feat_min) - 1.0

        # Set invalid markers to -2
        features_norm[invalid_mask, feat_idx] = normalized_invalid_marker

    return features_norm


def get_real_soil_moisture(collector, station_id, date):
    """Get actual soil moisture from timeseries data if available"""
    timeseries_df = pd.read_csv(collector.timeseries_file)
    timeseries_df['date'] = pd.to_datetime(timeseries_df['date'])

    data = timeseries_df[
        (timeseries_df['station_id'] == station_id) &
        (timeseries_df['date'] == date) &
        (timeseries_df['parameter_code'] == 'HS_CV_AVG_-0.2m')
    ]

    if not data.empty:
        return data.iloc[0]['value']
    return None


def create_moisture_map(
    model_path,
    target_date=None,
    output_file='galicia_moisture_map.png',
    device='cuda',
    include_weather_maps=False
):
    """
    Create a beautiful moisture map of all Galicia

    Args:
        model_path: Path to trained model checkpoint
        target_date: Date to predict (default: most recent available)
        output_file: Path to save the visualization
        device: 'cuda' or 'cpu'
        include_weather_maps: If True, also create cumulative precipitation and water balance maps
    """
    print("=" * 60)
    print("CREATING GALICIA SOIL MOISTURE MAP")
    print("=" * 60)

    # Initialize collector
    collector = MeteoGaliciaCollector()

    # Load stations
    stations_df = pd.read_csv(collector.stations_file)
    print(f"\nFound {len(stations_df)} stations total")
    print(f"  - {stations_df['has_soil_moisture'].sum()} with soil moisture sensors")
    print(f"  - {(~stations_df['has_soil_moisture']).sum()} without sensors (will predict)")

    # Load coastline data early (fail fast before expensive model inference)
    print("\nPreparing coastline data...")
    lon_min, lon_max = stations_df['longitude'].min(), stations_df['longitude'].max()
    lat_min, lat_max = stations_df['latitude'].min(), stations_df['latitude'].max()
    coastline_points, galicia_land = load_coastline_data(lon_min, lon_max, lat_min, lat_max)

    # Determine target date
    if target_date is None:
        # Use most recent date in timeseries
        timeseries_df = pd.read_csv(collector.timeseries_file)
        timeseries_df['date'] = pd.to_datetime(timeseries_df['date'])
        target_date = timeseries_df['date'].max()
    else:
        target_date = pd.to_datetime(target_date)

    print(f"\nTarget date: {target_date.strftime('%Y-%m-%d')}")

    # Load model
    model = load_model(model_path, device)
    if model is None:
        return

    # Load dataset for inference
    print("\nLoading dataset...")
    _, filtered_params = collector.analyze_parameter_coverage(coverage_threshold=0.25)

    dataset = SoilMoistureSequenceDataset(
        timeseries=str(collector.timeseries_file),
        stations=str(collector.stations_file),
        nearest=str(collector.nearest_file),
        seq_length=64,
        n_nearest=4,
        feature_params=filtered_params,
        precomputed_path=str(collector.data_dir / "precomputed_sequences"),  # .npy directory
        normalize=True,
        norm_stats_path=str(collector.data_dir / "normalization_stats.npz")
    )

    # Load timeseries and nearest data for on-the-fly predictions
    print("\nLoading timeseries data for predictions...")
    timeseries_df = pd.read_csv(collector.timeseries_file)
    timeseries_df['date'] = pd.to_datetime(timeseries_df['date'])
    nearest_df = pd.read_csv(collector.nearest_file)

    # Load normalization stats
    norm_stats = np.load(str(collector.data_dir / "normalization_stats.npz"))

    # Build fast lookup for timeseries data
    print("\nBuilding fast timeseries lookup...")
    start_date = target_date - timedelta(days=64 - 1)

    # Collect all station IDs we'll need (all stations + their nearby stations)
    all_needed_stations = set(stations_df['station_id'].tolist())
    for _, station in stations_df.iterrows():
        if not station['has_soil_moisture']:
            # Add nearby stations for this station
            nearest_info = nearest_df[nearest_df['station_id'] == station['station_id']]
            if not nearest_info.empty:
                nearest_info = nearest_info.iloc[0]
                for i in range(1, 50):
                    if f'nearest_{i}_id' not in nearest_info:
                        break
                    if nearest_info.get(f'nearest_{i}_has_soil_moisture', False):
                        all_needed_stations.add(int(nearest_info[f'nearest_{i}_id']))

    timeseries_lookup = build_fast_timeseries_lookup(
        timeseries_df, start_date, target_date,
        list(all_needed_stations), filtered_params
    )
    print(f"  ✓ Built lookup with {len(timeseries_lookup)} entries for {len(all_needed_stations)} stations")

    # Phase 1: Collect real data and build sequences for predictions
    print("\nPhase 1: Gathering real data and building sequences...")
    real_results = []
    sequences_to_predict = []  # List of (station_info, features_norm, mask)

    for idx, station in stations_df.iterrows():
        if idx % 20 == 0:
            print(f"  Processing station {idx+1}/{len(stations_df)}...")

        station_id = station['station_id']
        lat = station['latitude']
        lon = station['longitude']
        has_sensor = station['has_soil_moisture']

        if has_sensor:
            # Get real data
            moisture = get_real_soil_moisture(collector, station_id, target_date)
            if moisture is not None:
                real_results.append({
                    'station_id': station_id,
                    'latitude': lat,
                    'longitude': lon,
                    'moisture': moisture,
                    'type': 'real',
                    'name': station.get('name', f'Station {station_id}')
                })
        else:
            # Build sequence for later batch inference
            try:
                sequence_data = build_sequence_for_any_station(
                    station_id=station_id,
                    end_date=target_date,
                    timeseries_lookup=timeseries_lookup,
                    nearest_df=nearest_df,
                    feature_params=filtered_params,
                    norm_stats=norm_stats,
                    seq_length=64,
                    n_nearest=4
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

            except Exception as e:
                print(f"  Warning: Could not build sequence for station {station_id}: {e}")
                continue

    print(f"✓ Phase 1 complete: {len(real_results)} real, {len(sequences_to_predict)} to predict")

    # Phase 2: Batch inference for all predictions
    predicted_results = []
    if sequences_to_predict:
        print(f"\nPhase 2: Running batched inference for {len(sequences_to_predict)} stations...")

        batch_size = len(sequences_to_predict)

        # Stack all sequences - same as unsqueeze(0) but for multiple items
        X_batch = torch.stack([
            torch.from_numpy(features_norm) for _, features_norm, _ in sequences_to_predict
        ])  # [batch_size, 64, features]

        print(f"  Batch shape: {X_batch.shape}")

        # Run batched inference - same pattern as original but batch_size instead of 1
        x_gpu = torch.zeros([batch_size, model.embed_dim - 2, model.seq_length - model.n_class_tokens],
                           dtype=torch.float16, device=device)
        torch._dynamo.config.disable = True
        with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=True, cache_enabled=True, dtype=torch.bfloat16):
            x_gpu[:batch_size, :X_batch.shape[2], :].copy_(X_batch.permute(0, 2, 1), non_blocking=True)
            x = x_gpu[:batch_size, :, :]
            predictions_normalized = model(x).cpu()  # [batch_size, 1] or [batch_size]

        print(f"✓ Inference complete")

        # Phase 3: Denormalize and assemble results
        print(f"\nPhase 3: Denormalizing predictions...")
        for i, (station_info, _, _) in enumerate(sequences_to_predict):
            pred_normalized = predictions_normalized[i].item()  # Extract scalar same as original
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

    # Combine all results
    all_results = real_results + predicted_results
    results_df = pd.DataFrame(all_results)
    print(f"\n✓ Collected data for {len(results_df)} stations")
    print(f"  - Real: {(results_df['type'] == 'real').sum()}")
    print(f"  - Predicted: {(results_df['type'] == 'predicted').sum()}")

    # Create visualization
    print(f"\nCreating visualization...")
    create_visualization(results_df, target_date, output_file, coastline_points, galicia_land)

    print(f"\n✓ Map saved to {output_file}")

    # Create optional weather maps if requested
    if include_weather_maps:
        print("\n" + "=" * 60)
        print("CREATING CUMULATIVE WEATHER MAPS")
        print("=" * 60)

        # Compute cumulative precipitation and water balance over the 64-day sequence
        start_date = target_date - timedelta(days=63)  # 64 days total
        weather_results = compute_cumulative_weather(
            stations_df, timeseries_lookup, start_date, target_date
        )

        if not weather_results.empty:
            # Create precipitation map
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

            # Create water balance map
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

        # Sum up precipitation over all days
        precip_sum = 0
        precip_count = 0
        for date in date_range:
            date_str = str(date.date())
            key = (station_id, date_str, 'PP_SUM_1.5m')
            if key in timeseries_lookup:
                precip_sum += timeseries_lookup[key]
                precip_count += 1

        # Sum up water balance over all days
        balance_sum = 0
        balance_count = 0
        for date in date_range:
            date_str = str(date.date())
            key = (station_id, date_str, 'BH_SUM_1.5m')
            if key in timeseries_lookup:
                balance_sum += timeseries_lookup[key]
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

    # Custom colormap based on data type
    if 'precipitation' in value_column:
        # Blue-based for precipitation
        colors = ['#FFFFFF', '#E0F3FF', '#A8DAFF', '#70C1FF', '#4A90E2', '#2E5FA8', '#1A3A6F']
        cmap = LinearSegmentedColormap.from_list('precipitation', colors, N=100)
    else:
        # Green/blue for water balance (can be negative)
        colors = ['#8B4513', '#D2691E', '#FFFFFF', '#90EE90', '#32CD32', '#228B22', '#4682B4']
        cmap = LinearSegmentedColormap.from_list('water_balance', colors, N=100)

    # Get coordinate bounds with padding
    lon_min, lon_max = results_df['longitude'].min(), results_df['longitude'].max()
    lat_min, lat_max = results_df['latitude'].min(), results_df['latitude'].max()

    lon_pad = (lon_max - lon_min) * 0.15
    lat_pad = (lat_max - lat_min) * 0.15

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


def create_visualization(results_df, target_date, output_file, coastline_points=None, galicia_land=None):
    """Create beautiful moisture map visualization overlaid on Galicia map"""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree
    import numpy as np
    import contextily as ctx
    from shapely.geometry import Point

    fig, ax = plt.subplots(figsize=(16, 12))

    # Custom colormap: dry (brown) -> moist (green) -> wet (blue)
    colors = ['#8B4513', '#D2691E', '#F4A460', '#90EE90', '#32CD32', '#228B22', '#4682B4', '#1E90FF']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('moisture', colors, N=n_bins)

    # Get coordinate bounds with extra padding for extrapolation
    lon_min, lon_max = results_df['longitude'].min(), results_df['longitude'].max()
    lat_min, lat_max = results_df['latitude'].min(), results_df['latitude'].max()

    # Add padding
    lon_pad = (lon_max - lon_min) * 0.15
    lat_pad = (lat_max - lat_min) * 0.15

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
        levels=20, cmap=cmap, alpha=0.5, extend='both', zorder=2
    )

    # Plot station points
    real_data = results_df[results_df['type'] == 'real']
    pred_data = results_df[results_df['type'] == 'predicted']

    # Real data: solid circles with black outline
    scatter_real = ax.scatter(
        real_data['longitude'], real_data['latitude'],
        c=real_data['moisture'], s=150, cmap=cmap,
        edgecolors='black', linewidths=2.5,
        marker='o', label='Real data (sensors)',
        vmin=station_values.min(), vmax=station_values.max(), zorder=10
    )

    # Predicted data: triangles with gray outline
    if len(pred_data) > 0:
        scatter_pred = ax.scatter(
            pred_data['longitude'], pred_data['latitude'],
            c=pred_data['moisture'], s=130, cmap=cmap,
            edgecolors='white', linewidths=2,
            marker='^', label='Predicted (no sensor)',
            vmin=station_values.min(), vmax=station_values.max(), zorder=9
        )

    # Add colorbar
    cbar = plt.colorbar(moisture_plot, ax=ax, pad=0.02, shrink=0.8)
    cbar.set_label('Soil Moisture (%)', fontsize=14, weight='bold')
    cbar.ax.tick_params(labelsize=11)

    # Labels and title
    ax.set_xlabel('Longitude', fontsize=13, weight='bold')
    ax.set_ylabel('Latitude', fontsize=13, weight='bold')
    ax.set_title(
        f'Galicia Soil Moisture Map\n{target_date.strftime("%Y-%m-%d")}',
        fontsize=18, weight='bold', pad=20
    )

    # Legend with better styling
    legend = ax.legend(loc='upper right', fontsize=12, framealpha=0.95,
                       edgecolor='black', fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Stats text box
    stats_text = (
        f"Stations: {len(results_df)} total\n"
        f"Real: {len(real_data)} | Predicted: {len(pred_data)}\n"
        f"Range: {station_values.min():.1f}% - {station_values.max():.1f}%\n"
        f"Mean: {station_values.mean():.1f}%"
    )
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

    args = parser.parse_args()

    create_moisture_map(
        model_path=args.model,
        target_date=args.date,
        output_file=args.output,
        device=args.device,
        include_weather_maps=args.include_weather_maps
    )
