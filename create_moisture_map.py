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

    # You'll need to recreate the model with same architecture
    # Adjust these parameters to match your trained model!
    model = TROLOLO(
        seq_length=64,
        num_layers=6,
        num_heads=48,
        embed_dim=192,
        mlp_dim=192,
        n_class_tokens=2,
        num_classes=1,
        mlp_rank=0.05,
        qkv_rank=0.05,
        attnproj_rank=0.05,
        sequence_pyramid=[],  # Empty if you removed shenanigans!
        attn_rank_pyramid=[(0, 32), (1, 32)],
        rank_pyramid_begin=2,
        rank_pyramid_factor=1.0,
        head_constriction="ONE_CLASS_TOKEN",
        dropout=0.0,  # No dropout for inference
        attention_dropout=0.0,
        quantize_bits=None
    )

    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully")
    return model


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

    # Run inference
    with torch.no_grad():
        features = sample['features'].unsqueeze(0).to(device)  # [1, seq_len, features]
        prediction = model(features)
        pred_value = prediction.cpu().item()

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


def create_moisture_map(
    model_path,
    target_date=None,
    output_file='galicia_moisture_map.png',
    device='cuda'
):
    """
    Create a beautiful moisture map of all Galicia

    Args:
        model_path: Path to trained model checkpoint
        target_date: Date to predict (default: most recent available)
        output_file: Path to save the visualization
        device: 'cuda' or 'cpu'
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
        precomputed_path=str(collector.data_dir / "precomputed_sequences.npz"),
        normalize=True,
        norm_stats_path=str(collector.data_dir / "normalization_stats.npz")
    )

    # Get moisture values for all stations
    print("\nGathering soil moisture data...")
    results = []

    for _, station in stations_df.iterrows():
        station_id = station['station_id']
        lat = station['latitude']
        lon = station['longitude']
        has_sensor = station['has_soil_moisture']

        if has_sensor:
            # Get real data
            moisture = get_real_soil_moisture(collector, station_id, target_date)
            if moisture is not None:
                results.append({
                    'station_id': station_id,
                    'latitude': lat,
                    'longitude': lon,
                    'moisture': moisture,
                    'type': 'real',
                    'name': station.get('name', f'Station {station_id}')
                })
        else:
            # Predict using model
            try:
                pred_normalized = predict_for_station(model, dataset, station_id, target_date, device)
                if pred_normalized is not None:
                    pred_denorm = denormalize_soil_moisture(
                        pred_normalized,
                        str(collector.data_dir / "normalization_stats.npz")
                    )
                    results.append({
                        'station_id': station_id,
                        'latitude': lat,
                        'longitude': lon,
                        'moisture': pred_denorm,
                        'type': 'predicted',
                        'name': station.get('name', f'Station {station_id}')
                    })
            except Exception as e:
                print(f"  Warning: Could not predict for station {station_id}: {e}")
                continue

    results_df = pd.DataFrame(results)
    print(f"\n✓ Collected data for {len(results_df)} stations")
    print(f"  - Real: {(results_df['type'] == 'real').sum()}")
    print(f"  - Predicted: {(results_df['type'] == 'predicted').sum()}")

    # Create visualization
    print(f"\nCreating visualization...")
    create_visualization(results_df, target_date, output_file)

    print(f"\n✓ Map saved to {output_file}")
    print("=" * 60)

    return results_df


def create_visualization(results_df, target_date, output_file):
    """Create beautiful moisture map visualization"""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Custom colormap: dry (brown) -> moist (green) -> wet (blue)
    colors = ['#8B4513', '#D2691E', '#F4A460', '#90EE90', '#32CD32', '#228B22', '#4682B4', '#1E90FF']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('moisture', colors, N=n_bins)

    # Get coordinate bounds
    lon_min, lon_max = results_df['longitude'].min(), results_df['longitude'].max()
    lat_min, lat_max = results_df['latitude'].min(), results_df['latitude'].max()

    # Add padding
    lon_pad = (lon_max - lon_min) * 0.1
    lat_pad = (lat_max - lat_min) * 0.1

    # Create interpolation grid
    grid_lon = np.linspace(lon_min - lon_pad, lon_max + lon_pad, 300)
    grid_lat = np.linspace(lat_min - lat_pad, lat_max + lat_pad, 300)
    grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)

    # Interpolate moisture across the region
    points = results_df[['longitude', 'latitude']].values
    values = results_df['moisture'].values
    grid_moisture = griddata(points, values, (grid_lon_mesh, grid_lat_mesh), method='cubic')

    # Plot interpolated background
    moisture_plot = ax.contourf(
        grid_lon_mesh, grid_lat_mesh, grid_moisture,
        levels=20, cmap=cmap, alpha=0.6, extend='both'
    )

    # Plot station points
    real_data = results_df[results_df['type'] == 'real']
    pred_data = results_df[results_df['type'] == 'predicted']

    # Real data: solid circles
    scatter_real = ax.scatter(
        real_data['longitude'], real_data['latitude'],
        c=real_data['moisture'], s=100, cmap=cmap,
        edgecolors='black', linewidths=2,
        marker='o', label='Real data (sensors)',
        vmin=values.min(), vmax=values.max(), zorder=10
    )

    # Predicted data: triangles
    if len(pred_data) > 0:
        scatter_pred = ax.scatter(
            pred_data['longitude'], pred_data['latitude'],
            c=pred_data['moisture'], s=100, cmap=cmap,
            edgecolors='gray', linewidths=1.5,
            marker='^', label='Predicted (no sensor)',
            vmin=values.min(), vmax=values.max(), zorder=9
        )

    # Add colorbar
    cbar = plt.colorbar(moisture_plot, ax=ax, pad=0.02)
    cbar.set_label('Soil Moisture (%)', fontsize=12, weight='bold')

    # Labels and title
    ax.set_xlabel('Longitude', fontsize=12, weight='bold')
    ax.set_ylabel('Latitude', fontsize=12, weight='bold')
    ax.set_title(
        f'Galicia Soil Moisture Map\n{target_date.strftime("%Y-%m-%d")}',
        fontsize=16, weight='bold', pad=20
    )

    # Legend
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Stats text box
    stats_text = (
        f"Stations: {len(results_df)} total\n"
        f"Real: {len(real_data)} | Predicted: {len(pred_data)}\n"
        f"Range: {values.min():.1f}% - {values.max():.1f}%\n"
        f"Mean: {values.mean():.1f}%"
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
    parser.add_argument('--model', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--date', type=str, default=None, help='Target date (YYYY-MM-DD), default: most recent')
    parser.add_argument('--output', type=str, default='galicia_moisture_map.png', help='Output file path')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')

    args = parser.parse_args()

    create_moisture_map(
        model_path=args.model,
        target_date=args.date,
        output_file=args.output,
        device=args.device
    )
