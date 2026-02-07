# Moisturizer

A weather parameter prediction system for Galicia, Spain, using data from the MeteoGalicia API. Originally developed for soil moisture prediction, the dataset system now supports configurable target parameters.

## Overview

Moisturizer provides tools for:

- **Data Collection**: Automated collection of weather data from MeteoGalicia's network of 155+ weather stations
- **Dataset Creation**: PyTorch-ready datasets with configurable sequence lengths, spatial features from nearby stations, and data augmentation
- **Map Visualization**: Soil moisture maps with real sensor data, model predictions, and virtual grid interpolation

The system learns spatial and temporal patterns from stations with the target sensor to predict values at locations without sensors. For soil moisture, 39 stations have sensors with **10+ years of historical data** available.

## Features

### Data Pipeline
- Automatic discovery of stations with target sensors
- Historical data fetching with caching
- Parameter coverage analysis and filtering
- Dense array preprocessing for fast data access

### Configurable Target Parameter
The dataset system supports predicting any weather parameter available in the MeteoGalicia API, not just soil moisture. This is configured via `target_param` in `WeatherSequenceDataset`:

```python
# Default: soil moisture
dataset = WeatherSequenceDataset(..., target_param="HS_CV_AVG_-0.2m")

# Or predict temperature, humidity, etc.
dataset = WeatherSequenceDataset(..., target_param="TA_AVG_1.5m")
```

> **Note**: Multi-target support is experimental and currently only applies to dataset creation. Map visualization is still soil moisture specific.

### Dataset Augmentation

The number of nearby stations is configurable and directly affects augmentation:

- **`n_nearby_in_features`**: Number of nearby stations in the model input (e.g., 4)
- **`n_nearby_available`**: Number of nearby stations available for augmentation (e.g., 5)

When `n_nearby_available > n_nearby_in_features`:
- **Skip Patterns**: Drop one nearby station at a time (creates `n_nearby_available` patterns)
- **Permutations**: All orderings of selected stations (`n_nearby_in_features!` permutations)
- **Combined**: `n_nearby_available × n_nearby_in_features!` augmentation factor

Example with 5 available and 4 used: 5 × 24 = **120x augmentation**

> In testing, 4 nearby stations with a 5th for augmentation works well, but optimal values likely depend on the model architecture. Both higher and lower configurations have been tested.

### Map Creation
- Real soil moisture overlay on geographic maps
- Model predictions for stations without sensors
- Virtual grid predictions with feature interpolation (10,000+ points in ~1 second with TROLOLO)
- Ensemble prediction mode (averaging across augmentations)
- Cumulative precipitation and water balance maps

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

### Optional: TROLOLO Model

For training and inference, you can use the [TROLOLO](https://github.com/dcifrian/TROLOLO) transformer architecture:

```bash
git clone https://github.com/dcifrian/TROLOLO.git
cd TROLOLO
git checkout develop
pip install -e .
```

## Quick Start

### Building a Dataset

```python
from Moisturizer import buildDataset

# Build dataset with default parameters
# Note: 10+ years of soil moisture data available without losing any station
train_ds, val_ds, test_ds = buildDataset(
    seq_length=64,           # 64 days of historical data
    days=365,                # 1 year of data (example; can use 3650+ for full history)
    coverage_threshold=0.25, # Require 25% data coverage
    force_refresh=False      # Use cached data if available
)

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
```

### Loading a Pre-built Dataset

```python
from Moisturizer import loadDataset

# Load with live augmentation (memory efficient)
train_ds, val_ds, test_ds = loadDataset(
    augmented=True,
    n_nearby_available=5,    # 5 nearby stations available for augmentation
    n_nearby_in_features=4   # 4 used in model input
)

# Each sample is augmented 120x on-the-fly (5 skip patterns × 24 permutations)
print(f"Augmented samples: {len(train_ds)}")
```

### Creating a Soil Moisture Map

```bash
# Basic map with model predictions
python create_moisture_map.py --model path/to/model.pth --date 2025-01-15

# High-resolution virtual grid (100×100 = 10,000 points, runs in ~1 second)
python create_moisture_map.py --model model.pth --date 2025-01-15 \
    --virtual-grid 100 --all-maps

# Ensemble prediction (averages multiple augmented predictions)
python create_moisture_map.py --model model.pth --date 2025-01-15 \
    --ensemble --virtual-grid 100
```

## Project Structure

| File | Description |
|------|-------------|
| `Moisturizer.py` | Main entry point with `buildDataset()` and `loadDataset()` |
| `MeteoGaliciaCollector.py` | MeteoGalicia API client for data collection |
| `WeatherSequenceDataset.py` | PyTorch Dataset for weather parameter sequences |
| `augmented_live.py` | Live (on-the-fly) data augmentation |
| `precompute_augmented.py` | Precomputed augmentation with memory-mapped arrays |
| `create_moisture_map.py` | Map visualization and virtual station predictions |
| `model_loader.py` | TROLOLO model loading utilities |
| `tests.py` | Test suite |

## Feature Layout

For each sample, features are organized as:

```
[Target Station Features] + [Nearby Station 1] + ... + [Nearby Station N]

Target: n_params weather parameters
Each Nearby: distance + n_params weather + target_param_value

Total: n_params + n_nearby × (n_params + 2) features per timestep
```

Example with 32 weather parameters and 4 nearby stations:
- Target: 32 features
- Nearby: 4 × 34 = 136 features
- **Total: 168 features per timestep**

## Data Source

Weather data is collected from [MeteoGalicia](https://www.meteogalicia.gal/), the official meteorological agency of Galicia, Spain. The API provides:

- 155+ weather stations across Galicia
- 39 stations with soil moisture sensors (at -0.2m depth)
- 42 different weather parameters
- Daily aggregated measurements
- 10+ years of historical data for soil moisture

## Future Directions

- Full multi-parameter support for map visualization
- Additional data sources beyond MeteoGalicia
- Extended parameter coverage analysis

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MeteoGalicia](https://www.meteogalicia.gal/) for providing the weather data API
- [TROLOLO](https://github.com/dcifrian/TROLOLO) transformer architecture for model predictions
