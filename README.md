# Moisturizer

A soil moisture prediction system for Galicia, Spain, using weather data from the MeteoGalicia API.

## Overview

Moisturizer provides tools for:

- **Data Collection**: Automated collection of weather and soil moisture data from MeteoGalicia's network of 155+ weather stations
- **Dataset Creation**: PyTorch-ready datasets with configurable sequence lengths, spatial features from nearby stations, and data augmentation
- **Map Visualization**: Beautiful soil moisture maps with real sensor data, model predictions, and virtual grid interpolation

The system is designed to predict soil moisture at locations without sensors by learning spatial and temporal patterns from the 39 stations that have soil moisture sensors.

## Features

### Data Pipeline
- Automatic discovery of stations with soil moisture sensors
- Historical data fetching with caching
- Parameter coverage analysis and filtering
- Dense array preprocessing for fast data access

### Dataset Augmentation
- **Skip Patterns**: Drop one nearby station at a time (5 patterns with 5 available, 4 used)
- **Permutations**: All orderings of nearby stations (24 permutations)
- **Combined**: 120x augmentation factor (5 × 24)
- Both live (on-the-fly) and precomputed augmentation modes

### Map Creation
- Real soil moisture overlay on geographic maps
- Model predictions for stations without sensors
- Virtual grid predictions with feature interpolation
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
train_ds, val_ds, test_ds = buildDataset(
    seq_length=64,           # 64 days of historical data
    days=365,                # 1 year of data
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
    n_nearby_available=5,    # 5 nearby stations available
    n_nearby_in_features=4   # 4 used in model input
)

# Each sample is augmented 120x on-the-fly
print(f"Augmented samples: {len(train_ds)}")
```

### Creating a Soil Moisture Map

```bash
# Basic map with model predictions
python create_moisture_map.py --model path/to/model.pth --date 2025-01-15

# Full map suite with virtual grid
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
| `WeatherSequenceDataset.py` | PyTorch Dataset for weather/soil moisture sequences |
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
Each Nearby: distance + n_params weather + soil_moisture

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

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MeteoGalicia](https://www.meteogalicia.gal/) for providing the weather data API
- [TROLOLO](https://github.com/dcifrian/TROLOLO) transformer architecture for model predictions
