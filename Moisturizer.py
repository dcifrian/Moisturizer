"""
MeteoGalicia Weather Station Data Collector
Collects historical and live data from MeteoGalicia API for ML model training
with focus on soil moisture prediction from nearby stations
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple, Set, Union
import json
import time
from pathlib import Path
import zipfile

from augmented_live import AugmentedLiveDataset

# Model loader - optional, only needed for inference
try:
    from model_loader import load_model
    MODEL_LOADER_AVAILABLE = True
except ImportError:
    MODEL_LOADER_AVAILABLE = False
    load_model = None  # Will raise error if actually used

# PyTorch imports - optional, only needed for Dataset class
try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. SoilMoistureSequenceDataset will not work.")
    print("         Data collection functionality will still work normally.")


class MeteoGaliciaCollector:
    """Collector for MeteoGalicia weather station data"""

    BASE_URL = "https://servizos.meteogalicia.gal/mgrss/observacion/datosDiariosEstacionsMeteo.action"
    STATIONS_URL = "https://servizos.meteogalicia.gal/mgrss/observacion/listaEstacionsMeteo.action"
    SOIL_MOISTURE_PARAM = "HS_CV_AVG_-0.2m"

    # All available sensors from MeteoGalicia API
    ALL_SENSORS = [
        'TO_AVG_1.5m',      # Dew point temperature at 1.5m
        'TO_AVG_15m',       # Dew point temperature at 15m
        'TS_AVG_-0.1m',     # Soil temperature at -0.1m
        'TA_MAX_1.5m',      # Maximum air temperature at 1.5m
        'TA_MAX_15m',       # Maximum air temperature at 15m
        'TA_AVG_0.1m',      # Average air temperature at 0.1m
        'TA_AVG_15m',       # Average air temperature at 15m
        'TA_AVG_1.5m',      # Average air temperature at 1.5m
        'TA_MIN_1.5m',      # Minimum air temperature at 1.5m
        'TA_MIN_15m',       # Minimum air temperature at 15m
        'HFRIO7_RECUENTO_1.5m',  # Cold hours count at 1.5m
        'HFRIO7_RECUENTO_15m',   # Cold hours count at 15m
        'HR_MAX_15m',       # Maximum relative humidity at 15m
        'HR_MIN_15m',       # Minimum relative humidity at 15m
        'HR_MAX_1.5m',      # Maximum relative humidity at 1.5m
        'HR_MIN_1.5m',      # Minimum relative humidity at 1.5m
        'HR_AVG_15m',       # Average relative humidity at 15m
        'HR_AVG_1.5m',      # Average relative humidity at 1.5m
        'BH_SUM_1.5m',      # Leaf wetness sum at 1.5m
        'PP_SUM_1.5m',      # Precipitation sum at 1.5m
        'ET0_SUM_1.5m',     # Reference evapotranspiration sum at 1.5m
        'DV_CONDICION_10m', # Wind direction condition at 10m
        'DV_CONDICION_2m',  # Wind direction condition at 2m
        'DVP_MODA_10m',     # Wind direction mode at 10m
        'DVP_MODA_2m',      # Wind direction mode at 2m
        'VV_MAX_10m',       # Maximum wind speed at 10m
        'VV_MAX_2m',        # Maximum wind speed at 2m
        'VV_AVG_10m',       # Average wind speed at 10m
        'VV_AVG_2m',        # Average wind speed at 2m
        'BCN_AVG_1.5m',     # Average solar radiation balance at 1.5m
        'BCN_MAX_1.5m',     # Maximum solar radiation balance at 1.5m
        'HSOL_SUM_1.5m',    # Sunshine hours sum at 1.5m
        'IUVX_MAX_1.5m',    # Maximum UV index at 1.5m
        'INS_RATIO_1.5m',   # Insolation ratio at 1.5m
        'IRD_SUM_1.5m',     # Direct solar radiation sum at 1.5m
        'PR_AVG_1.5m',      # Average atmospheric pressure at 1.5m
        'PRED_AVG_1.5m',    # Average reduced pressure at 1.5m
        'HF_SUM_2m',        # Leaf wetness hours sum at 2m
        'HS_CV_AVG_-0.2m',  # Soil moisture average at -0.2m (volumetric)
        'VB_MAX_10m',       # Maximum wind gust at 10m
        'VB_AVG_10m',       # Average wind gust at 10m
        'VB_MIN_10m'        # Minimum wind gust at 10m
    ]

    def __init__(self, data_dir: str = "./meteogalicia_data"):
        self.session = requests.Session()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.stations_file = self.data_dir / "stations_metadata.csv"
        self.nearest_file = self.data_dir / "nearest_stations.csv"
        self.timeseries_file = self.data_dir / "raw_timeseries.csv"

    def get_all_stations(self) -> pd.DataFrame:
        """
        Fetch list of all MeteoGalicia stations

        Returns:
            DataFrame with station metadata
        """
        try:
            response = self.session.get(self.STATIONS_URL, timeout=30)
            response.raise_for_status()
            data = response.json()

            stations = []
            for station in data.get('listaEstacionsMeteo', []):
                stations.append({
                    'station_id': station.get('idEstacion'),
                    'station_name': station.get('estacion'),
                    'municipality': station.get('concello'),
                    'province': station.get('provincia'),
                    'latitude': station.get('lat'),
                    'longitude': station.get('lon'),
                    'altitude': station.get('altitude'),
                    'utmx': station.get('utmx'),
                    'utmy': station.get('utmy')
                })

            return pd.DataFrame(stations)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching stations list: {e}")
            return pd.DataFrame()

    def check_soil_moisture_availability(
            self,
            station_id: int,
            soil_param: str = SOIL_MOISTURE_PARAM
    ) -> bool:
        """
        Check if a station has soil moisture data by testing a recent query

        Args:
            station_id: Station ID to check
            soil_param: Soil moisture parameter code

        Returns:
            True if station has soil moisture data, False otherwise
        """
        # Query yesterday's data to test availability
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%d/%m/%Y')

        params = {
            'idEst': str(station_id),
            'idParam': soil_param,
            'dataIni': yesterday,
            'dataFin': yesterday
        }

        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            # If listDatosDiarios is empty, no soil moisture data
            return len(data.get('listDatosDiarios', [])) > 0
        except:
            return False

    def discover_stations_with_soil_moisture(
            self,
            force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Discover all stations and test which ones have soil moisture data
        Caches results to disk

        Args:
            force_refresh: If True, re-discover even if cache exists

        Returns:
            DataFrame with station metadata and has_soil_moisture flag
        """
        if self.stations_file.exists() and not force_refresh:
            print(f"Loading cached stations from {self.stations_file}")
            return pd.read_csv(self.stations_file)

        print("Discovering all stations...")
        stations_df = self.get_all_stations()

        if stations_df.empty:
            print("Failed to fetch stations list")
            return pd.DataFrame()

        print(f"Found {len(stations_df)} stations. Testing for soil moisture availability...")

        # Test each station for soil moisture
        has_soil_moisture = []
        for idx, row in stations_df.iterrows():
            station_id = row['station_id']
            has_sm = self.check_soil_moisture_availability(station_id)
            has_soil_moisture.append(has_sm)

            status = "✓" if has_sm else "✗"
            print(f"  {status} {station_id}: {row['station_name']}")

            # Be nice to the API
            time.sleep(0.5)

        stations_df['has_soil_moisture'] = has_soil_moisture

        # Convert data types before saving
        # Handle potential None values in coordinates
        stations_df['utmx'] = pd.to_numeric(stations_df['utmx'], errors='coerce')
        stations_df['utmy'] = pd.to_numeric(stations_df['utmy'], errors='coerce')
        stations_df['altitude'] = pd.to_numeric(stations_df['altitude'], errors='coerce')
        stations_df['latitude'] = pd.to_numeric(stations_df['latitude'], errors='coerce')
        stations_df['longitude'] = pd.to_numeric(stations_df['longitude'], errors='coerce')

        # Save to cache
        stations_df.to_csv(self.stations_file, index=False)
        print(f"\n✓ Stations metadata saved to {self.stations_file}")
        print(f"  Total: {len(stations_df)}, With soil moisture: {sum(has_soil_moisture)}")

        return stations_df

    def calculate_nearest_stations(
            self,
            stations_df: pd.DataFrame,
            n_nearest: int = 4
    ) -> pd.DataFrame:
        """
        Calculate n nearest stations WITH SOIL MOISTURE for each station using Euclidean distance on UTM coords

        This finds the nearest stations that have soil moisture data, which is essential for
        predicting/imputing soil moisture at stations that don't have it.

        Args:
            stations_df: DataFrame with station metadata including utmx, utmy, has_soil_moisture
            n_nearest: Number of nearest stations WITH SOIL MOISTURE to find

        Returns:
            DataFrame with columns: station_id, nearest_1_id, nearest_1_distance,
                                   nearest_2_id, nearest_2_distance, ...
        """
        print(f"\nCalculating {n_nearest} nearest stations WITH SOIL MOISTURE for each station...")

        # Create coordinate matrix for distance calculation
        stations_df = stations_df.copy()

        # Filter out stations with invalid coordinates
        valid_coords_mask = stations_df['utmx'].notna() & stations_df['utmy'].notna()
        stations_with_coords = stations_df[valid_coords_mask].copy()

        if len(stations_with_coords) == 0:
            print("✗ No stations with valid coordinates found!")
            return pd.DataFrame()

        # Get stations with soil moisture (these are our candidates for nearest neighbors)
        soil_moisture_stations = stations_with_coords[stations_with_coords['has_soil_moisture'] == True].copy()

        if len(soil_moisture_stations) == 0:
            print("✗ No stations with soil moisture data found!")
            return pd.DataFrame()

        print(f"  Total stations with valid coordinates: {len(stations_with_coords)}")
        print(f"  Stations with soil moisture data: {len(soil_moisture_stations)}")

        # Get coordinates for all stations and for soil moisture stations only
        all_coords = stations_with_coords[['utmx', 'utmy']].values
        all_station_ids = stations_with_coords['station_id'].values

        soil_coords = soil_moisture_stations[['utmx', 'utmy']].values
        soil_station_ids = soil_moisture_stations['station_id'].values

        nearest_data = []

        for i, station_id in enumerate(all_station_ids):
            # Calculate distances from this station to all stations WITH soil moisture
            current_coords = all_coords[i]
            distances = np.sqrt(
                (soil_coords[:, 0] - current_coords[0]) ** 2 +
                (soil_coords[:, 1] - current_coords[1]) ** 2
            )

            # Sort by distance and get N nearest
            # If this station itself has soil moisture, it will be in the list with distance 0
            sorted_indices = np.argsort(distances)

            # Filter out the station itself (distance = 0) if it has soil moisture
            filtered_indices = []
            for idx in sorted_indices:
                if soil_station_ids[idx] != station_id:
                    filtered_indices.append(idx)
                if len(filtered_indices) == n_nearest:
                    break

            row_data = {'station_id': station_id}

            # If we found fewer than n_nearest stations, record what we found
            for j, idx in enumerate(filtered_indices, 1):
                row_data[f'nearest_{j}_id'] = soil_station_ids[idx]
                row_data[f'nearest_{j}_distance'] = distances[idx]
                row_data[f'nearest_{j}_has_soil_moisture'] = True  # Always True by construction

            # Fill remaining slots with NaN if we found fewer than n_nearest
            for j in range(len(filtered_indices) + 1, n_nearest + 1):
                row_data[f'nearest_{j}_id'] = np.nan
                row_data[f'nearest_{j}_distance'] = np.nan
                row_data[f'nearest_{j}_has_soil_moisture'] = False

            nearest_data.append(row_data)

        nearest_df = pd.DataFrame(nearest_data)
        nearest_df.to_csv(self.nearest_file, index=False)
        print(f"✓ Nearest stations saved to {self.nearest_file}")

        return nearest_df

    def get_daily_data(
            self,
            station_ids: Optional[List[int]] = None,
            parameter_ids: Optional[List[str]] = None,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None
    ) -> Dict:
        """
        Fetch daily data from MeteoGalicia API

        Args:
            station_ids: List of station IDs
            parameter_ids: List of parameter codes
            start_date: Start date in format 'dd/MM/yyyy'
            end_date: End date in format 'dd/MM/yyyy'

        Returns:
            Dictionary with the API response
        """
        params = {}

        if station_ids:
            params['idEst'] = ','.join(map(str, station_ids))

        if parameter_ids:
            params['idParam'] = ','.join(parameter_ids)

        if start_date:
            params['dataIni'] = start_date

        if end_date:
            params['dataFin'] = end_date

        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return None

    def parse_data_to_dataframe(self, json_data: Dict) -> pd.DataFrame:
        """
        Parse JSON response into a structured DataFrame

        Args:
            json_data: Raw JSON response from API

        Returns:
            DataFrame in long format: date, station_id, parameter_code, value, unit, validation_code
        """
        if not json_data or 'listDatosDiarios' not in json_data:
            return pd.DataFrame()

        rows = []

        for daily_data in json_data['listDatosDiarios']:
            date = daily_data.get('data')

            for station in daily_data.get('listaEstacions', []):
                station_id = station.get('idEstacion')

                for measure in station.get('listaMedidas', []):
                    rows.append({
                        'date': date,
                        'station_id': station_id,
                        'parameter_code': measure.get('codigoParametro'),
                        'value': measure.get('valor'),
                        'unit': measure.get('unidade'),
                        'validation_code': measure.get('lnCodigoValidacion')
                    })

        df = pd.DataFrame(rows)

        if not df.empty:
            # Let pandas auto-detect the date format (API returns ISO format)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['value'] = pd.to_numeric(df['value'], errors='coerce')

        return df

    def build_historical_dataset(
            self,
            station_ids: List[int],
            parameter_ids: List[str],
            start_date: datetime,
            end_date: datetime,
            chunk_days: int = 30,
            force_refresh = False
    ) -> pd.DataFrame:
        """
        Build historical dataset by fetching data in chunks
        Saves to raw_timeseries.csv

        Args:
            station_ids: List of station IDs to collect
            parameter_ids: List of parameters to collect
            start_date: Start date as datetime object
            end_date: End date as datetime object
            chunk_days: Number of days to fetch per request

        Returns:
            Complete DataFrame with all historical data in long format
        """
        all_data = []
        current_date = start_date
        if self.timeseries_file.exists() and not force_refresh:
            print(f"Loading cached stations from {self.timeseries_file}")
            return pd.read_csv(self.timeseries_file)
        print(f"\nFetching historical data from {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')}")
        print(f"Stations: {len(station_ids)}")
        print(f"Parameters: {parameter_ids}")

        while current_date <= end_date:
            chunk_end = min(current_date + timedelta(days=chunk_days - 1), end_date)

            start_str = current_date.strftime('%d/%m/%Y')
            end_str = chunk_end.strftime('%d/%m/%Y')

            print(f"\nFetching: {start_str} to {end_str}")

            json_data = self.get_daily_data(
                station_ids=station_ids,
                parameter_ids=parameter_ids,
                start_date=start_str,
                end_date=end_str
            )

            if json_data:
                df_chunk = self.parse_data_to_dataframe(json_data)
                if not df_chunk.empty:
                    all_data.append(df_chunk)
                    print(f"  Retrieved {len(df_chunk)} records")

            current_date = chunk_end + timedelta(days=1)
            time.sleep(1)

        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            final_df.to_csv(self.timeseries_file, index=False)
            print(f"\n✓ Raw timeseries saved to {self.timeseries_file}")
            print(f"  Total records: {len(final_df)}")
            print(f"  Date range: {final_df['date'].min()} to {final_df['date'].max()}")
            return final_df
        else:
            print("\n✗ No data collected")
            return pd.DataFrame()

    def create_ml_ready_dataset(
            self,
            target_station_ids: Optional[List[int]] = None,
            n_nearest: int = 4,
            output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create ML-ready dataset from raw timeseries and nearest stations data
        Each row contains target station data + n nearest stations' data

        Args:
            target_station_ids: List of target stations (if None, use all with soil moisture)
            n_nearest: Number of nearest stations to include
            output_file: Output filename (defaults to ml_ready_dataset.csv)

        Returns:
            Wide-format DataFrame ready for ML training
        """
        print("\nCreating ML-ready dataset...")

        # Load necessary data
        stations_df = pd.read_csv(self.stations_file)
        nearest_df = pd.read_csv(self.nearest_file)
        timeseries_df = pd.read_csv(self.timeseries_file)
        timeseries_df['date'] = pd.to_datetime(timeseries_df['date'])

        # Filter to target stations with soil moisture
        if target_station_ids is None:
            target_station_ids = stations_df[stations_df['has_soil_moisture']]['station_id'].tolist()

        print(f"Processing {len(target_station_ids)} target stations...")

        # Pivot timeseries to wide format for easier joining
        timeseries_wide = timeseries_df.pivot_table(
            index=['date', 'station_id'],
            columns='parameter_code',
            values='value',
            aggfunc='first'
        ).reset_index()

        ml_rows = []

        for target_id in target_station_ids:
            # Get target station data
            target_data = timeseries_wide[timeseries_wide['station_id'] == target_id].copy()
            target_data.columns = ['date', 'station_id'] + [f'target_{col}' for col in target_data.columns[2:]]

            # Get nearest stations info
            nearest_info = nearest_df[nearest_df['station_id'] == target_id].iloc[0]

            # Get coordinates for target station
            target_coords = stations_df[stations_df['station_id'] == target_id].iloc[0]

            for i in range(1, n_nearest + 1):
                nearest_id = nearest_info[f'nearest_{i}_id']
                nearest_dist = nearest_info[f'nearest_{i}_distance']

                # Get nearest station data
                nearest_data = timeseries_wide[timeseries_wide['station_id'] == nearest_id].copy()

                if nearest_data.empty:
                    continue

                # Rename columns for this nearest station
                nearest_data = nearest_data.rename(columns={
                    col: f'nearby{i}_{col}'
                    for col in nearest_data.columns if col not in ['date', 'station_id']
                })
                nearest_data = nearest_data.rename(columns={'station_id': f'nearby{i}_id'})

                # Add distance
                nearest_data[f'nearby{i}_distance'] = nearest_dist

                # Merge with target data
                if i == 1:
                    merged = target_data.merge(nearest_data, on='date', how='inner')
                else:
                    merged = merged.merge(nearest_data, on='date', how='inner')

            # Add target station coordinates
            merged['target_utmx'] = target_coords['utmx']
            merged['target_utmy'] = target_coords['utmy']
            merged['target_altitude'] = target_coords['altitude']

            ml_rows.append(merged)

        if ml_rows:
            ml_df = pd.concat(ml_rows, ignore_index=True)

            if output_file is None:
                output_file = self.data_dir / "ml_ready_dataset.csv"

            ml_df.to_csv(output_file, index=False)
            print(f"✓ ML-ready dataset saved to {output_file}")
            print(f"  Total rows: {len(ml_df)}")
            print(f"  Columns: {len(ml_df.columns)}")

            return ml_df
        else:
            print("✗ No data to create ML dataset")
            return pd.DataFrame()

    def analyze_parameter_coverage(
            self,
            timeseries_df: Optional[pd.DataFrame] = None,
            stations_df: Optional[pd.DataFrame] = None,
            coverage_threshold: float = 0.25,
            soil_moisture_param: str = "HS_CV_AVG_-0.2m",
            add_coordinate_features: bool = True,
            force_recompute: bool = False
    ) -> Tuple[Dict[str, float], List[str]]:
        """
        Analyze parameter coverage and return parameters above threshold.

        Results are cached to avoid re-computation (~30s savings). Cache is
        invalidated if coverage_threshold or number of days in data changes.

        Can analyze from either:
        1. timeseries_df + stations_df (for buildDataset - before ml_ready_dataset exists)
        2. ml_ready_dataset.csv (existing functionality)

        Args:
            timeseries_df: Timeseries data
            stations_df: Stations data
            coverage_threshold: Minimum fraction of stations that must have data (0.0 to 1.0)
            soil_moisture_param: Soil moisture parameter to exclude (it's the target, not a feature!)
            add_coordinate_features: If True, add altitude/utmx/utmy to filtered params
            force_recompute: If True, ignore cache and recompute

        Returns:
            Tuple of (coverage_dict, filtered_params):
            - coverage_dict: Dictionary mapping parameter_code to coverage percentage
            - filtered_params: List of parameters that meet the coverage threshold (excluding soil moisture)
        """
        cache_path = self.data_dir / "filtered_params_cache.npz"

        # Try to load from cache first
        if not force_recompute and cache_path.exists():
            try:
                cache = np.load(cache_path, allow_pickle=True)
                cached_threshold = float(cache['coverage_threshold'])
                cached_n_days = int(cache['n_days'])
                cached_add_coords = bool(cache['add_coordinate_features'])

                # Get current n_days from timeseries
                if timeseries_df is None:
                    ts_file = self.data_dir / "raw_timeseries.csv"
                    if ts_file.exists():
                        # Quick check: count unique dates without loading full DataFrame
                        current_n_days = len(pd.read_csv(ts_file, usecols=['date'])['date'].unique())
                    else:
                        current_n_days = -1
                else:
                    current_n_days = timeseries_df['date'].nunique()

                # Validate cache
                if (cached_threshold == coverage_threshold and
                    cached_n_days == current_n_days and
                    cached_add_coords == add_coordinate_features):

                    filtered_params = list(cache['filtered_params'])
                    coverage = dict(zip(cache['coverage_params'], cache['coverage_values']))
                    print(f"✓ Loaded cached filtered_params ({len(filtered_params)} params, "
                          f"{cached_n_days} days, {cached_threshold*100:.0f}% threshold)")
                    return coverage, filtered_params
                else:
                    print(f"Cache invalidated: threshold={cached_threshold}→{coverage_threshold}, "
                          f"days={cached_n_days}→{current_n_days}, coords={cached_add_coords}→{add_coordinate_features}")
            except Exception as e:
                print(f"Could not load cache: {e}")

        # Load data if not provided
        if timeseries_df is None:
            timeseries_df = pd.read_csv(self.data_dir / "raw_timeseries.csv")
        if stations_df is None:
            stations_df = pd.read_csv(self.data_dir / "stations_metadata.csv")

        print(f"\nAnalyzing parameter coverage from timeseries data...")

        # Get stations with soil moisture
        soil_moisture_stations = stations_df[stations_df['has_soil_moisture']]['station_id'].tolist()
        all_params = timeseries_df['parameter_code'].unique()
        n_days = timeseries_df['date'].nunique()

        print(f"Analyzing {len(all_params)} parameters on {len(soil_moisture_stations)} stations with soil moisture...")
        print(f"Coverage threshold: {coverage_threshold * 100:.0f}%")
        print(f"\nParameter coverage:")
        print("-" * 70)

        coverage = {}
        filtered_params = []

        for param in all_params:
            if param == soil_moisture_param:
                continue  # Skip soil moisture - it's the target

            # Count how many soil moisture stations have this parameter
            param_data = timeseries_df[
                (timeseries_df['parameter_code'] == param) &
                (timeseries_df['station_id'].isin(soil_moisture_stations))
            ]

            stations_with_param = param_data['station_id'].nunique()
            coverage_pct = stations_with_param / len(soil_moisture_stations) if soil_moisture_stations else 0
            coverage[param] = coverage_pct

            status = "✓" if coverage_pct >= coverage_threshold else "✗"
            print(f"{status} {param:30s}: {coverage_pct*100:5.1f}% ({stations_with_param}/{len(soil_moisture_stations)} stations)")

            if coverage_pct >= coverage_threshold:
                filtered_params.append(param)

        print("-" * 70)
        print(f"\nParameters passing {coverage_threshold*100:.0f}% threshold: {len(filtered_params)}/{len(all_params)}")

        # Add coordinate features if requested
        if add_coordinate_features:
            coordinate_features = ['altitude', 'utmx', 'utmy']
            filtered_params.extend(coordinate_features)
            print(f"✓ Added {len(coordinate_features)} coordinate features: {coordinate_features}")

        print(f"\nTotal filtered parameters: {len(filtered_params)}")

        # Save to cache
        try:
            np.savez(
                cache_path,
                filtered_params=np.array(filtered_params, dtype='U50'),
                coverage_params=np.array(list(coverage.keys()), dtype='U50'),
                coverage_values=np.array(list(coverage.values()), dtype=np.float32),
                coverage_threshold=np.array([coverage_threshold]),
                n_days=np.array([n_days]),
                add_coordinate_features=np.array([add_coordinate_features]),
            )
            print(f"✓ Cached filtered_params to {cache_path}")
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")

        return coverage, filtered_params

    def get_live_prediction_data(
            self,
            target_station_id: int,
            n_nearest: int = 4,
            parameter_ids: Optional[List[str]] = None
    ) -> Dict:
        """
        Get live data for prediction: target station data + n nearest stations with soil moisture

        Args:
            target_station_id: Station ID to predict for
            n_nearest: Number of nearest stations to include
            parameter_ids: Parameters to fetch (if None, uses common parameters)

        Returns:
            Dictionary with target_data and nearby_stations_data
        """
        if parameter_ids is None:
            parameter_ids = [
                self.SOIL_MOISTURE_PARAM,
                'PP_SUM_1.5m',  # Precipitation
                'TA_AVG_1.5m',  # Temperature
                'HR_AVG_1.5m'  # Humidity
            ]

        # Load stations metadata and nearest stations
        stations_df = pd.read_csv(self.stations_file)
        nearest_df = pd.read_csv(self.nearest_file)

        # Get target station info
        target_info = stations_df[stations_df['station_id'] == target_station_id].iloc[0]

        # Get nearest stations info
        nearest_info = nearest_df[nearest_df['station_id'] == target_station_id].iloc[0]

        # Find n nearest stations WITH soil moisture
        nearby_with_soil = []
        for i in range(1, len(nearest_df.columns) // 3 + 1):  # Iterate through all nearest stations
            if f'nearest_{i}_id' not in nearest_info:
                break
            if nearest_info[f'nearest_{i}_has_soil_moisture']:
                nearby_with_soil.append({
                    'station_id': int(nearest_info[f'nearest_{i}_id']),
                    'distance': nearest_info[f'nearest_{i}_distance']
                })
                if len(nearby_with_soil) == n_nearest:
                    break

        # Fetch target station data
        target_json = self.get_daily_data(
            station_ids=[target_station_id],
            parameter_ids=parameter_ids
        )
        target_df = self.parse_data_to_dataframe(target_json) if target_json else pd.DataFrame()

        # Fetch nearby stations data
        nearby_ids = [s['station_id'] for s in nearby_with_soil]
        nearby_json = self.get_daily_data(
            station_ids=nearby_ids,
            parameter_ids=parameter_ids
        )
        nearby_df = self.parse_data_to_dataframe(nearby_json) if nearby_json else pd.DataFrame()

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
            'target_data': target_df,
            'nearby_stations': nearby_with_soil,
            'nearby_data': nearby_df
        }


# Only define Dataset class if PyTorch is available
if TORCH_AVAILABLE:
    _BaseDataset = Dataset
else:
    _BaseDataset = object


def _decompress_npz_if_needed(npz_path: str) -> str:
    """
    Check if NPZ file is compressed and decompress it for true memory-mapping.

    Compressed NPZ files cause memory issues: even with mmap_mode='r',
    accessing arrays decompresses the entire array into RAM.

    Returns path to uncompressed NPZ file.
    """
    npz_path = Path(npz_path)

    if not npz_path.exists():
        return str(npz_path)

    # Check if compressed by trying to open as ZIP
    try:
        with zipfile.ZipFile(npz_path, 'r') as zf:
            is_compressed = True
    except zipfile.BadZipFile:
        # Not a ZIP = already uncompressed
        return str(npz_path)

    # Check for existing uncompressed version
    uncompressed_path = npz_path.parent / f"{npz_path.stem}_uncompressed.npz"
    if uncompressed_path.exists():
        return str(uncompressed_path)

    # Need to decompress
    print(f"  Compressed NPZ detected - decompressing for memory efficiency...")
    print(f"  This is a one-time operation...")

    # Load and decompress
    data = np.load(npz_path, allow_pickle=False)
    arrays_dict = {}
    for key in data.keys():
        arrays_dict[key] = np.array(data[key])
    data.close()

    # Save uncompressed
    np.savez(uncompressed_path, **arrays_dict)

    # Free memory
    del arrays_dict
    import gc
    gc.collect()

    print(f"  ✓ Created: {uncompressed_path.name}")
    return str(uncompressed_path)


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


class SoilMoistureSequenceDataset(_BaseDataset):
    """
    PyTorch Dataset for soil moisture prediction with temporal sequences
    Suitable for transformer models

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
            soil_moisture_param: str = "HS_CV_AVG_-0.2m",
            missing_value: float = -1000.0,
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
            target_stations: List of station IDs to use (if None, use all with soil moisture)
            feature_params: List of parameter codes to include as features
                           (if None, uses all except soil moisture for target station)
                           Tip: Use analyze_parameter_coverage() to get filtered params
            soil_moisture_param: Parameter code for soil moisture (target variable)
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
        self.soil_moisture_param = soil_moisture_param
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
            if 'feature_params' in self.precomputed_data:
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
            else:
                # No feature_params in precomputed data (old format)
                if feature_params is None:
                    # Use all parameters except soil moisture
                    all_params = self.timeseries_df['parameter_code'].unique()
                    self.feature_params = [p for p in all_params if p != soil_moisture_param]
                else:
                    self.feature_params = feature_params

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
                self.feature_params = [p for p in all_params if p != soil_moisture_param]
            else:
                self.feature_params = feature_params

            # Build index of valid samples
            self._build_sample_index()
            self._indices = None

        # Check if soil moisture is in feature_params (data leakage!)
        self.soil_in_features = self.soil_moisture_param in self.feature_params
        if self.soil_in_features:
            print(f"⚠ WARNING: Soil moisture ({self.soil_moisture_param}) found in feature_params!")
            print(f"  This will be filtered out from target station features to prevent data leakage.")
            print(f"  Nearby stations will still have soil moisture as context.")
            self.soil_feature_idx = self.feature_params.index(self.soil_moisture_param)
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
            self.timeseries_df['parameter_code'] == self.soil_moisture_param
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
                if target_val is None or target_val == -9999.0 or target_val == self.missing_value or pd.isna(target_val):
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
                key = (nearby_station_id, date, self.soil_moisture_param)
                soil_idx = nearby_offset + 1 + len(self.feature_params)
                if key in self.timeseries_index:
                    features[t, soil_idx] = self.timeseries_index[key]
                    mask[t, soil_idx] = True

        # Get target (soil moisture at end_date for target station)
        # Use the last date from date_range (already at midnight)
        target_key = (target_station_id, date_range[-1], self.soil_moisture_param)
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
            if self.soil_moisture_param in self.dense_arrays['feature_params']:
                soil_idx_in_dense = self.dense_arrays['feature_params'].index(self.soil_moisture_param)
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
            target_key = (target_station_id, end_date, self.soil_moisture_param)
            target = self.timeseries_index.get(target_key, self.missing_value) if self.timeseries_index else self.missing_value

        return (
            torch.from_numpy(features),
            torch.tensor(target, dtype=torch.float32).unsqueeze(0),
            torch.from_numpy(mask)
        )

    def _compute_norm_stats_from_precomputed(self):
        """
        Compute normalization statistics from precomputed data.

        Computes both per-slot stats (for base dataset) and canonical per-feature-type
        stats (for augmented datasets). The canonical stats are computed by finding
        min/max across ALL nearby station slots for each feature type.
        """
        print("Computing min/max for each feature (excluding invalid values)...")

        n_samples = len(self.precomputed_data['features'])
        seq_length = self.precomputed_data['features'].shape[1]
        n_features = self.precomputed_data['features'].shape[2]

        # Per-slot stats (for base dataset normalization)
        feature_mins = np.full(n_features, np.inf, dtype=np.float32)
        feature_maxs = np.full(n_features, -np.inf, dtype=np.float32)

        # Canonical per-feature-type stats (for augmented datasets)
        n_params = len(self.feature_params)
        nearby_features_per_station = 1 + n_params + 1  # distance + params + soil
        target_feat_mins = np.full(n_params, np.inf, dtype=np.float32)
        target_feat_maxs = np.full(n_params, -np.inf, dtype=np.float32)
        nearby_feat_mins = np.full(nearby_features_per_station, np.inf, dtype=np.float32)
        nearby_feat_maxs = np.full(nearby_features_per_station, -np.inf, dtype=np.float32)

        # Invalid markers to exclude
        invalid_markers = [-9999.0, self.missing_value]

        # Process in batches to save memory
        batch_size = 1000
        for i in range(0, n_samples, batch_size):
            end_i = min(i + batch_size, n_samples)
            features_batch = self.precomputed_data['features'][i:end_i]
            masks_batch = self.precomputed_data['masks'][i:end_i]

            # Per-slot stats
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

            # Canonical target feature stats
            target_feats = features_batch[:, :, :n_params]
            for feat_idx in range(n_params):
                feat_data = target_feats[:, :, feat_idx].ravel()
                valid = feat_data[(feat_data != -1000.0) & (feat_data != -9999.0)]
                if len(valid) > 0:
                    target_feat_mins[feat_idx] = min(target_feat_mins[feat_idx], valid.min())
                    target_feat_maxs[feat_idx] = max(target_feat_maxs[feat_idx], valid.max())

            # Canonical nearby feature stats (across ALL nearby stations)
            nearby_data = features_batch[:, :, n_params:]
            nearby_reshaped = nearby_data.reshape(
                end_i - i, seq_length, self.n_nearest, nearby_features_per_station
            )
            for feat_idx in range(nearby_features_per_station):
                feat_data = nearby_reshaped[:, :, :, feat_idx].ravel()
                valid = feat_data[(feat_data != -1000.0) & (feat_data != -9999.0)]
                if len(valid) > 0:
                    nearby_feat_mins[feat_idx] = min(nearby_feat_mins[feat_idx], valid.min())
                    nearby_feat_maxs[feat_idx] = max(nearby_feat_maxs[feat_idx], valid.max())

        # Compute for target as well
        targets = self.precomputed_data['targets']
        valid_targets = targets.copy()
        for marker in invalid_markers:
            valid_targets = valid_targets[valid_targets != marker]

        target_min = valid_targets.min() if len(valid_targets) > 0 else 0.0
        target_max = valid_targets.max() if len(valid_targets) > 0 else 1.0

        # Store both formats
        self.norm_stats = {
            # Per-slot stats (for base dataset)
            'feature_mins': feature_mins,
            'feature_maxs': feature_maxs,
            'target_min': target_min,
            'target_max': target_max,
            # Canonical per-feature-type stats (for augmented datasets)
            'target_feature_mins': target_feat_mins,
            'target_feature_maxs': target_feat_maxs,
            'nearby_feature_mins': nearby_feat_mins,
            'nearby_feature_maxs': nearby_feat_maxs,
            # Metadata
            'n_params': n_params,
            'n_base_samples': n_samples,
            'seq_length': seq_length,
            'feature_params': self.feature_params,
        }

        print(f"  Feature min range: [{feature_mins.min():.2f}, {feature_mins.max():.2f}]")
        print(f"  Feature max range: [{feature_maxs.min():.2f}, {feature_maxs.max():.2f}]")
        print(f"  Target range: [{target_min:.2f}, {target_max:.2f}]")

    def _apply_normalization(self, features, target, mask):
        """
        Normalize features and target to [-1, 1] range
        Invalid markers (-9999, missing_value) are changed to -2
        """
        invalid_markers = [-9999.0, self.missing_value]
        normalized_invalid_marker = -2.0

        # Normalize features
        for feat_idx in range(features.shape[1]):
            feat_min = self.norm_stats['feature_mins'][feat_idx]
            feat_max = self.norm_stats['feature_maxs'][feat_idx]

            # Handle invalid markers
            invalid_mask = np.zeros(features.shape[0], dtype=bool)
            for marker in invalid_markers:
                invalid_mask |= (features[:, feat_idx] == marker)

            # Normalize valid values to [-1, 1]
            if feat_max > feat_min:
                features[:, feat_idx] = 2.0 * (features[:, feat_idx] - feat_min) / (feat_max - feat_min) - 1.0

            # Set invalid markers to -2
            features[invalid_mask, feat_idx] = normalized_invalid_marker

        # Normalize target
        target_min = self.norm_stats['target_min']
        target_max = self.norm_stats['target_max']

        # Check if target is invalid
        target_invalid = False
        for marker in invalid_markers:
            if np.any(target == marker):
                target_invalid = True
                break

        if target_invalid:
            target[:] = normalized_invalid_marker
        elif target_max > target_min:
            target[:] = 2.0 * (target - target_min) / (target_max - target_min) - 1.0

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
            self._compute_norm_stats_from_precomputed()

            print(f"Normalizing all data...")
            # Normalize all samples in-place
            for idx in tqdm(range(len(self.sample_index)), desc="Normalizing data"):
                all_features[idx], all_targets[idx] = self._apply_normalization(
                    all_features[idx],
                    all_targets[idx],
                    all_masks[idx]
                )

            is_normalized = True

            # Save normalization statistics (canonical format for reuse by augmented datasets)
            if norm_stats_path:
                print(f"Saving normalization stats to {norm_stats_path}...")
                np.savez(
                    norm_stats_path,
                    # Per-slot stats (for base dataset)
                    feature_mins=self.norm_stats['feature_mins'],
                    feature_maxs=self.norm_stats['feature_maxs'],
                    target_min=np.array([self.norm_stats['target_min']]),
                    target_max=np.array([self.norm_stats['target_max']]),
                    # Canonical per-feature-type stats (for augmented datasets)
                    target_feature_mins=self.norm_stats['target_feature_mins'],
                    target_feature_maxs=self.norm_stats['target_feature_maxs'],
                    nearby_feature_mins=self.norm_stats['nearby_feature_mins'],
                    nearby_feature_maxs=self.norm_stats['nearby_feature_maxs'],
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
            feature_names.append(f'nearby{n_idx + 1}_{self.soil_moisture_param}')

        return feature_names

    def _split_precomputed(
            self,
            train_stations: List[int],
            val_stations: List[int],
            test_stations: List[int]
    ) -> Tuple['SoilMoistureSequenceDataset', 'SoilMoistureSequenceDataset', 'SoilMoistureSequenceDataset']:
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
    ) -> 'SoilMoistureSequenceDataset':
        """Create a dataset from filtered precomputed data"""
        # Create new dataset instance
        split_dataset = SoilMoistureSequenceDataset.__new__(SoilMoistureSequenceDataset)

        # Copy basic attributes
        split_dataset.seq_length = self.seq_length
        split_dataset.n_nearest = self.n_nearest
        split_dataset.soil_moisture_param = self.soil_moisture_param
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
            dataset: 'SoilMoistureSequenceDataset',
            val_stations_ratio: float = 0.15,
            test_stations_ratio: float = 0.0,
            random_seed: int = 42
    ) -> Tuple['SoilMoistureSequenceDataset', 'SoilMoistureSequenceDataset', 'SoilMoistureSequenceDataset']:
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
            train_dataset = SoilMoistureSequenceDataset(
                timeseries=dataset.timeseries_df,
                stations=dataset.stations_df,
                nearest=dataset.nearest_df,
                seq_length=dataset.seq_length,
                n_nearest=dataset.n_nearest,
                target_stations=train_stations,
                feature_params=dataset.feature_params,
                soil_moisture_param=dataset.soil_moisture_param,
                missing_value=dataset.missing_value,
                normalize=dataset.normalize
            )
            val_dataset = None
            if val_stations_ratio > 0:
                val_dataset = SoilMoistureSequenceDataset(
                    timeseries=dataset.timeseries_df,
                    stations=dataset.stations_df,
                    nearest=dataset.nearest_df,
                    seq_length=dataset.seq_length,
                    n_nearest=dataset.n_nearest,
                    target_stations=val_stations,
                    feature_params=dataset.feature_params,
                    soil_moisture_param=dataset.soil_moisture_param,
                    missing_value=dataset.missing_value,
                    normalize=dataset.normalize
                )
            test_dataset = None
            if test_stations_ratio > 0:
                test_dataset = SoilMoistureSequenceDataset(
                    timeseries=dataset.timeseries_df,
                    stations=dataset.stations_df,
                    nearest=dataset.nearest_df,
                    seq_length=dataset.seq_length,
                    n_nearest=dataset.n_nearest,
                    target_stations=test_stations,
                    feature_params=dataset.feature_params,
                    soil_moisture_param=dataset.soil_moisture_param,
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

        # Load stations metadata and nearest stations
        stations_df = pd.read_csv(self.stations_file)
        nearest_df = pd.read_csv(self.nearest_file)

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

def build_dense_feature_array(
    timeseries_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    feature_params: List[str],
    soil_moisture_param: str = "HS_CV_AVG_-0.2m",
    missing_value: float = -1000.0
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
    from tqdm import tqdm

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


def buildDataset(seq_length: int = 64, days: int = 3705):
    """
    Build the complete dataset with optimizations

    Args:
        seq_length: Number of days in each sequence (default: 64)
        days: Number of days of historical data to collect (default: 3705 = ~10 years,
              maximum range without losing stations with moisture data)

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    collector = MeteoGaliciaCollector()

    # Step 1: Discover stations with soil moisture
    print("=" * 60)
    print("STEP 1: Discovering stations")
    print("=" * 60)
    stations_df = collector.discover_stations_with_soil_moisture(force_refresh=False)

    # Step 2: Calculate nearest stations
    print("\n" + "=" * 60)
    print("STEP 2: Calculating nearest stations")
    print("=" * 60)
    nearest_df = collector.calculate_nearest_stations(stations_df, n_nearest=4)

    # Step 3: Build historical dataset
    print("\n" + "=" * 60)
    print("STEP 3: Building historical dataset")
    print("=" * 60)

    # Get all stations (not just those with soil moisture)
    all_station_ids = stations_df['station_id'].tolist()

    # Use ALL sensors from MeteoGalicia API
    # This includes all 42 available parameters (temperature, humidity, wind, solar radiation, etc.)
    parameters = collector.ALL_SENSORS

    # Collect specified days of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    timeseries_df = collector.build_historical_dataset(
        station_ids=all_station_ids,
        parameter_ids=parameters,
        start_date=start_date,
        end_date=end_date,
        chunk_days=30
    )

    # Step 4: Analyze parameter coverage from timeseries
    print("\n" + "=" * 60)
    print("STEP 4: Analyzing parameter coverage")
    print("=" * 60)

    # Use analyze_parameter_coverage() method to get filtered parameters
    coverage_dict, filtered_params = collector.analyze_parameter_coverage(
        timeseries_df=timeseries_df,
        stations_df=stations_df,
        coverage_threshold=0.25,
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
        missing_value=-1000.0
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
        n_nearest=4,
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

def precomputeDataset():
    """Precompute dataset sequences and save to disk for fast loading"""
    collector = MeteoGaliciaCollector()

    print("=" * 60)
    print("PRECOMPUTING DATASET SEQUENCES")
    print("=" * 60)
    print("This will take a while but only needs to be done once.")
    print("Subsequent loads will be MUCH faster!")
    print("=" * 60)

    # Get filtered parameters
    _, filtered_params = collector.analyze_parameter_coverage(coverage_threshold=0.25)

    if not filtered_params:
        print("\n✗ No parameters passed the threshold!")
        return

    print(f"\nUsing {len(filtered_params)} filtered parameters...")

    # Create dataset without precomputed data
    dataset = SoilMoistureSequenceDataset(
        timeseries=str(collector.timeseries_file),
        stations=str(collector.stations_file),
        nearest=str(collector.nearest_file),
        seq_length=64,
        n_nearest=4,
        feature_params=filtered_params,
        normalize=False  # Don't normalize yet, we'll compute stats during precomputation
    )

    # Precompute and save
    precomputed_path = collector.data_dir / "precomputed_sequences"  # Directory, not .npz
    norm_stats_path = collector.data_dir / "normalization_stats.npz"

    dataset.precompute_and_save(
        output_path=str(precomputed_path),
        norm_stats_path=str(norm_stats_path)
    )

    print(f"\n✓ Precomputed sequences saved to: {precomputed_path}/")
    print(f"✓ Normalization stats saved to: {norm_stats_path}")
    print(f"\nYou can now use loadDataset() for fast loading!")

def loadDataset(use_precomputed=True, normalize=True , precomputed_path = None, norm_stats_path = None):
    """
    Load PyTorch Dataset

    Args:
        use_precomputed: If True, load from precomputed file (much faster)
        normalize: If True, normalize data to [-1, 1] range
    """
    collector = MeteoGaliciaCollector() # Does nothing, just for the paths
    print("\n" + "=" * 60)
    print("STEP 1: Loading PyTorch Dataset")
    print("=" * 60)

    # Get filtered parameters
    _, filtered_params = collector.analyze_parameter_coverage(coverage_threshold=0.25)

    if not filtered_params:
        print("\n✗ No parameters passed the threshold!")
        return

    print(f"\nUsing {len(filtered_params)} filtered parameters...")

    # Check for precomputed data
    if precomputed_path is None:
        precomputed_path = collector.data_dir / "precomputed_sequences"
    if norm_stats_path is None:
        norm_stats_path = collector.data_dir / "normalization_stats.npz"

    if use_precomputed and not precomputed_path.exists():
        print(f"\n⚠ Precomputed data not found at {precomputed_path}")
        print("  Run precomputeDataset() first for much faster loading!")
        print("  Falling back to on-the-fly sequence building (SLOW)...")
        use_precomputed = False

    dataset = SoilMoistureSequenceDataset(
        timeseries=str(collector.timeseries_file),
        stations=str(collector.stations_file),
        nearest=str(collector.nearest_file),
        seq_length=64,
        n_nearest=4,
        feature_params=filtered_params,
        precomputed_path=str(precomputed_path) if use_precomputed else None,
        normalize=normalize,
        norm_stats_path=str(norm_stats_path) if normalize else None
    )

    print(f"\nFeature names: {dataset.get_feature_names()}")
    torch.set_printoptions(threshold=1000000)
    # Example sample
    sample = dataset[10]
    print(f"\nExample sample:")
    print(sample)
    print(f"  Features shape: {sample['features'].shape}")
    print(f"  Target: {sample['target']}")
    print(f"  Mask shape: {sample['mask'].shape}")
    print(f"  Station ID: {sample['target_station_id']}")

    # Train/val/test split
    print("\n" + "=" * 60)
    print("STEP 2: Creating train/val/test splits")
    print("=" * 60)

    train_ds, val_ds, test_ds = SoilMoistureSequenceDataset.train_val_test_split(
        dataset,
        val_stations_ratio=0.15,
        test_stations_ratio=0.0
    )
    return train_ds, val_ds, test_ds

def loadDatasetLiveAugmented():
    collector = MeteoGaliciaCollector() # Does nothing, just for the paths
    print("\n" + "=" * 60)
    print("STEP 1: Loading PyTorch Dataset")
    print("=" * 60)

    # Get filtered parameters
    _, filtered_params = collector.analyze_parameter_coverage(coverage_threshold=0.25)

    if not filtered_params:
        print("\n✗ No parameters passed the threshold!")
        return

    print(f"\nUsing {len(filtered_params)} filtered parameters...")
    dataset = AugmentedLiveDataset.from_base_dataset(
        timeseries=str(collector.timeseries_file),
        stations=str(collector.stations_file),
        nearest=str(collector.nearest_file),
        dense_array_path=str(collector.data_dir / "dense_features.npz"),
        feature_params=filtered_params,
        seq_length=64,
        n_nearby_available=5,
        n_nearby_in_features=4,
        normalize=True,
    )
    # Train/val/test split
    print("\n" + "=" * 60)
    print("STEP 2: Creating train/val/test splits")
    print("=" * 60)

    train_ds, val_ds, test_ds = AugmentedLiveDataset.train_val_test_split(
        dataset,
        val_stations_ratio=0.15,
        test_stations_ratio=0.0
    )
    return train_ds, val_ds, test_ds

if __name__ == "__main2__":
    buildDataset()

# Example usage
if __name__ == "__main__":
    # Check if precomputed data exists
    collector = MeteoGaliciaCollector()
    precomputed_path = collector.data_dir / "precomputed_sequences_augmented"
    norm_stats_path = collector.data_dir / "normalization_stats_augmented.npz"
    features_file = precomputed_path / "features.npy"

    if not features_file.exists():
        print("\n" + "=" * 60)
        print("FIRST TIME SETUP: Precomputing dataset sequences")
        print("=" * 60)
        print("This will take ~30-60 minutes but only needs to be done once!")
        print("Subsequent runs will be MUCH faster (10,000+ samples/sec)")
        print("=" * 60)
        input("Press Enter to start precomputation (or Ctrl+C to cancel)...")
        precomputeDataset()
        print("\n" + "=" * 60)
        print("✓ Precomputation complete! Starting training...")
        print("=" * 60)

    #train_ds, val_ds, _ = loadDataset(use_precomputed=True, normalize=True,precomputed_path=precomputed_path,norm_stats_path=norm_stats_path)
    train_ds, val_ds, _ = loadDatasetLiveAugmented()

    # After loading dataset, check one sample:
    sample = train_ds[0]
    features = sample['features']  # [64, total_features]
    target = sample['target']  # [1]

    print(f"\n=== Data Leakage Check ===")
    print(f"Feature shape: {features.shape}")
    print(f"Target value: {target.item():.4f}")

    # Check if target value appears anywhere in the target station features. Nearby stations may happen to have equal or similar values but that is ok.
    # (it shouldn't if there's no leakage!)
    features_np = features.numpy()
    matches = (np.abs(features_np - target.item())[:,:26] < 0.001).sum()
    print(f"Features matching target value: {matches}")

    if matches > 0:
        print("⚠️  LEAKAGE DETECTED: Target value found in features!")
    else:
        print("✓ No obvious leakage detected")

    # Check last timestep specifically (most likely leak point)
    last_timestep = features_np[-1, :]
    last_matches = (np.abs(last_timestep - target.item()) < 0.001).sum()
    print(f"Last timestep matches: {last_matches}")

    trololo = load_model()
    trololo.training_loop(train_data=train_ds,val_data=val_ds,lr=4.1e-4,lr_mid=4.0e-4,lr_min=3e-5,n_epochs=20,batch_size=512,transfer=0)