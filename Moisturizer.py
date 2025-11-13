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
            ml_dataset_file: Optional[str] = None,
            coverage_threshold: float = 0.25,
            soil_moisture_param: str = "HS_CV_AVG_-0.2m"
    ) -> Tuple[Dict[str, float], List[str]]:
        """
        Analyze parameter coverage in the ML-ready dataset and return parameters above threshold

        Args:
            ml_dataset_file: Path to ml_ready_dataset.csv (if None, uses default location)
            coverage_threshold: Minimum fraction of stations that must have data (0.0 to 1.0)
            soil_moisture_param: Soil moisture parameter to exclude (it's the target, not a feature!)

        Returns:
            Tuple of (coverage_dict, filtered_params):
            - coverage_dict: Dictionary mapping parameter_code to coverage percentage
            - filtered_params: List of parameters that meet the coverage threshold (excluding soil moisture)
        """
        if ml_dataset_file is None:
            ml_dataset_file = self.data_dir / "ml_ready_dataset.csv"

        print(f"\nAnalyzing parameter coverage in {ml_dataset_file}...")

        # Load ML-ready dataset
        ml_df = pd.read_csv(ml_dataset_file)

        # Get all parameter columns (those starting with 'target_' or 'nearby')
        param_columns = [col for col in ml_df.columns if col.startswith('target_') or col.startswith('nearby')]

        # Extract unique parameter names
        # For 'target_PARAM' or 'nearby1_PARAM', extract 'PARAM'
        param_names = set()
        for col in param_columns:
            if col.startswith('target_'):
                param_name = col.replace('target_', '')
                param_names.add(param_name)
            elif '_' in col:  # nearby1_PARAM format
                parts = col.split('_', 1)
                if len(parts) == 2:
                    param_name = parts[1]
                    # Skip distance and has_soil_moisture columns
                    if param_name not in ['distance', 'has_soil_moisture', 'id']:
                        param_names.add(param_name)

        # Calculate coverage for each parameter
        coverage = {}
        total_rows = len(ml_df)

        print(f"\nTotal stations in dataset: {total_rows}")
        print(f"Coverage threshold: {coverage_threshold * 100:.0f}%")
        print(f"\nParameter coverage:")
        print("-" * 70)

        for param in sorted(param_names):
            # Count rows where this parameter has non-null data in target column
            target_col = f'target_{param}'
            if target_col in ml_df.columns:
                non_null_count = ml_df[target_col].notna().sum()
                coverage_pct = non_null_count / total_rows if total_rows > 0 else 0
                coverage[param] = coverage_pct
                status = "✓" if coverage_pct >= coverage_threshold else "✗"
                print(f"{status} {param:25s}: {coverage_pct*100:5.1f}% ({non_null_count}/{total_rows} stations)")

        # Filter parameters that meet threshold (excluding soil moisture - it's the target!)
        filtered_params = [
            param for param, cov in coverage.items()
            if cov >= coverage_threshold and param != soil_moisture_param
        ]

        print("-" * 70)
        print(f"\nParameters passing {coverage_threshold*100:.0f}% threshold: {len(filtered_params)}/{len(param_names)}")
        if soil_moisture_param in coverage:
            print(f"  (Excluded {soil_moisture_param} - it's the target variable)")
        print(f"Filtered parameters: {sorted(filtered_params)}")

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


def _get_uncompressed_npz_path(npz_path: str) -> str:
    """
    Get path to uncompressed version of NPZ file, decompressing if necessary.

    Compressed NPZ files cause zlib errors with multiple DataLoader workers.
    This function ensures we always use uncompressed NPZ for memory-mapping.

    Args:
        npz_path: Path to potentially compressed NPZ file

    Returns:
        Path to uncompressed NPZ file (may decompress if needed)
    """
    npz_path = Path(npz_path)

    if not npz_path.exists():
        return str(npz_path)  # Return as-is, let caller handle missing file

    # Check if file is compressed by checking if it's a valid ZIP archive
    is_compressed = False
    try:
        with zipfile.ZipFile(npz_path, 'r') as zf:
            # If we can open it as a ZIP, it's compressed
            is_compressed = True
    except zipfile.BadZipFile:
        # Not a ZIP file, so it's uncompressed
        is_compressed = False

    if not is_compressed:
        # Already uncompressed, use as-is
        return str(npz_path)

    # File is compressed - look for or create uncompressed version
    uncompressed_path = npz_path.parent / f"{npz_path.stem}_uncompressed.npz"

    if uncompressed_path.exists():
        print(f"  Using existing uncompressed file: {uncompressed_path.name}")
        return str(uncompressed_path)

    # Need to decompress
    print(f"  Compressed NPZ detected: {npz_path.name}")
    print(f"  Decompressing to avoid zlib errors with multiple workers...")
    print(f"  This is a one-time operation and may take a few minutes...")

    # Load compressed data
    data = np.load(npz_path)
    arrays_dict = {key: data[key] for key in data.keys()}

    # Save uncompressed
    np.savez(uncompressed_path, **arrays_dict)
    data.close()

    compressed_size = npz_path.stat().st_size / 1e9
    uncompressed_size = uncompressed_path.stat().st_size / 1e9
    print(f"  ✓ Decompressed: {compressed_size:.2f} GB → {uncompressed_size:.2f} GB")
    print(f"  ✓ Saved to: {uncompressed_path.name}")

    return str(uncompressed_path)


class SoilMoistureSequenceDataset(_BaseDataset):
    """
    PyTorch Dataset for soil moisture prediction with temporal sequences
    Suitable for transformer models

    Note: Requires PyTorch to be installed. If PyTorch is not available,
    this class can still be instantiated but PyTorch-specific functionality
    (tensors, DataLoader) will not work.

    For optimal performance, use precompute_and_save() to precompute all sequences
    and save to disk, then load with precomputed_path parameter.
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
            precomputed_path: Path to precomputed .npz file (for fast loading)
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
            self.dense_date_to_idx = {date: idx for idx, date in enumerate(self.dense_arrays['dates'])}
            print(f"  Loaded dense arrays: {self.dense_arrays['features'].shape}")
            print(f"  Memory: ~{self.dense_arrays['features'].nbytes / 1e6:.1f} MB")

            # Still build timeseries index as fallback for edge cases
            print("  Building fallback index for edge cases...")
            self.timeseries_index = {
                (int(row.station_id), row.date, row.parameter_code): float(row.value)
                for row in self.timeseries_df.itertuples(index=False)
            }
        else:
            # Use dict lookup index
            print("Creating fast lookup index for timeseries...")
            # Use itertuples() - 10x faster than iterrows()
            self.timeseries_index = {
                (int(row.station_id), row.date, row.parameter_code): float(row.value)
                for row in self.timeseries_df.itertuples(index=False)
            }
            print(f"  Indexed {len(self.timeseries_index):,} data points")

        # Determine target stations
        if target_stations is None:
            self.target_stations = self.stations_df[
                self.stations_df['has_soil_moisture']
            ]['station_id'].tolist()
        else:
            self.target_stations = target_stations

        # Determine feature parameters
        if feature_params is None:
            # Use all parameters except soil moisture
            all_params = self.timeseries_df['parameter_code'].unique()
            self.feature_params = [p for p in all_params if p != soil_moisture_param]
        else:
            self.feature_params = feature_params

        # Track if data is already normalized in precomputed file
        self.is_prenormalized = False

        # Check if soil moisture is in feature_params (data leakage!)
        self.soil_in_features = self.soil_moisture_param in self.feature_params if feature_params else False
        if self.soil_in_features:
            print(f"⚠ WARNING: Soil moisture ({self.soil_moisture_param}) found in feature_params!")
            print(f"  This will be filtered out from target station features to prevent data leakage.")
            print(f"  Nearby stations will still have soil moisture as context.")
            self.soil_feature_idx = self.feature_params.index(self.soil_moisture_param)
        else:
            self.soil_feature_idx = None

        # Load precomputed data if available
        if precomputed_path and os.path.exists(precomputed_path):
            print(f"Loading precomputed sequences from {precomputed_path}...")
            # Ensure we use uncompressed NPZ to avoid zlib errors with multiple workers
            uncompressed_path = _get_uncompressed_npz_path(precomputed_path)
            # Use memory-mapped mode to avoid loading entire dataset into RAM
            # Only accessed samples will be loaded, allowing training on large datasets
            self.precomputed_data = np.load(uncompressed_path, mmap_mode='r')
            print(f"  Using memory-mapped arrays (dataset will not be loaded into RAM)")

            # Check if data is already normalized
            if 'is_normalized' in self.precomputed_data:
                self.is_prenormalized = bool(self.precomputed_data['is_normalized'][0])

            # Keep metadata as memory-mapped arrays - don't build sample_index
            # This avoids creating thousands/millions of Python objects
            self.sample_index = None  # Will use array-based indexing
            self.n_samples = len(self.precomputed_data['target_stations'])
            self.indices = None  # No index mapping (use all data)

            print(f"  Loaded {self.n_samples} precomputed samples")
            if self.is_prenormalized:
                print(f"  Data is pre-normalized (fast path enabled!)")
        else:
            # Build index of valid samples
            self._build_sample_index()
            self.n_samples = len(self.sample_index)
            self.indices = None  # No index mapping for non-precomputed

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
        print(f"  Valid samples: {self.n_samples}")
        print(f"  Using precomputed: {precomputed_path is not None and os.path.exists(precomputed_path)}")
        print(f"  Pre-normalized: {self.is_prenormalized}")
        print(f"  Runtime normalization: {self.normalize}")

    def _build_sample_index(self):
        """Build index of valid samples (target_station, end_date) pairs"""
        self.sample_index = []

        print("Building sample index...")

        for target_id in self.target_stations:
            # Get all dates for this target station with soil moisture data
            target_soil_data = self.timeseries_df[
                (self.timeseries_df['station_id'] == target_id) &
                (self.timeseries_df['parameter_code'] == self.soil_moisture_param)
                ]['date'].unique()

            # Sort dates
            target_dates = sorted(target_soil_data)

            # For each date, check if we have enough history
            for date in target_dates:
                # Need seq_length days including this date
                start_date = date - pd.Timedelta(days=self.seq_length - 1)

                # Check if we have data for the full sequence
                date_range = pd.date_range(start=start_date, end=date, freq='D')

                # Check if target has at least one data point in the sequence
                target_data_in_range = self.timeseries_df[
                    (self.timeseries_df['station_id'] == target_id) &
                    (self.timeseries_df['date'].isin(date_range))
                    ]

                if not target_data_in_range.empty:
                    self.sample_index.append({
                        'target_station': target_id,
                        'end_date': date,
                        'start_date': start_date
                    })

    def _get_nearest_stations(self, target_station_id: int) -> List[Dict]:
        """Get n nearest stations with soil moisture for a target station"""
        nearest_info = self.nearest_df[
            self.nearest_df['station_id'] == target_station_id
            ].iloc[0]

        nearby_with_soil = []
        for i in range(1, len(self.nearest_df.columns) // 3 + 1):
            if f'nearest_{i}_id' not in nearest_info:
                break
            if nearest_info[f'nearest_{i}_has_soil_moisture']:
                nearby_with_soil.append({
                    'station_id': int(nearest_info[f'nearest_{i}_id']),
                    'distance': nearest_info[f'nearest_{i}_distance']
                })
                if len(nearby_with_soil) == self.n_nearest:
                    break

        return nearby_with_soil

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
        mask = np.zeros((self.seq_length, total_features), dtype=np.float32)

        # Fill target station features using FAST DICT LOOKUP
        for t, date in enumerate(date_range):
            # Target station features
            for f_idx, param in enumerate(self.feature_params):
                key = (target_station_id, date, param)
                if key in self.timeseries_index:
                    features[t, f_idx] = self.timeseries_index[key]
                    mask[t, f_idx] = 1.0

            # Fill nearby stations features using FAST DICT LOOKUP
            for n_idx, nearby in enumerate(nearby_stations):
                nearby_station_id = nearby['station_id']
                nearby_offset = target_features_per_timestep + (n_idx * nearby_features_per_timestep)

                # Distance (constant across time)
                features[t, nearby_offset] = nearby['distance']
                mask[t, nearby_offset] = 1.0

                # Features
                for f_idx, param in enumerate(self.feature_params):
                    key = (nearby_station_id, date, param)
                    feat_idx = nearby_offset + 1 + f_idx
                    if key in self.timeseries_index:
                        features[t, feat_idx] = self.timeseries_index[key]
                        mask[t, feat_idx] = 1.0

                # Soil moisture for nearby station
                key = (nearby_station_id, date, self.soil_moisture_param)
                soil_idx = nearby_offset + 1 + len(self.feature_params)
                if key in self.timeseries_index:
                    features[t, soil_idx] = self.timeseries_index[key]
                    mask[t, soil_idx] = 1.0

        # Get target (soil moisture at end_date for target station)
        target_key = (target_station_id, end_date, self.soil_moisture_param)
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

        # Get date range indices
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        date_indices = [self.dense_date_to_idx.get(date) for date in date_range]

        # Check if all dates are in our dense array
        if None in date_indices:
            # Fall back to dict method if dates not available
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
        mask = np.zeros((self.seq_length, total_features), dtype=np.float32)

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
            mask[:, nearby_offset] = 1.0

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
        else:
            # Fall back to dict lookup for soil moisture if not in dense array
            target_key = (target_station_id, end_date, self.soil_moisture_param)
            target = self.timeseries_index.get(target_key, self.missing_value) if self.timeseries_index else self.missing_value

        return (
            torch.from_numpy(features),
            torch.tensor(target, dtype=torch.float32).unsqueeze(0),
            torch.from_numpy(mask)
        )

    def _compute_norm_stats_from_precomputed(self):
        """Compute normalization statistics from precomputed data"""
        print("Computing min/max for each feature (excluding invalid values)...")

        n_features = self.precomputed_data['features'].shape[2]
        feature_mins = np.full(n_features, np.inf, dtype=np.float32)
        feature_maxs = np.full(n_features, -np.inf, dtype=np.float32)

        # Invalid markers to exclude
        invalid_markers = [-9999.0, self.missing_value]

        # Process in batches to save memory
        batch_size = 1000
        for i in range(0, len(self.precomputed_data['features']), batch_size):
            end_i = min(i + batch_size, len(self.precomputed_data['features']))
            features_batch = self.precomputed_data['features'][i:end_i]
            masks_batch = self.precomputed_data['masks'][i:end_i]

            for feat_idx in range(n_features):
                feat_data = features_batch[:, :, feat_idx]
                feat_mask = masks_batch[:, :, feat_idx]

                # Get valid data (masked and not invalid marker)
                valid_mask = feat_mask > 0
                for marker in invalid_markers:
                    valid_mask &= (feat_data != marker)

                valid_data = feat_data[valid_mask]

                if len(valid_data) > 0:
                    feature_mins[feat_idx] = min(feature_mins[feat_idx], valid_data.min())
                    feature_maxs[feat_idx] = max(feature_maxs[feat_idx], valid_data.max())

        # Compute for target as well
        targets = self.precomputed_data['targets']
        valid_targets = targets.copy()
        for marker in invalid_markers:
            valid_targets = valid_targets[valid_targets != marker]

        target_min = valid_targets.min() if len(valid_targets) > 0 else 0.0
        target_max = valid_targets.max() if len(valid_targets) > 0 else 1.0

        self.norm_stats = {
            'feature_mins': feature_mins,
            'feature_maxs': feature_maxs,
            'target_min': target_min,
            'target_max': target_max
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
        all_masks = np.zeros((len(self.sample_index), seq_length, n_features), dtype=np.float32)

        # Store first sample
        all_features[0] = features0.numpy()
        all_targets[0] = target0.numpy()
        all_masks[0] = mask0.numpy()

        # Precompute all samples
        total = len(self.sample_index)
        for idx in range(1, total):
            if idx % 1000 == 0 or idx == total - 1:
                print(f"  Progress: {idx}/{total} ({100*idx/total:.1f}%)")

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
            for idx in range(len(self.sample_index)):
                if idx % 1000 == 0 or idx == total - 1:
                    print(f"  Normalizing: {idx}/{total} ({100*idx/total:.1f}%)")

                all_features[idx], all_targets[idx] = self._apply_normalization(
                    all_features[idx],
                    all_targets[idx],
                    all_masks[idx]
                )

            is_normalized = True

            # Save normalization statistics
            if norm_stats_path:
                print(f"Saving normalization stats to {norm_stats_path}...")
                np.savez(
                    norm_stats_path,
                    feature_mins=self.norm_stats['feature_mins'],
                    feature_maxs=self.norm_stats['feature_maxs'],
                    target_min=self.norm_stats['target_min'],
                    target_max=self.norm_stats['target_max']
                )

        # Save to disk
        print(f"Saving to {output_path}...")

        # Extract sample_index components to avoid pickle requirement
        target_stations = np.array([s['target_station'] for s in self.sample_index], dtype=np.int32)
        end_dates = np.array([s['end_date'].timestamp() for s in self.sample_index], dtype=np.float64)
        start_dates = np.array([s['start_date'].timestamp() for s in self.sample_index], dtype=np.float64)

        np.savez_compressed(
            output_path,
            features=all_features,
            targets=all_targets,
            masks=all_masks,
            target_stations=target_stations,
            end_dates=end_dates,
            start_dates=start_dates,
            is_normalized=np.array([is_normalized], dtype=bool)
        )

        print(f"  Saved {len(self.sample_index)} sequences")
        print(f"  Shape: features={all_features.shape}, targets={all_targets.shape}")
        print(f"  Data is {'normalized' if is_normalized else 'not normalized'}")

        print("Done!")

    def __len__(self) -> int:
        if self.sample_index is None:
            # Precomputed data - use n_samples
            return self.n_samples
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
            actual_idx = self.indices[idx] if self.indices is not None else idx

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
            # Build on-the-fly (slow fallback)
            features_tensor, target_tensor, mask_tensor = self._build_sequence_tensor(
                self.sample_index[idx]['target_station'],
                self.sample_index[idx]['start_date'],
                self.sample_index[idx]['end_date']
            )

        # Get metadata
        if self.sample_index is None:
            # Precomputed data - get from arrays directly
            # Map index if this is a split dataset
            actual_idx = self.indices[idx] if self.indices is not None else idx
            target_station_id = int(self.precomputed_data['target_stations'][actual_idx])
            end_date_unix = float(self.precomputed_data['end_dates'][actual_idx])
        else:
            # Non-precomputed - get from sample_index
            sample = self.sample_index[idx]
            target_station_id = sample['target_station']
            end_date_unix = sample['end_date'].timestamp() if hasattr(sample['end_date'], 'timestamp') else float(sample['end_date'])

        return {
            'features': features_tensor,
            'target': target_tensor,
            'mask': mask_tensor,
            'target_station_id': target_station_id,
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
        if self.sample_index is None:
            # Precomputed data - use arrays directly
            all_stations = self.precomputed_data['target_stations']
            train_indices = [i for i in range(len(all_stations)) if int(all_stations[i]) in train_stations]
            val_indices = [i for i in range(len(all_stations)) if int(all_stations[i]) in val_stations]
            test_indices = [i for i in range(len(all_stations)) if int(all_stations[i]) in test_stations]
        else:
            # Non-precomputed - use sample_index
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
        # This avoids loading hundreds of GB into RAM
        split_dataset.precomputed_data = self.precomputed_data

        # Store index mapping for this split
        split_dataset.indices = indices
        split_dataset.n_samples = len(indices)

        # Don't build sample_index (use lazy access instead)
        split_dataset.sample_index = None

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

def buildDataset():
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

    # Collect 2 years of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)

    timeseries_df = collector.build_historical_dataset(
        station_ids=all_station_ids,
        parameter_ids=parameters,
        start_date=start_date,
        end_date=end_date,
        chunk_days=30
    )

    # Step 4: Create ML-ready dataset
    print("\n" + "=" * 60)
    print("STEP 4: Creating ML-ready dataset")
    print("=" * 60)

    ml_df = collector.create_ml_ready_dataset(n_nearest=4)

    # Step 5: Example of getting live data
    print("\n" + "=" * 60)
    print("STEP 5: Example - Getting live data for prediction")
    print("=" * 60)

    # Get a station with soil moisture for demo
    soil_moisture_stations = stations_df[stations_df['has_soil_moisture']]['station_id'].tolist()
    if soil_moisture_stations:
        demo_station = soil_moisture_stations[0]

        # Get live prediction data
        live_data = collector.get_live_prediction_data(
            target_station_id=demo_station,
            n_nearest=4
        )

        print(f"\nTarget station: {live_data['target_station_id']}")
        print(f"Target coordinates: {live_data['target_coordinates']}")
        print(f"Nearby stations: {len(live_data['nearby_stations'])}")
        print(f"\nTarget station data:")
        print(live_data['target_data'].head() if not live_data['target_data'].empty else "No data")
        print(f"\nNearby stations data:")
        print(live_data['nearby_data'].head() if not live_data['nearby_data'].empty else "No data")

    # Step 6: Create PyTorch Dataset
    print("\n" + "=" * 60)
    print("STEP 6: Creating PyTorch Dataset")
    print("=" * 60)

    # Get filtered parameters
    _, filtered_params = collector.analyze_parameter_coverage(coverage_threshold=0.25)

    if not filtered_params:
        print("\n✗ No parameters passed the threshold!")
        return

    print(f"\nUsing {len(filtered_params)} filtered parameters...")

    dataset = SoilMoistureSequenceDataset(
        timeseries=str(collector.timeseries_file),
        stations=str(collector.stations_file),
        nearest=str(collector.nearest_file),
        seq_length=64,
        n_nearest=4,
        feature_params=filtered_params
    )

    print(f"\nFeature names: {dataset.get_feature_names()}")

    # Example sample
    sample = dataset[0]
    print(f"\nExample sample:")
    print(f"  Features shape: {sample['features'].shape}")
    print(f"  Target: {sample['target']}")
    print(f"  Mask shape: {sample['mask'].shape}")
    print(f"  Station ID: {sample['target_station_id']}")

    # Train/val/test split
    print("\n" + "=" * 60)
    print("STEP 7: Creating train/val/test splits")
    print("=" * 60)

    train_ds, val_ds, test_ds = SoilMoistureSequenceDataset.train_val_test_split(
        dataset,
        val_stations_ratio=0.15,
        test_stations_ratio=0.0
    )
    return train_ds,val_ds,test_ds

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
    precomputed_path = collector.data_dir / "precomputed_sequences.npz"
    norm_stats_path = collector.data_dir / "normalization_stats.npz"

    dataset.precompute_and_save(
        output_path=str(precomputed_path),
        norm_stats_path=str(norm_stats_path)
    )

    print(f"\n✓ Precomputed sequences saved to: {precomputed_path}")
    print(f"✓ Normalization stats saved to: {norm_stats_path}")
    print(f"\nYou can now use loadDataset() for fast loading!")

def loadDataset(use_precomputed=True, normalize=True):
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
    precomputed_path = collector.data_dir / "precomputed_sequences.npz"
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

# Example usage
if __name__ == "__main__":
    # Check if precomputed data exists
    collector = MeteoGaliciaCollector()
    precomputed_path = collector.data_dir / "precomputed_sequences.npz"

    if not precomputed_path.exists():
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

    train_ds, val_ds, _ = loadDataset(use_precomputed=True, normalize=True)

    # After loading dataset, check one sample:
    sample = train_ds[0]
    features = sample['features']  # [64, total_features]
    target = sample['target']  # [1]

    print(f"\n=== Data Leakage Check ===")
    print(f"Feature shape: {features.shape}")
    print(f"Target value: {target.item():.4f}")

    # Check if target value appears anywhere in features
    # (it shouldn't if there's no leakage!)
    features_np = features.numpy()
    matches = (np.abs(features_np - target.item()) < 0.001).sum()
    print(f"Features matching target value: {matches}")

    if matches > 0:
        print("⚠️  LEAKAGE DETECTED: Target value found in features!")
    else:
        print("✓ No obvious leakage detected")

    # Check last timestep specifically (most likely leak point)
    last_timestep = features_np[-1, :]
    last_matches = (np.abs(last_timestep - target.item()) < 0.001).sum()
    print(f"Last timestep matches: {last_matches}")

    #torch._dynamo.config.disable = True
    from TROLOLO.TROLOLO_pyramid import TROLOLO
    quantize = False
    trololo = TROLOLO(seq_length=64,
                      num_layers=6,
                      num_heads=48,
                      embed_dim=192,
                      mlp_dim=192,
                      n_class_tokens=2,
                      num_classes=1,
                      mlp_rank=0.05,
                      qkv_rank=0.05,
                      attnproj_rank=0.05,
                      sequence_pyramid=[],
                      attn_rank_pyramid=[(0, 32), (1, 32)],
                      rank_pyramid_begin=2,
                      rank_pyramid_factor=1.0,
                      head_constriction="ONE_CLASS_TOKEN",
                      dropout=0.05,
                      attention_dropout=0.01,
                      quantize_bits= None if not quantize else 8
                      )
    trololo.training_loop(train_data=train_ds,val_data=val_ds,lr=4.1e-4,lr_mid=4.0e-4,lr_min=3e-5,n_epochs=10000,batch_size=512,transfer=0)