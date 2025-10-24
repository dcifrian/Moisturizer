"""
MeteoGalicia Weather Station Data Collector
Collects historical and live data from MeteoGalicia API for ML model training
with focus on soil moisture prediction from nearby stations
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple, Set
import json
import time
from pathlib import Path

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

    def __init__(self, cache_dir: str = "./meteogalicia_data"):
        self.session = requests.Session()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.stations_file = self.cache_dir / "stations_metadata.csv"
        self.nearest_file = self.cache_dir / "nearest_stations.csv"
        self.timeseries_file = self.cache_dir / "raw_timeseries.csv"

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
            chunk_days: int = 30
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
                output_file = self.cache_dir / "ml_ready_dataset.csv"

            ml_df.to_csv(output_file, index=False)
            print(f"✓ ML-ready dataset saved to {output_file}")
            print(f"  Total rows: {len(ml_df)}")
            print(f"  Columns: {len(ml_df.columns)}")

            return ml_df
        else:
            print("✗ No data to create ML dataset")
            return pd.DataFrame()

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


class SoilMoistureSequenceDataset(_BaseDataset):
    """
    PyTorch Dataset for soil moisture prediction with temporal sequences
    Suitable for transformer models

    Note: Requires PyTorch to be installed. If PyTorch is not available,
    this class can still be instantiated but PyTorch-specific functionality
    (tensors, DataLoader) will not work.
    """

    def __init__(
            self,
            timeseries_file: str,
            stations_file: str,
            nearest_file: str,
            seq_length: int,
            n_nearest: int = 4,
            target_stations: Optional[List[int]] = None,
            feature_params: Optional[List[str]] = None,
            soil_moisture_param: str = "HS_CV_AVG_-0.2m",
            missing_value: float = -1000.0
    ):
        """
        Initialize dataset

        Args:
            timeseries_file: Path to raw_timeseries.csv
            stations_file: Path to stations_metadata.csv
            nearest_file: Path to nearest_stations.csv
            seq_length: Number of days in each sequence
            n_nearest: Number of nearest stations to include
            target_stations: List of station IDs to use (if None, use all with soil moisture)
            feature_params: List of parameter codes to include as features
                           (if None, uses all except soil moisture for target station)
            soil_moisture_param: Parameter code for soil moisture (target variable)
            missing_value: Value to use for missing data
        """
        self.seq_length = seq_length
        self.n_nearest = n_nearest
        self.soil_moisture_param = soil_moisture_param
        self.missing_value = missing_value

        # Load data
        print("Loading data files...")
        self.timeseries_df = pd.read_csv(timeseries_file)
        self.timeseries_df['date'] = pd.to_datetime(self.timeseries_df['date'])
        self.stations_df = pd.read_csv(stations_file)
        self.nearest_df = pd.read_csv(nearest_file)

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

        # Build index of valid samples
        self._build_sample_index()

        print(f"Dataset initialized:")
        print(f"  Sequence length: {seq_length}")
        print(f"  Target stations: {len(self.target_stations)}")
        print(f"  Feature parameters: {len(self.feature_params)}")
        print(f"  Valid samples: {len(self.sample_index)}")

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
        Build sequence tensor for a sample

        Returns:
            features: [seq_length, total_features] tensor
            target: scalar tensor (soil moisture at end_date)
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

        # Fill target station features
        for t, date in enumerate(date_range):
            target_data = self.timeseries_df[
                (self.timeseries_df['station_id'] == target_station_id) &
                (self.timeseries_df['date'] == date)
                ]

            for f_idx, param in enumerate(self.feature_params):
                param_data = target_data[target_data['parameter_code'] == param]
                if not param_data.empty:
                    features[t, f_idx] = param_data.iloc[0]['value']
                    mask[t, f_idx] = 1.0

            # Fill nearby stations features
            for n_idx, nearby in enumerate(nearby_stations):
                nearby_data = self.timeseries_df[
                    (self.timeseries_df['station_id'] == nearby['station_id']) &
                    (self.timeseries_df['date'] == date)
                    ]

                # Offset for this nearby station's features
                nearby_offset = target_features_per_timestep + (n_idx * nearby_features_per_timestep)

                # Distance (constant across time)
                features[t, nearby_offset] = nearby['distance']
                mask[t, nearby_offset] = 1.0

                # Features
                for f_idx, param in enumerate(self.feature_params):
                    param_data = nearby_data[nearby_data['parameter_code'] == param]
                    feat_idx = nearby_offset + 1 + f_idx
                    if not param_data.empty:
                        features[t, feat_idx] = param_data.iloc[0]['value']
                        mask[t, feat_idx] = 1.0

                # Soil moisture for nearby station
                soil_data = nearby_data[nearby_data['parameter_code'] == self.soil_moisture_param]
                soil_idx = nearby_offset + 1 + len(self.feature_params)
                if not soil_data.empty:
                    features[t, soil_idx] = soil_data.iloc[0]['value']
                    mask[t, soil_idx] = 1.0

        # Get target (soil moisture at end_date for target station)
        target_data = self.timeseries_df[
            (self.timeseries_df['station_id'] == target_station_id) &
            (self.timeseries_df['date'] == end_date) &
            (self.timeseries_df['parameter_code'] == self.soil_moisture_param)
            ]

        target = target_data.iloc[0]['value'] if not target_data.empty else self.missing_value

        return (
            torch.from_numpy(features),
            torch.tensor(target, dtype=torch.float32),
            torch.from_numpy(mask)
        )

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int) -> 'Dict[str, torch.Tensor]':
        """
        Get a sample

        Returns:
            Dictionary with:
                - features: [seq_length, total_features]
                - target: scalar
                - mask: [seq_length, total_features]
                - target_station_id: int
                - end_date: timestamp
        """
        sample = self.sample_index[idx]

        features, target, mask = self._build_sequence_tensor(
            sample['target_station'],
            sample['start_date'],
            sample['end_date']
        )

        return {
            'features': features,
            'target': target,
            'mask': mask,
            'target_station_id': sample['target_station'],
            'end_date': sample['end_date']
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

    @staticmethod
    def train_val_test_split(
            dataset: 'SoilMoistureSequenceDataset',
            val_stations_ratio: float = 0.15,
            test_stations_ratio: float = 0.15,
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

        # Create new datasets with filtered stations
        train_dataset = SoilMoistureSequenceDataset(
            timeseries_file=dataset.timeseries_df,  # Pass dataframe directly
            stations_file=dataset.stations_df,
            nearest_file=dataset.nearest_df,
            seq_length=dataset.seq_length,
            n_nearest=dataset.n_nearest,
            target_stations=train_stations,
            feature_params=dataset.feature_params,
            soil_moisture_param=dataset.soil_moisture_param,
            missing_value=dataset.missing_value
        )

        val_dataset = SoilMoistureSequenceDataset(
            timeseries_file=dataset.timeseries_df,
            stations_file=dataset.stations_df,
            nearest_file=dataset.nearest_df,
            seq_length=dataset.seq_length,
            n_nearest=dataset.n_nearest,
            target_stations=val_stations,
            feature_params=dataset.feature_params,
            soil_moisture_param=dataset.soil_moisture_param,
            missing_value=dataset.missing_value
        )

        test_dataset = SoilMoistureSequenceDataset(
            timeseries_file=dataset.timeseries_df,
            stations_file=dataset.stations_df,
            nearest_file=dataset.nearest_df,
            seq_length=dataset.seq_length,
            n_nearest=dataset.n_nearest,
            target_stations=test_stations,
            feature_params=dataset.feature_params,
            soil_moisture_param=dataset.soil_moisture_param,
            missing_value=dataset.missing_value
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


# Example usage
if __name__ == "__main__":
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

    dataset = SoilMoistureSequenceDataset(
        timeseries_file=str(collector.timeseries_file),
        stations_file=str(collector.stations_file),
        nearest_file=str(collector.nearest_file),
        seq_length=7,
        n_nearest=4
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
        test_stations_ratio=0.15
    )

    # Example DataLoader usage
    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    print(f"\nDataLoader created with batch_size=32")
    print(f"  Batches per epoch: {len(train_loader)}")