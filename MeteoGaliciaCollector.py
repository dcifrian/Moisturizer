import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
import time
from pathlib import Path

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

        # Cached DataFrames (lazy loaded)
        self._stations_df_cache = None
        self._nearest_df_cache = None
        self._timeseries_df_cache = None

    def get_stations_df(self, force_reload: bool = False) -> pd.DataFrame:
        """Get stations DataFrame with caching"""
        if self._stations_df_cache is None or force_reload:
            if self.stations_file.exists():
                self._stations_df_cache = pd.read_csv(self.stations_file)
            else:
                return None
        return self._stations_df_cache

    def get_nearest_df(self, force_reload: bool = False) -> pd.DataFrame:
        """Get nearest stations DataFrame with caching"""
        if self._nearest_df_cache is None or force_reload:
            if self.nearest_file.exists():
                self._nearest_df_cache = pd.read_csv(self.nearest_file)
            else:
                return None
        return self._nearest_df_cache

    def get_timeseries_df(self, force_reload: bool = False) -> pd.DataFrame:
        """Get timeseries DataFrame with caching"""
        if self._timeseries_df_cache is None or force_reload:
            if self.timeseries_file.exists():
                self._timeseries_df_cache = pd.read_csv(self.timeseries_file)
            else:
                return None
        return self._timeseries_df_cache

    def clear_cache(self):
        """Clear all cached DataFrames"""
        self._stations_df_cache = None
        self._nearest_df_cache = None
        self._timeseries_df_cache = None

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
        except (requests.RequestException, KeyError, ValueError) as e:
            print(f"Warning: soil moisture check failed for station: {e}")
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
            return self.get_stations_df()

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
            n_nearest: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate n nearest stations WITH SOIL MOISTURE for each station using Euclidean distance on UTM coords

        This finds the nearest stations that have soil moisture data, which is essential for
        predicting/imputing soil moisture at stations that don't have it.

        Args:
            stations_df: DataFrame with station metadata including utmx, utmy, has_soil_moisture
            n_nearest: Number of nearest stations WITH SOIL MOISTURE to find.
                      If None, uses all available (n_soil_moisture_stations - 1)

        Returns:
            DataFrame with columns: station_id, nearest_1_id, nearest_1_distance,
                                   nearest_2_id, nearest_2_distance, ...
        """
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

        # Calculate max possible neighbors (all soil moisture stations except self)
        max_neighbors = len(soil_moisture_stations) - 1

        if n_nearest is None:
            n_nearest = max_neighbors
            print(f"\nCalculating ALL {n_nearest} nearest stations WITH SOIL MOISTURE for each station...")
        else:
            if n_nearest > max_neighbors:
                raise ValueError(
                    f"Requested n_nearest={n_nearest} but only {max_neighbors} neighbors available "
                    f"({len(soil_moisture_stations)} soil moisture stations - 1 for self)"
                )
            print(f"\nCalculating {n_nearest} nearest stations WITH SOIL MOISTURE for each station...")

        print(f"  Total stations with valid coordinates: {len(stations_with_coords)}")
        print(f"  Stations with soil moisture data: {len(soil_moisture_stations)}")
        print(f"  Max possible neighbors: {max_neighbors}")

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
            print(f"Loading cached timeseries from {self.timeseries_file}")
            return self.get_timeseries_df()
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
                # Use .item() to properly extract scalar from 0-d or 1-d array
                cached_threshold = float(np.asarray(cache['coverage_threshold']).item())
                cached_n_days = int(np.asarray(cache['n_days']).item())
                cached_add_coords = bool(np.asarray(cache['add_coordinate_features']).item())

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

        # Load stations metadata and nearest stations (using cache)
        stations_df = self.get_stations_df()
        nearest_df = self.get_nearest_df()

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

