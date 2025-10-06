import warnings
import requests
import numpy as np
import pandas as pd
import xarray as xr
from scipy import signal, stats
from datetime import datetime
from pathlib import Path
from io import StringIO
from typing import List, Dict, Optional, Tuple

# Suppress common warnings for a cleaner output
warnings.filterwarnings('ignore')


class MultiVariableSharkAnalyzer:
    """
    Analyzes shark tracking data against various environmental variables.

    This class provides a comprehensive toolkit to download, align, and analyze
    shark movement data from Movebank with oceanographic data from sources like
    NOAA. It supports both time-domain (correlation) and frequency-domain
    (coherence) analyses to identify potential environmental drivers of shark
    behavior.

    Attributes:
        temporal_resolution (str): The time bin size for gridding data (e.g., '1D', '3D').
        spatial_resolution (float): The spatial grid cell size in degrees.
        cache_dir (Path): The directory to store downloaded data files.
        shark_data (Optional[pd.DataFrame]): DataFrame holding the raw shark tracking data.
        environmental_data (Dict[str, xr.Dataset]): A dictionary of environmental datasets.
        aligned_data (Optional[pd.DataFrame]): Gridded data aligning shark counts and env. variables.
    """

    def __init__(self, temporal_resolution: str = '1D', spatial_resolution: float = 0.25, cache_dir: str = './data'):
        """
        Initializes the MultiVariableSharkAnalyzer.

        Args:
            temporal_resolution (str): The temporal binning resolution (pandas frequency string).
            spatial_resolution (float): The spatial grid resolution in decimal degrees.
            cache_dir (str): Path to the directory for caching downloaded data.
        """
        self.temporal_resolution = temporal_resolution
        self.spatial_resolution = spatial_resolution
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.shark_data: Optional[pd.DataFrame] = None
        self.environmental_data: Dict[str, xr.Dataset] = {}
        self.aligned_data: Optional[pd.DataFrame] = None

        self.available_variables = {
            'sst': 'Sea Surface Temperature (NOAA OISST)',
            'chlorophyll': 'Chlorophyll-a concentration (NASA MODIS)',
            'salinity': 'Sea Surface Salinity (HYCOM)',
            'bathymetry': 'Ocean depth (GEBCO/ETOPO)',
            'currents_u': 'Ocean current U component (HYCOM)',
            'currents_v': 'Ocean current V component (HYCOM)',
            'moon_phase': 'Moon phase (calculated)',
            'distance_to_shore': 'Distance to coastline (calculated)'
        }

    def run_full_analysis(
        self,
        study_id: int = None,
        shark_file: str = None,
        start_date: str = None,
        end_date: str = None,
        freq_min_observations: int = 20
    ) -> Dict:
        """
        Executes the complete analysis pipeline from data download to result generation.

        Args:
            study_id (int, optional): The Movebank study ID to download.
            shark_file (str, optional): Path to a local CSV file with shark data.
            start_date (str, optional): The start date for the analysis (YYYY-MM-DD).
            end_date (str, optional): The end date for the analysis (YYYY-MM-DD).
            freq_min_observations (int): Minimum time series length for coherence analysis.

        Returns:
            Dict: A dictionary containing correlation dataframes, predictor rankings,
                  and matplotlib figures for time and frequency domain analyses.
        """
        # 1. Load Shark Data
        if study_id is not None:
            self.download_movebank_data(study_id)
        elif shark_file is not None:
            df = pd.read_csv(shark_file)
            self.load_shark_data_from_dataframe(df)
        else:
            raise ValueError("Must provide either a 'study_id' or a 'shark_file'.")

        if start_date is None:
            start_date = self.shark_data['timestamp'].min()
        if end_date is None:
            end_date = self.shark_data['timestamp'].max()

        # 2. Load Environmental Data
        print("\n" + "="*70)
        print("DOWNLOADING AND LOADING ENVIRONMENTAL DATA")
        print("="*70)

        print("\n1. Sea Surface Temperature (SST)")
        try:
            self.download_noaa_sst_data(start_date, end_date)
        except Exception as e:
            print(f"  SST download failed: {e}")

        print("\n2. Checking for manually downloaded data...")
        self._load_manual_files()

        # 3. Align Data
        self.extract_environmental_variables()
        self.create_spatial_temporal_grid()

        # 4. Time Domain Analysis
        correlations = self.compute_multi_variable_correlations()
        predictor_quality, ranked_predictors = self.identify_best_predictor(correlations)
        fig_time = self.visualize_multi_variable_results(
            correlations, top_n=min(3, len(ranked_predictors)))

        # 5. Frequency Domain Analysis
        freq_df = self.frequency_domain_coherence_analysis(
            min_observations=freq_min_observations, plot=False)

        freq_ranking = []
        fig_freq = None
        if freq_df is not None and not freq_df.empty:
            fig_freq = self.visualize_multi_variable_results_frequency(
                freq_df, top_n=min(3, len(freq_df['variable'].unique())))

        return {
            'correlations': correlations,
            'predictor_quality': predictor_quality,
            'ranked_predictors': ranked_predictors,
            'figure_time_domain': fig_time,
            'coherence_results': freq_df,
            'figure_frequency_domain': fig_freq
        }

    def download_movebank_data(self, study_id: int, username: str = None, password: str = None, use_cache: bool = True) -> pd.DataFrame:
        """
        Downloads and loads animal tracking data from Movebank.org.

        Args:
            study_id (int): The numeric ID of the study on Movebank.
            username (str, optional): Your Movebank username for private studies.
            password (str, optional): Your Movebank password for private studies.
            use_cache (bool): If True, loads data from a local cache file if available.

        Returns:
            pd.DataFrame: A DataFrame containing the cleaned shark tracking data.
        """
        print(f"Downloading Movebank data for study {study_id}...")
        cache_file = self.cache_dir / f"movebank_study_{study_id}.csv"

        if use_cache and cache_file.exists():
            print(f"Loading cached data from {cache_file}")
            df = pd.read_csv(cache_file)
            return self.load_shark_data_from_dataframe(df)

        base_url = "https://www.movebank.org/movebank/service/direct-read"
        params = {'entity_type': 'event', 'study_id': study_id, 'attributes': 'all'}
        auth = (username, password) if username and password else None

        try:
            response = requests.get(base_url, params=params, auth=auth)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text))
            df.to_csv(cache_file, index=False)
            print(f"Data cached to {cache_file}")
            return self.load_shark_data_from_dataframe(df)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise ValueError("Authentication error. This study may be private or require credentials.")
            else:
                raise

    def load_shark_data_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardizes and cleans a DataFrame of shark tracking data.

        Args:
            df (pd.DataFrame): The raw DataFrame to process.

        Returns:
            pd.DataFrame: The cleaned and standardized DataFrame.
        """
        col_map = {
            'timestamp': ['timestamp', 'study-timestamp'],
            'latitude': ['location-lat', 'lat'],
            'longitude': ['location-long', 'lon', 'long']
        }
        
        def find_col(possible_names):
            for name in possible_names:
                for col in df.columns:
                    if name in col.lower():
                        return col
            return None

        ts_col = find_col(col_map['timestamp'])
        lat_col = find_col(col_map['latitude'])
        lon_col = find_col(col_map['longitude'])

        if not all([ts_col, lat_col, lon_col]):
            raise ValueError(f"Could not find required columns (timestamp, latitude, longitude) in {df.columns.tolist()}")

        df = df.rename(columns={ts_col: 'timestamp', lat_col: 'latitude', lon_col: 'longitude'})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.dropna(subset=['latitude', 'longitude'])
        df = df[(df['latitude'].between(-90, 90)) & (df['longitude'].between(-180, 180))]
        
        self.shark_data = df.sort_values('timestamp').reset_index(drop=True)
        print(f"Loaded {len(self.shark_data)} valid shark observations.")
        print(f"  Time range: {self.shark_data['timestamp'].min()} to {self.shark_data['timestamp'].max()}")
        print(f"  Spatial range: Lat [{self.shark_data['latitude'].min():.2f}, {self.shark_data['latitude'].max():.2f}], "
              f"Lon [{self.shark_data['longitude'].min():.2f}, {self.shark_data['longitude'].max():.2f}]")
        return self.shark_data

    def download_noaa_sst_data(self, start_date: str, end_date: str, use_cache: bool = True) -> xr.Dataset:
        """
        Downloads NOAA OISST v2 High-Resolution Sea Surface Temperature data.

        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
            use_cache (bool): Whether to use cached data if available.

        Returns:
            xr.Dataset: An xarray Dataset containing the SST data.
        """
        print(f"Downloading SST data from {start_date} to {end_date}...")
        start_dt, end_dt = pd.to_datetime(start_date), pd.to_datetime(end_date)
        
        files_to_load = set()
        for year in range(start_dt.year, end_dt.year + 1):
            filename = f"sst.day.mean.{year}.nc"
            cache_file = self.cache_dir / filename
            files_to_load.add(cache_file)
            
            if use_cache and cache_file.exists():
                continue

            url = f"https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/{filename}"
            try:
                print(f"  Downloading {filename}...")
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(cache_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            except requests.exceptions.HTTPError:
                print(f"  Warning: Could not download {filename}. It may not be available.")
                files_to_load.remove(cache_file)

        if not files_to_load:
            raise ValueError("No SST data files could be downloaded for the specified date range.")

        print(f"Loading {len(files_to_load)} SST file(s)...")
        data = xr.open_mfdataset([str(f) for f in files_to_load], combine='by_coords')
        data = data.sel(time=slice(start_dt, end_dt))
        self.environmental_data['sst'] = data
        print(f"SST data loaded successfully.")
        return data
    
    def _load_manual_files(self):
        """Loads environmental data files that were manually downloaded by the user."""
        file_patterns = {
            'chlorophyll': 'chlorophyll*.nc',
            'bathymetry': 'bathymetry*.nc',
            'water_velocity': 'water_velocity*.nc',
            'salinity_temperature': 'salinity_temperature*.nc',
            'sea_level': 'depth*.nc'
        }

        for key, pattern in file_patterns.items():
            files = list(self.cache_dir.glob(pattern))
            for f in files:
                print(f"  Found manual file: {f.name}")
                try:
                    ds = xr.open_dataset(f)
                    # For files with multiple variables, load each one separately
                    if len(ds.data_vars) > 1:
                        for var in ds.data_vars:
                            env_key = f"{key}__{var}"
                            self.environmental_data[env_key] = ds[[var]]
                            print(f"    Loaded variable '{var}' as '{env_key}'")
                    else:
                        self.environmental_data[key] = ds
                        print(f"    Loaded dataset as '{key}'")
                except Exception as e:
                    print(f"    Error loading {f.name}: {e}")

    def extract_environmental_variables(self):
        """
        Extracts values from all loaded environmental datasets at each shark location.
        """
        print("\n" + "="*70)
        print("EXTRACTING ENVIRONMENTAL VARIABLES AT SHARK LOCATIONS")
        print("="*70)

        for key in self.environmental_data:
            var_name = key.split('__')[-1] # e.g., 'water_velocity__vo' -> 'vo'
            self._extract_variable_at_locations(key, var_name)

        # Calculated variables
        self.calculate_moon_phase()
        self.calculate_distance_to_shore()

        available_vars = [col for col in self.shark_data.columns if col not in ['timestamp', 'latitude', 'longitude']]
        print(f"\nAvailable variables for analysis: {available_vars}")
        return self.shark_data

    def _extract_variable_at_locations(self, data_key: str, var_in_dataset: str):
        """
        Generic helper to extract values from a timed/untimed environmental dataset.

        Args:
            data_key (str): The key for the dataset in `self.environmental_data`.
            var_in_dataset (str): The name of the data variable within the xarray Dataset.
        """
        print(f"Extracting {data_key}...")
        dataset = self.environmental_data[data_key]
        
        # Find coordinate names (lat, lon, time) dynamically
        coords = {
            'lat': next((c for c in dataset.coords if 'lat' in c.lower()), None),
            'lon': next((c for c in dataset.coords if 'lon' in c.lower()), None),
            'time': next((c for c in dataset.coords if 'time' in c.lower()), None)
        }
        if not coords['lat'] or not coords['lon']:
            print(f"  Could not identify lat/lon coordinates for '{data_key}'. Skipping.")
            return

        # Handle longitude wrapping (e.g., 0 to 360 vs -180 to 180)
        ds_lon_min = float(dataset[coords['lon']].min())
        shark_lon = self.shark_data['longitude'].copy()
        if ds_lon_min >= 0:
            shark_lon[shark_lon < 0] += 360

        # Create xarray DataArrays for shark coordinates for efficient indexing
        shark_lat_xr = xr.DataArray(self.shark_data['latitude'], dims="event")
        shark_lon_xr = xr.DataArray(shark_lon, dims="event")
        
        selector = {coords['lat']: shark_lat_xr, coords['lon']: shark_lon_xr}
        if coords['time']:
            selector[coords['time']] = xr.DataArray(self.shark_data['timestamp'], dims="event")

        # Use xarray's advanced interpolation/selection
        try:
            extracted_values = dataset[var_in_dataset].sel(**selector, method='nearest').values
            self.shark_data[data_key] = extracted_values
            
            valid_count = np.isfinite(extracted_values).sum()
            print(f"  Extracted {data_key} for {valid_count}/{len(self.shark_data)} observations "
                  f"({100*valid_count/len(self.shark_data):.1f}%)")
        except Exception as e:
            print(f"  Failed to extract {data_key}: {e}")
            self.shark_data[data_key] = np.nan

    def calculate_moon_phase(self) -> pd.Series:
        """
        Calculates the moon phase for each observation.
        Phase is represented as a value from 0 (new moon) to 1.
        """
        print("Calculating moon phase...")
        # A simple approximation for moon phase calculation
        def get_phase(date):
            jd = date.to_julian_date()
            # Reference Julian date for a known new moon
            ref_jd = 2451549.5 
            # Synodic month period (days)
            synodic_month = 29.53058867
            phase = ((jd - ref_jd) / synodic_month) % 1
            return phase

        self.shark_data['moon_phase'] = self.shark_data['timestamp'].apply(get_phase)
        print(f"Moon phase calculated for {len(self.shark_data)} observations.")
        return self.shark_data['moon_phase']

    def calculate_distance_to_shore(self) -> pd.Series:
        """
        Calculates an approximate distance to the nearest coastline.
        Note: This is a highly simplified method. For accuracy, use a GIS library
        with proper coastline shapefiles (e.g., geopandas, shapely).
        """
        print("Calculating distance to shore (using a simplified method)...")
        from math import radians, sin, cos, sqrt, atan2

        # Haversine formula to calculate distance between two lat/lon points
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371.0  # Earth radius in km
            dlat = radians(lat2 - lat1)
            dlon = radians(lon2 - lon1)
            a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            return R * c
        
        # A very coarse representation of the North American coastline
        # For a real analysis, this should be replaced with a shapefile.
        coastline = [
            (25.8, -80.1), (27.7, -82.7), (29.5, -95.3), (29.7, -93.2),
            (28.7, -89.0), (30.2, -88.0), (30.4, -85.7), (32.0, -81.0)
        ]

        def min_dist(lat, lon):
            return min([haversine(lat, lon, clat, clon) for clat, clon in coastline])

        self.shark_data['distance_to_shore'] = self.shark_data.apply(
            lambda row: min_dist(row['latitude'], row['longitude']), axis=1
        )
        print(f"Distance to shore: {self.shark_data['distance_to_shore'].min():.1f} - "
              f"{self.shark_data['distance_to_shore'].max():.1f} km")
        return self.shark_data['distance_to_shore']

    def create_spatial_temporal_grid(self) -> pd.DataFrame:
        """
        Aggregates shark observations and environmental data into a regular grid.
        This process bins the data by time, latitude, and longitude.

        Returns:
            pd.DataFrame: The gridded and aligned data.
        """
        print("\nCreating spatial-temporal grid...")
        env_vars = [v for v in self.available_variables if v in self.shark_data.columns]
        
        if not env_vars:
            raise ValueError("No environmental variables are available for gridding.")
        
        print(f"Using environmental variables: {env_vars}")

        # Binning
        lat_bins = np.arange(self.shark_data['latitude'].min(), self.shark_data['latitude'].max() + self.spatial_resolution, self.spatial_resolution)
        lon_bins = np.arange(self.shark_data['longitude'].min(), self.shark_data['longitude'].max() + self.spatial_resolution, self.spatial_resolution)
        time_bins = pd.date_range(self.shark_data['timestamp'].min(), self.shark_data['timestamp'].max(), freq=self.temporal_resolution)

        binned_data = self.shark_data.copy()
        binned_data['lat_bin'] = pd.cut(binned_data['latitude'], bins=lat_bins, labels=lat_bins[:-1])
        binned_data['lon_bin'] = pd.cut(binned_data['longitude'], bins=lon_bins, labels=lon_bins[:-1])
        binned_data['time_bin'] = pd.cut(binned_data['timestamp'], bins=time_bins, labels=time_bins[:-1])
        
        binned_data = binned_data.dropna(subset=['lat_bin', 'lon_bin', 'time_bin'])

        # Aggregation
        agg_dict = {'timestamp': 'count'}
        for var in env_vars:
            agg_dict[var] = 'mean'

        gridded = binned_data.groupby(['time_bin', 'lat_bin', 'lon_bin']).agg(agg_dict).rename(columns={'timestamp': 'shark_count'})
        
        self.aligned_data = gridded.reset_index()
        self.aligned_data['time_bin'] = pd.to_datetime(self.aligned_data['time_bin'])
        self.aligned_data['lat_bin'] = pd.to_numeric(self.aligned_data['lat_bin'])
        self.aligned_data['lon_bin'] = pd.to_numeric(self.aligned_data['lon_bin'])

        print(f"Created grid with {len(self.aligned_data)} cells containing data.")
        return self.aligned_data

    def compute_multi_variable_correlations(self, min_observations: int = 5) -> pd.DataFrame:
        """
        Computes Pearson and Spearman correlations in each spatial grid cell.

        Args:
            min_observations (int): Minimum number of time points in a cell to compute correlation.

        Returns:
            pd.DataFrame: DataFrame with correlation coefficients for each variable in each cell.
        """
        print(f"\nComputing correlations (min {min_observations} obs per cell)...")
        env_vars = [col for col in self.aligned_data.columns if col not in ['time_bin', 'lat_bin', 'lon_bin', 'shark_count']]
        results = []

        for name, group in self.aligned_data.groupby(['lat_bin', 'lon_bin']):
            if len(group) < min_observations:
                continue
            
            result = {'lat': name[0], 'lon': name[1], 'n_observations': len(group)}
            for var in env_vars:
                valid_data = group[['shark_count', var]].dropna()
                if len(valid_data) >= min_observations and valid_data[var].std() > 0:
                    pearson_r, pearson_p = stats.pearsonr(valid_data['shark_count'], valid_data[var])
                    spearman_r, _ = stats.spearmanr(valid_data['shark_count'], valid_data[var])
                    result[f'{var}_pearson_r'] = pearson_r
                    result[f'{var}_pearson_p'] = pearson_p
                    result[f'{var}_spearman_r'] = spearman_r
            results.append(result)

        correlation_df = pd.DataFrame(results).dropna(how='all', axis=1)
        print(f"Computed correlations for {len(correlation_df)} spatial cells.")
        return correlation_df

    def identify_best_predictor(self, correlation_df: pd.DataFrame) -> Tuple[Dict, List]:
        """
        Ranks environmental variables based on their predictive power.
        A composite score is calculated from mean correlation, max correlation,
        statistical significance, and directional consistency.

        Args:
            correlation_df (pd.DataFrame): The output from `compute_multi_variable_correlations`.

        Returns:
            Tuple[Dict, List]: A dictionary of detailed metrics and a list of sorted predictors.
        """
        print("\n" + "="*70)
        print("IDENTIFYING BEST PREDICTOR (TIME DOMAIN)")
        print("="*70)
        env_vars = [col.replace('_pearson_r', '') for col in correlation_df.columns if col.endswith('_pearson_r')]
        predictor_quality = {}

        for var in env_vars:
            r_col, p_col = f'{var}_pearson_r', f'{var}_pearson_p'
            valid_corrs = correlation_df[r_col].dropna()
            if valid_corrs.empty:
                continue

            mean_abs_corr = valid_corrs.abs().mean()
            max_abs_corr = valid_corrs.abs().max()
            significant_fraction = (correlation_df[p_col] < 0.05).mean()
            consistent_direction = max((valid_corrs > 0).mean(), (valid_corrs < 0).mean())
            
            # Weighted score for ranking
            score = (mean_abs_corr * 0.4 + max_abs_corr * 0.2 + 
                     significant_fraction * 0.2 + consistent_direction * 0.2)

            predictor_quality[var] = {
                'mean_abs_correlation': mean_abs_corr,
                'max_abs_correlation': max_abs_corr,
                'significant_fraction': significant_fraction,
                'consistent_direction': consistent_direction,
                'composite_score': score
            }

        sorted_predictors = sorted(predictor_quality.items(), key=lambda x: x[1]['composite_score'], reverse=True)
        
        print("\nPredictor Quality Ranking:")
        print("-" * 70)
        for i, (var, metrics) in enumerate(sorted_predictors, 1):
            print(f"{i}. {var.upper()}")
            print(f"  Composite Score:      {metrics['composite_score']:.3f}")
            print(f"  Mean |Correlation|:   {metrics['mean_abs_correlation']:.3f}")
            print(f"  Consistency:          {metrics['consistent_direction']*100:.1f}%")
            print(f"  % Significant (p<0.05): {metrics['significant_fraction']*100:.1f}%")
            print()
            
        return predictor_quality, sorted_predictors
    
    def visualize_multi_variable_results(self, correlation_df: pd.DataFrame, top_n: int = 3):
        """
        Creates plots for the top N predictors from the time-domain analysis.
        Generates a map of spatial correlations and a histogram of correlation values.

        Args:
            correlation_df (pd.DataFrame): Correlation results.
            top_n (int): The number of top predictors to visualize.

        Returns:
            matplotlib.figure.Figure: The generated figure object.
        """
        import matplotlib.pyplot as plt
        
        _, sorted_predictors = self.identify_best_predictor(correlation_df)
        top_vars = [var for var, _ in sorted_predictors[:top_n]]
        if not top_vars:
             print("No predictors to visualize.")
             return None

        n_vars = len(top_vars)
        fig, axes = plt.subplots(2, n_vars, figsize=(6 * n_vars, 10), squeeze=False)

        for i, var in enumerate(top_vars):
            r_col = f'{var}_pearson_r'
            valid_df = correlation_df.dropna(subset=[r_col])

            # Spatial Plot
            ax1 = axes[0, i]
            if not valid_df.empty:
                scatter = ax1.scatter(valid_df['lon'], valid_df['lat'], c=valid_df[r_col], 
                                      s=100, cmap='RdBu_r', vmin=-1, vmax=1, edgecolors='k')
                fig.colorbar(scatter, ax=ax1, label='Pearson r')
            ax1.set_title(f'{var.upper()}\nSpatial Correlation')
            ax1.set_xlabel('Longitude')
            ax1.set_ylabel('Latitude')

            # Histogram
            ax2 = axes[1, i]
            if not valid_df.empty:
                ax2.hist(valid_df[r_col], bins=20, edgecolor='k', alpha=0.7)
                ax2.axvline(0, color='r', linestyle='--')
                ax2.text(0.05, 0.95, f"Mean: {valid_df[r_col].mean():.3f}\nStd: {valid_df[r_col].std():.3f}",
                         transform=ax2.transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax2.set_title(f'{var.upper()}\nCorrelation Distribution')
            ax2.set_xlabel('Pearson Correlation')
            ax2.set_ylabel('Cell Count')

        plt.tight_layout()
        plt.show()
        return fig

    def frequency_domain_coherence_analysis(self, min_observations: int = 20, plot: bool = True) -> pd.DataFrame:
        """
        Performs a coherence analysis between shark counts and environmental variables.
        Coherence measures the correlation between two time series at different frequencies.

        Args:
            min_observations (int): Minimum length of a time series for analysis.
            plot (bool): If True, generates a plot of the mean coherence spectra.

        Returns:
            pd.DataFrame: A DataFrame containing coherence values for each variable,
                          frequency, and grid cell.
        """
        print("\n" + "="*70)
        print("FREQUENCY DOMAIN COHERENCE ANALYSIS")
        print("="*70)
        
        variables = [c for c in self.aligned_data.columns if c not in ['time_bin', 'lat_bin', 'lon_bin', 'shark_count']]
        results = []

        for (lat, lon), group in self.aligned_data.groupby(['lat_bin', 'lon_bin']):
            if len(group) < min_observations:
                continue
            
            group = group.sort_values('time_bin')
            shark_ts = group['shark_count'].values

            for var in variables:
                env_ts = group[var].values
                mask = ~np.isnan(shark_ts) & ~np.isnan(env_ts)
                if mask.sum() < min_observations or np.std(shark_ts[mask]) == 0 or np.std(env_ts[mask]) == 0:
                    continue

                x, y = shark_ts[mask], env_ts[mask]
                try:
                    nperseg = min(32, len(x))
                    f, Cxy = signal.coherence(x, y, fs=1, nperseg=nperseg)
                    for i in range(len(f)):
                        results.append({
                            'lat': lat, 'lon': lon, 'variable': var,
                            'frequency': f[i], 'coherence': Cxy[i],
                            'n_observations': len(x)
                        })
                except Exception as e:
                    print(f"  Coherence failed for {var} at ({lat},{lon}): {e}")

        df = pd.DataFrame(results)
        if df.empty:
            print("Could not compute coherence for any cells. Check data quality or `min_observations`.")
            return df
        
        print(f"Computed coherence spectra for {len(df['variable'].unique())} variables across applicable cells.")

        if plot:
            self.visualize_multi_variable_results_frequency(df)

        return df

    def visualize_multi_variable_results_frequency(self, freq_df: pd.DataFrame, top_n: int = 3):
        """
        Creates plots for the top N predictors from the frequency-domain analysis.
        Generates a coherence distribution histogram and a mean coherence spectrum plot.

        Args:
            freq_df (pd.DataFrame): The coherence results dataframe.
            top_n (int): The number of top predictors to visualize.

        Returns:
            matplotlib.figure.Figure: The generated figure object.
        """
        import matplotlib.pyplot as plt
        
        # Rank variables by mean coherence to find the top predictors
        ranking = freq_df.groupby('variable')['coherence'].mean().sort_values(ascending=False)
        top_vars = ranking.index[:top_n].tolist()
        
        if not top_vars:
            print("No frequency-domain results to visualize.")
            return None

        print("\n" + "="*70)
        print("IDENTIFYING BEST PREDICTOR (FREQUENCY DOMAIN)")
        print("="*70)
        print(ranking)
        
        n_vars = len(top_vars)
        fig, axes = plt.subplots(2, n_vars, figsize=(6 * n_vars, 10), squeeze=False)

        for i, var in enumerate(top_vars):
            sub_df = freq_df[freq_df['variable'] == var]

            # Coherence distribution plot
            ax1 = axes[0, i]
            coh_vals = sub_df['coherence'].dropna()
            ax1.hist(coh_vals, bins=30, color='skyblue', edgecolor='k', alpha=0.7)
            ax1.axvline(coh_vals.mean(), color='r', linestyle='--', label=f"Mean: {coh_vals.mean():.2f}")
            ax1.set_title(f"{var.upper()}\nCoherence Distribution")
            ax1.set_xlabel('Coherence')
            ax1.set_ylabel('Count')
            ax1.legend()
            
            # Mean coherence spectrum plot
            ax2 = axes[1, i]
            spectrum = sub_df.groupby('frequency')['coherence'].agg(['mean', 'std'])
            ax2.plot(spectrum.index, spectrum['mean'], label=f"Mean Spectrum: {var}")
            ax2.fill_between(spectrum.index, spectrum['mean'] - spectrum['std'],
                             spectrum['mean'] + spectrum['std'], alpha=0.2)
            ax2.set_ylabel('Coherence')
            ax2.set_xlabel('Frequency (cycles/bin)')
            ax2.set_title(f'{var.upper()}\nMean Coherence Spectrum')
            ax2.set_ylim(0, 1)
            ax2.legend()
        
        plt.tight_layout()
        plt.show()
        return fig


if __name__ == "__main__":
    analyzer = MultiVariableSharkAnalyzer(
        temporal_resolution='3D',  # Aggregate data into 3-day bins
        spatial_resolution=0.5,    # Aggregate data into 0.5x0.5 degree cells
        cache_dir='./data'         # Cache directory
    )

    try:
        print("="*70)
        print("MULTI-VARIABLE SHARK HABITAT ANALYSIS")
        print("="*70)
        print("\nThis script analyzes shark movement against environmental variables.")
        print("It will automatically download SST and calculate derived variables.")
        print("Place any manual downloads (chlorophyll, bathymetry, etc.) in the './data' directory.")
        print("\nThe analysis ranks variables by predictive power in both time and frequency domains.\n")

        # Run the full analysis on a public whale shark dataset from Movebank
        results = analyzer.run_full_analysis(
            study_id=1153270750, 
            freq_min_observations=15 # Need at least 15 time points in a cell for frequency analysis
        )

        # Save the results
        if results:
            output_dir = Path('results')
            output_dir.mkdir(exist_ok=True)
            
            results['correlations'].to_csv(output_dir / 'time_domain_correlations.csv', index=False)
            results['coherence_results'].to_csv(output_dir / 'frequency_domain_coherence.csv', index=False)
            
            if results['figure_time_domain']:
                results['figure_time_domain'].savefig(output_dir / 'time_domain_analysis.png', dpi=300)
            if results['figure_frequency_domain']:
                results['figure_frequency_domain'].savefig(output_dir / 'frequency_domain_analysis.png', dpi=300)
                
            print(f"\nAnalysis complete. Results saved to the '{output_dir}' directory.")

    except Exception as e:
        import traceback
        print("\n" + "="*70)
        print(f"AN ERROR OCCURRED: {e}")
        print("="*70)
        traceback.print_exc()