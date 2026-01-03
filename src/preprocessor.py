"""
Preprocessor Module.
Handles data cleaning, feature engineering, and preparation for ML models.
Includes lag generation and date parsing.
"""
import pandas as pd
import numpy as np
from src.config import M30_EAST_SENSORS

class DataPreprocessor:
    """
    Class to handle preprocessing of traffic data.
    """
    
    def __init__(self, sensor_ids=None):
        self.sensor_ids = sensor_ids

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs basic cleaning: filtering sensors, handling NaNs, logic checks.
        
        Args:
            df (pd.DataFrame): Raw data.
            
        Returns:
            pd.DataFrame: Cleaned data.
        """
        if df.empty:
            return df

        # 1. Filter by Sensor ID (if specified)
        if self.sensor_ids is not None:
            valid_ids = set(self.sensor_ids)
            mask = df['id'].isin(valid_ids)
            df_clean = df[mask].copy()
        else:
            df_clean = df.copy()
        
        # 2. Parse Dates
        # Format in csv: "YYYY-MM-DD HH:MM:SS"
        if 'fecha' in df_clean.columns:
            df_clean['fecha'] = pd.to_datetime(df_clean['fecha'], errors='coerce')
        
        # 3. Sort by time
        df_clean = df_clean.sort_values(by=['id', 'fecha'])

        # 4. Remove logical errors
        # Speed (vmed) < 0 or Intensity (intensidad) < 0
        df_clean = df_clean[
            (df_clean['vmed'] >= 0) & 
            (df_clean['intensidad'] >= 0)
        ]
        
        # 5. Handle Missing Values
        # Interpolate linear per group (sensor)
        # We need to set index to time to interpolate properly, or just use linear on columns
        # Ideally, we should pivot or group by ID.
        # Simple approach: Group by ID, then apply interpolation.
        df_clean['vmed'] = df_clean.groupby('id')['vmed'].transform(lambda x: x.interpolate(method='linear'))
        df_clean['intensidad'] = df_clean.groupby('id')['intensidad'].transform(lambda x: x.interpolate(method='linear'))
        
        # Fill remaining NaNs (edges) with ffill/bfill or 0
        df_clean = df_clean.ffill().bfill()
        
        return df_clean

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates features for the ML model: Density, Time features.
        
        Args:
            df (pd.DataFrame): Cleaned data.
            
        Returns:
            pd.DataFrame: Data with new features.
        """
        df = df.copy()
        
        # 1. Density (K) = Intensity (limit/hour) / Speed (km/h)
        # Result is vehicles/km.
        # FIX: Handle cases where speed is close to 0 to avoid exploding density.
        # We enforce a minimum effective speed for the calculation (e.g. 5 km/h)
        # and cap the maximum density to realistic values (e.g. 400 veh/km).
        
        # 1. Density (K) Calculation
        # Standard: K = Q / V
        # Edge Case: Stopped traffic (V->0, Q->0) => K should be High, but formula gives 0/0 or 0.
        
        # Method A: Q / V (clipped)
        effective_speed = df['vmed'].clip(lower=5.0)
        df['density_qv'] = df['intensidad'] / effective_speed

        # Method B: Occupancy-based (if available)
        # K ~ alpha * Occupancy. 
        # Tuning: Max Density (Jam) ~ 400 veh/km corresponds to Occupancy ~ 100%.
        # So factor approx 4. Let's use 3.5 as a conservative estimate for mixed traffic.
        if 'ocupacion' in df.columns:
            df['density_occ'] = df['ocupacion'] * 3.5
        else:
            df['density_occ'] = df['density_qv'] # Fallback

        # Hybrid Approach:
        # If Speed is "normal" (> 10 km/h), trust Q/V provided Q > 0.
        # If Speed is "low" (<= 10 km/h) OR Q is 0, trust Occupancy.
        
        def calculate_hybrid_density(row):
            v = row['vmed']
            q = row['intensidad']
            
            # If valid flow and speed, Q/V is most accurate physical measure
            if v > 10 and q > 0:
                return row['density_qv']
            else:
                # Low speed or zero flow -> Use Occupancy Proxy
                # If occupancy is missing (NaN/0), fallback to Q/V (which implies 0 density)
                # But if V is low and Occ is high, this saves us from "Ghost Empty Roads"
                return row.get('density_occ', row['density_qv'])

        df['density'] = df.apply(calculate_hybrid_density, axis=1)
        
        # Cap outliers
        df['density'] = df['density'].clip(upper=400.0)
        
        # 2. Time Features
        if 'fecha' in df.columns:
            df['hour'] = df['fecha'].dt.hour
            df['day_of_week'] = df['fecha'].dt.dayofweek
            df['month'] = df['fecha'].dt.month
            
        # 3. Predicted Density (t + 15min) -> Shift -1
        # Assuming data is sorted by id, fecha and 15 min intervals
        df['density_pred'] = df.groupby('id')['density'].shift(-1)
        # Fill last value with current
        df['density_pred'] = df['density_pred'].fillna(df['density'])
        
        return df
