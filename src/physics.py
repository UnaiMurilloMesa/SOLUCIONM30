"""
Traffic Physics module.
Handles Fundamental Diagram calculations and traffic state estimation.
"""
import pandas as pd
import numpy as np

class TrafficPhysics:
    """
    Class for handling traffic flow physics calculations.
    Based on the Fundamental Diagram of Traffic Flow (Greenshields, etc.).
    """
    
    @staticmethod
    def calculate_density(intensity: float, speed: float) -> float:
        """
        Calculates traffic density based on flow (intensity) and speed.
        
        Formula: k = q / v
        
        Args:
            intensity (float): Traffic flow in vehicles/hour.
            speed (float): Average speed in km/h.
            
        Returns:
            float: Density in vehicles/km. Returns 0 if speed is 0 to avoid division by zero.
        """
        if speed <= 0:
            return 0.0
        return intensity / speed

    @staticmethod
    def get_fundamental_diagram(df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares data for plotting the Fundamental Diagram (Flow vs Density).
        
        Args:
            df (pd.DataFrame): DataFrame containing 'intensity' and 'speed' columns.
            
        Returns:
            pd.DataFrame: DataFrame with an added 'density' column, ready for plotting.
        """
        if 'intensity' not in df.columns or 'speed' not in df.columns:
            raise ValueError("DataFrame must contain 'intensity' and 'speed' columns.")
        
        df_fd = df.copy()
        # Vectorized calculation for efficiency
        # Handle division by zero by replacing 0 speed with NaN or inf, then handling
        # For simplicity here, we assume cleaned data or handle row-wise
        df_fd['density'] = df_fd.apply(
            lambda row: row['intensity'] / row['speed'] if row['speed'] > 0 else 0, axis=1
        )
        
        return df_fd
