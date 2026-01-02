"""
Simulation Engine.
Compares Real historical scenarios vs Optimized scenarios (Digital Twin).
"""
import pandas as pd
from src.optimizer import calculate_optimal_speed
from src.physics import TrafficPhysics

class DigitalTwinEngine:
    """
    Engine to run the simulation comparing baseline (historical) vs optimized speed limits.
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.physics = TrafficPhysics()

    def run_simulation(self):
        """
        Iterates through the data time steps and applies the optimization logic.
        
        Returns:
            pd.DataFrame: Results with 'speed_limit_optimized' and estimated impact.
        """
        results = self.data.copy()
        
        # Placeholder logic for applying optimization row by row
        # In a real scenario, this might update future states based on current actions.
        results['optimized_limit'] = results.apply(
            lambda row: calculate_optimal_speed(row.get('density', 0)), axis=1
        )
        
        return results
