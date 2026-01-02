"""
Optimizer Module.
Contains the logic for determining the Optimal Variable Speed Limit.
"""
from .config import CRITICAL_DENSITY_THRESHOLD

def calculate_optimal_speed(predicted_density: float, critical_density: float = CRITICAL_DENSITY_THRESHOLD) -> int:
    """
    Calculates the optimal speed limit based on predicted traffic density.
    
    Logic:
        If the predicted density exceeds the critical threshold, lower the speed limit
        to reduce inflow and dampen the accordion effect (shockwaves).
        Otherwise, maintain standard speed limit.
        
    Args:
        predicted_density (float): The estimated/predicted traffic density (occupancy or veh/km).
        critical_density (float): The threshold density where flow becomes unstable.
        
    Returns:
        int: Optimal speed limit in km/h (70 or 90).
    """
    if predicted_density > critical_density:
        return 70
    else:
        return 90
