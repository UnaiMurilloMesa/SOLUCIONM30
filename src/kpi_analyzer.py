"""
KPI Analyzer Module
Calculates and analyzes Key Performance Indicators for traffic optimization.
Provides hourly metrics comparison between Reality and Digital Twin.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class HourlyKPIAnalyzer:
    """
    Analyzes and calculates KPIs for traffic improvement by time windows.
    Compares Reality vs Digital Twin performance metrics.
    """

    def __init__(self, reality_data: pd.DataFrame, twin_data: pd.DataFrame):
        """
        Initialize KPI Analyzer with reality and digital twin data.

        Args:
            reality_data: DataFrame with columns ['fecha', 'vmed', 'intensidad', 'density']
            twin_data: DataFrame with columns ['fecha', 'simulated_speed', 'intensidad_opt', 'simulated_density']
        """
        self.reality = reality_data.copy()
        self.twin = twin_data.copy()

        # Ensure both have hour column
        if "fecha" in self.reality.columns:
            self.reality["hour"] = self.reality["fecha"].dt.hour
        if "fecha" in self.twin.columns:
            self.twin["hour"] = self.twin["fecha"].dt.hour

    def calculate_hourly_metrics(self) -> pd.DataFrame:
        """
        Calculate aggregated metrics by hour.

        Returns:
            DataFrame with hourly metrics including improvement percentages
        """
        # Group by hour - Reality
        reality_hourly = (
            self.reality.groupby("hour")
            .agg({"vmed": "mean", "intensidad": "mean", "density": "mean"})
            .reset_index()
        )

        # Group by hour - Digital Twin
        twin_hourly = (
            self.twin.groupby("hour")
            .agg(
                {
                    "simulated_speed": "mean",
                    "intensidad_opt": "mean",
                    "simulated_density": "mean",
                }
            )
            .reset_index()
        )

        # Merge both datasets
        metrics = pd.merge(reality_hourly, twin_hourly, on="hour", how="inner")

        # Calculate improvements
        metrics["speed_improvement_pct"] = (
            (metrics["simulated_speed"] - metrics["vmed"]) / metrics["vmed"] * 100
        )

        metrics["flow_improvement_pct"] = (
            (metrics["intensidad_opt"] - metrics["intensidad"])
            / metrics["intensidad"]
            * 100
        )

        metrics["density_reduction_pct"] = (
            (metrics["density"] - metrics["simulated_density"])
            / metrics["density"]
            * 100
        )

        return metrics

    def get_last_hour_improvement(self, current_hour: int) -> Dict[str, float]:
        """
        Get improvement metrics for the last complete hour.

        Args:
            current_hour: Current simulation hour (0-23)

        Returns:
            Dictionary with improvement metrics
        """
        # Filter data for current hour
        reality_current_hour = self.reality[self.reality["hour"] == current_hour]
        twin_current_hour = self.twin[self.twin["hour"] == current_hour]

        if len(reality_current_hour) == 0 or len(twin_current_hour) == 0:
            return {
                "speed_improvement": 0.0,
                "reality_speed": 0.0,
                "twin_speed": 0.0,
                "hour": current_hour,
            }

        reality_avg_speed = reality_current_hour["vmed"].mean()
        twin_avg_speed = twin_current_hour["simulated_speed"].mean()

        # Avoid division by zero
        if reality_avg_speed > 0:
            improvement = (twin_avg_speed - reality_avg_speed) / reality_avg_speed * 100
        else:
            improvement = 0.0

        return {
            "speed_improvement": improvement,
            "reality_speed": reality_avg_speed,
            "twin_speed": twin_avg_speed,
            "hour": current_hour,
        }

    def get_cumulative_improvement(self, up_to_hour: int) -> Dict[str, float]:
        """
        Calculate cumulative improvement from start up to specified hour.

        Args:
            up_to_hour: Hour up to which to calculate (0-23)

        Returns:
            Dictionary with cumulative metrics
        """
        reality_subset = self.reality[self.reality["hour"] <= up_to_hour]
        twin_subset = self.twin[self.twin["hour"] <= up_to_hour]

        if len(reality_subset) == 0 or len(twin_subset) == 0:
            return {
                "cumulative_improvement": 0.0,
                "hours_analyzed": 0,
                "reality_avg": 0.0,
                "twin_avg": 0.0,
            }

        reality_avg = reality_subset["vmed"].mean()
        twin_avg = twin_subset["simulated_speed"].mean()

        if reality_avg > 0:
            improvement = (twin_avg - reality_avg) / reality_avg * 100
        else:
            improvement = 0.0

        return {
            "cumulative_improvement": improvement,
            "hours_analyzed": up_to_hour + 1,
            "reality_avg": reality_avg,
            "twin_avg": twin_avg,
        }

    def generate_improvement_history(
        self, current_hour: int, window_size: int = 6
    ) -> List[float]:
        """
        Generate improvement history for the last N hours (for sparkline charts).

        Args:
            current_hour: Current hour
            window_size: Number of hours to include

        Returns:
            List of improvement percentages
        """
        improvements = []

        start_hour = max(0, current_hour - window_size + 1)

        for hour in range(start_hour, current_hour + 1):
            reality_hour = self.reality[self.reality["hour"] == hour]
            twin_hour = self.twin[self.twin["hour"] == hour]

            if len(reality_hour) > 0 and len(twin_hour) > 0:
                r_speed = reality_hour["vmed"].mean()
                t_speed = twin_hour["simulated_speed"].mean()

                if r_speed > 0:
                    improvement = (t_speed - r_speed) / r_speed * 100
                else:
                    improvement = 0.0

                improvements.append(improvement)
            else:
                improvements.append(0.0)

        return improvements

    def get_flow_metrics(self, current_hour: int) -> Dict[str, float]:
        """
        Get flow (intensity) metrics for the last complete hour.

        Args:
            current_hour: Current simulation hour (0-23)

        Returns:
            Dictionary with flow metrics
        """
        if current_hour == 0:
            return {"flow_improvement": 0.0, "reality_flow": 0.0, "twin_flow": 0.0}

        last_hour = current_hour - 1

        reality_last_hour = self.reality[self.reality["hour"] == last_hour]
        twin_last_hour = self.twin[self.twin["hour"] == last_hour]

        if len(reality_last_hour) == 0 or len(twin_last_hour) == 0:
            return {"flow_improvement": 0.0, "reality_flow": 0.0, "twin_flow": 0.0}

        reality_avg_flow = reality_last_hour["intensidad"].mean()
        twin_avg_flow = twin_last_hour["intensidad_opt"].mean()

        if reality_avg_flow > 0:
            improvement = (twin_avg_flow - reality_avg_flow) / reality_avg_flow * 100
        else:
            improvement = 0.0

        return {
            "flow_improvement": improvement,
            "reality_flow": reality_avg_flow,
            "twin_flow": twin_avg_flow,
        }

    def get_density_metrics(self, current_hour: int) -> Dict[str, float]:
        """
        Get density metrics for the last complete hour.

        Args:
            current_hour: Current simulation hour (0-23)

        Returns:
            Dictionary with density metrics (reduction is positive)
        """
        if current_hour == 0:
            return {
                "density_reduction": 0.0,
                "reality_density": 0.0,
                "twin_density": 0.0,
            }

        last_hour = current_hour - 1

        reality_last_hour = self.reality[self.reality["hour"] == last_hour]
        twin_last_hour = self.twin[self.twin["hour"] == last_hour]

        if len(reality_last_hour) == 0 or len(twin_last_hour) == 0:
            return {
                "density_reduction": 0.0,
                "reality_density": 0.0,
                "twin_density": 0.0,
            }

        reality_avg_density = reality_last_hour["density"].mean()
        twin_avg_density = twin_last_hour["simulated_density"].mean()

        # Reduction is positive (we want lower density)
        if reality_avg_density > 0:
            reduction = (
                (reality_avg_density - twin_avg_density) / reality_avg_density * 100
            )
        else:
            reduction = 0.0

        return {
            "density_reduction": reduction,
            "reality_density": reality_avg_density,
            "twin_density": twin_avg_density,
        }
