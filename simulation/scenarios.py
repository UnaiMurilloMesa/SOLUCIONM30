"""
Scenarios Module.
Definitions of specific test scenarios (e.g., Morning Rush Hour, Accident, Holiday).
"""

class ScenarioDefinition:
    """
    Defines parameters for a specific traffic scenario.
    """
    def __init__(self, name: str, start_time: str, end_time: str, sensor_id: str):
        self.name = name
        self.start_time = start_time
        self.end_time = end_time
        self.sensor_id = sensor_id

# Example Scenarios
SCENARIOS = {
    "Morning_Rush": ScenarioDefinition("Morning Rush", "07:00", "10:00", "PM-30-01"),
    "Evening_Rush": ScenarioDefinition("Evening Rush", "17:00", "20:00", "PM-30-01"),
}
