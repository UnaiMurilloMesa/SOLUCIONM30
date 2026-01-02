"""
Data Loader Module.
Responsible for ingesting raw data from CSV files or other sources.
"""
import pandas as pd
from typing import Optional
from pathlib import Path

def load_csv_data(filepath: Path) -> pd.DataFrame:
    """
    Loads traffic data from a CSV file.
    
    Args:
        filepath (Path): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded data from {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()
