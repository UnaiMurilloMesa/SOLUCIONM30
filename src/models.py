"""
Models Module.
Wrappers for Machine Learning models (Random Forest, XGBoost) for traffic prediction.
"""
from sklearn.ensemble import RandomForestRegressor
# import xgboost as xgb # Uncomment when installed
import numpy as np

class TrafficModel:
    """
    Wrapper class for traffic prediction models.
    """
    
    def __init__(self, model_type: str = 'rf'):
        """
        Initialize the model.
        
        Args:
            model_type (str): 'rf' for Random Forest, 'xgb' for XGBoost.
        """
        self.model_type = model_type
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        if self.model_type == 'rf':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.model_type == 'xgb':
            # self.model = xgb.XGBRegressor(objective='reg:squarederror')
            pass
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, X, y):
        """
        Train the model.
        """
        if self.model:
            self.model.fit(X, y)

    def predict(self, X):
        """
        Make predictions.
        """
        if self.model:
            return self.model.predict(X)
        return np.zeros(len(X))
