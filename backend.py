"""
Solar Power Prediction Backend
A backend service for predicting solar power generation based on weather data and historical patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SolarPowerPredictor:
    """
    A class to handle solar power prediction based on weather data and historical patterns.
    """
    
    def __init__(self):
        self.model = None
        self.feature_columns = [
            'temperature', 'humidity', 'wind_speed', 'cloud_cover', 
            'solar_irradiance', 'hour', 'day_of_year', 'season'
        ]
        self.is_trained = False
        
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the input data for model training/prediction.
        
        Args:
            data: DataFrame containing weather and solar data
            
        Returns:
            Preprocessed DataFrame
        """
        try:
            # Create a copy to avoid modifying original data
            processed_data = data.copy()
            
            # Handle missing values
            processed_data = processed_data.fillna(processed_data.mean())
            
            # Extract time-based features
            if 'timestamp' in processed_data.columns:
                processed_data['timestamp'] = pd.to_datetime(processed_data['timestamp'])
                processed_data['hour'] = processed_data['timestamp'].dt.hour
                processed_data['day_of_year'] = processed_data['timestamp'].dt.dayofyear
                processed_data['season'] = processed_data['timestamp'].dt.month % 12 // 3
            
            # Normalize numerical features
            numerical_cols = ['temperature', 'humidity', 'wind_speed', 'cloud_cover', 'solar_irradiance']
            for col in numerical_cols:
                if col in processed_data.columns:
                    processed_data[col] = (processed_data[col] - processed_data[col].mean()) / processed_data[col].std()
            
            logger.info("Data preprocessing completed successfully")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise
    
    def train_model(self, training_data: pd.DataFrame) -> Dict:
        """
        Train the solar power prediction model.
        
        Args:
            training_data: DataFrame containing training data with features and target
            
        Returns:
            Dictionary containing training results and model metrics
        """
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, r2_score
            
            # Preprocess the training data
            processed_data = self.preprocess_data(training_data)
            
            # Prepare features and target
            X = processed_data[self.feature_columns]
            y = processed_data['solar_power_output'] if 'solar_power_output' in processed_data.columns else processed_data['power_generated']
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train the model
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            
            # Make predictions and calculate metrics
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.is_trained = True
            
            results = {
                'status': 'success',
                'mse': mse,
                'r2_score': r2,
                'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_)),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            logger.info(f"Model training completed. R² Score: {r2:.4f}, MSE: {mse:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, weather_data: Dict) -> Dict:
        """
        Predict solar power output based on weather conditions.
        
        Args:
            weather_data: Dictionary containing weather parameters
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            if not self.is_trained:
                return {'status': 'error', 'message': 'Model not trained yet'}
            
            # Convert input to DataFrame
            input_data = pd.DataFrame([weather_data])
            
            # Preprocess the input data
            processed_data = self.preprocess_data(input_data)
            
            # Ensure all required features are present
            missing_features = set(self.feature_columns) - set(processed_data.columns)
            if missing_features:
                return {'status': 'error', 'message': f'Missing features: {missing_features}'}
            
            # Make prediction
            X = processed_data[self.feature_columns]
            prediction = self.model.predict(X)[0]
            
            # Calculate confidence interval (simplified)
            confidence = 0.85  # This could be calculated based on model uncertainty
            
            result = {
                'status': 'success',
                'predicted_power': float(prediction),
                'confidence': confidence,
                'timestamp': datetime.now().isoformat(),
                'input_features': weather_data
            }
            
            logger.info(f"Prediction completed: {prediction:.2f} kW")
            return result
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def get_model_info(self) -> Dict:
        """
        Get information about the current model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'is_trained': self.is_trained,
            'feature_columns': self.feature_columns,
            'model_type': 'RandomForestRegressor' if self.model else None,
            'last_updated': datetime.now().isoformat()
        }

class WeatherDataProcessor:
    """
    A class to handle weather data processing and API integration.
    """
    
    def __init__(self):
        self.api_key = None  # Should be set from environment variables
    
    def fetch_weather_data(self, location: str, days: int = 7) -> Dict:
        """
        Fetch weather data for a specific location.
        
        Args:
            location: Location name or coordinates
            days: Number of days to fetch data for
            
        Returns:
            Dictionary containing weather data
        """
        try:
            # This is a placeholder for actual weather API integration
            # In a real implementation, you would integrate with services like:
            # - OpenWeatherMap API
            # - WeatherAPI
            # - National Weather Service API
            
            logger.info(f"Fetching weather data for {location} for {days} days")
            
            # Mock data for demonstration
            mock_data = {
                'location': location,
                'forecast': [
                    {
                        'date': (datetime.now() + timedelta(days=i)).isoformat(),
                        'temperature': np.random.normal(25, 5),
                        'humidity': np.random.normal(60, 10),
                        'wind_speed': np.random.normal(10, 3),
                        'cloud_cover': np.random.normal(30, 20),
                        'solar_irradiance': np.random.normal(500, 100)
                    }
                    for i in range(days)
                ]
            }
            
            return {'status': 'success', 'data': mock_data}
            
        except Exception as e:
            logger.error(f"Error fetching weather data: {str(e)}")
            return {'status': 'error', 'message': str(e)}

# API endpoints simulation
def create_api_endpoints():
    """
    Create API endpoints for the solar power prediction service.
    """
    predictor = SolarPowerPredictor()
    weather_processor = WeatherDataProcessor()
    
    def train_endpoint(training_data: Dict) -> Dict:
        """API endpoint for model training"""
        try:
            df = pd.DataFrame(training_data['data'])
            return predictor.train_model(df)
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def predict_endpoint(weather_data: Dict) -> Dict:
        """API endpoint for solar power prediction"""
        return predictor.predict(weather_data)
    
    def weather_endpoint(location: str, days: int = 7) -> Dict:
        """API endpoint for weather data"""
        return weather_processor.fetch_weather_data(location, days)
    
    def model_info_endpoint() -> Dict:
        """API endpoint for model information"""
        return predictor.get_model_info()
    
    return {
        'train': train_endpoint,
        'predict': predict_endpoint,
        'weather': weather_endpoint,
        'model_info': model_info_endpoint
    }

# Example usage
if __name__ == "__main__":
    # Initialize the predictor
    predictor = SolarPowerPredictor()
    
    # Example training data
    sample_data = {
        'data': [
            {
                'timestamp': '2024-01-01 12:00:00',
                'temperature': 25.5,
                'humidity': 60.0,
                'wind_speed': 10.0,
                'cloud_cover': 30.0,
                'solar_irradiance': 500.0,
                'solar_power_output': 45.2
            },
            {
                'timestamp': '2024-01-02 12:00:00',
                'temperature': 28.0,
                'humidity': 55.0,
                'wind_speed': 12.0,
                'cloud_cover': 20.0,
                'solar_irradiance': 600.0,
                'solar_power_output': 52.8
            }
        ]
    }
    
    # Create API endpoints
    api = create_api_endpoints()
    
    # Example usage
    print("Solar Power Prediction Backend")
    print("=" * 40)
    
    # Train the model
    print("Training model...")
    training_result = api['train'](sample_data)
    print(f"Training result: {training_result}")
    
    # Make a prediction
    print("\nMaking prediction...")
    weather_input = {
        'temperature': 26.0,
        'humidity': 58.0,
        'wind_speed': 11.0,
        'cloud_cover': 25.0,
        'solar_irradiance': 550.0
    }
    prediction_result = api['predict'](weather_input)
    print(f"Prediction result: {prediction_result}")
    
    # Get model info
    print("\nModel information:")
    model_info = api['model_info']()
    print(f"Model info: {model_info}")
