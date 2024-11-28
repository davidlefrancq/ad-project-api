import os
import datetime
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any

script_dir = os.path.dirname(__file__)
current_model_path = os.path.join(script_dir, 'model/GradientBoosting_model.pkl')

@dataclass
class CarFeatures:
    carmodel: str
    year: int
    mileage: int
    energy: str
    gearbox: str
    color: str
    doors: int
    first_hand: bool
    power: int
    metallic_color: bool

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CarFeatures':
      return cls(
          carmodel=data['carmodel'].lower(),
          year=data['year'],
          mileage=data['mileage'],
          energy=data['energy'].lower(),
          gearbox=data['gearbox'].lower(),
          color=data['color'].lower(),
          doors=data['doors'],
          first_hand=data['first_hand'],
          power=data['power'],
          metallic_color=data['metallic_color']
      )        

class FrenchSecondHandCarsClient:
  def __init__(self, model_path=current_model_path):
    self.encoders = {
      'carmodel': None,
      'energy': None,
      'gearbox': None,
      'color': None
    }
    self.scalers = {
      'year': None,
      'mileage': None,
      'doors': None,
      'power': None,
      'price': None
    }
    self.model = None
    self._load_model(model_path)
    self._load_label_encoders()
    self._load_scalers()
  
  def _load_model(self, model_path):
    if not os.path.exists(model_path):
      raise FileNotFoundError(f'File {model_path} not found.')
    model_path = os.path.join(script_dir, model_path)
    self.model = joblib.load(model_path)
  
  def _load_label_encoders(self):
    self.encoders["gearbox"] = joblib.load(os.path.join(script_dir, 'model/dataset_encoder_boite.pkl'))
    self.encoders["carmodel"] = joblib.load(os.path.join(script_dir, 'model/dataset_encoder_carmodel.pkl'))
    self.encoders['color'] = joblib.load(os.path.join(script_dir, 'model/dataset_encoder_couleur.pkl'))
    self.encoders['energy'] = joblib.load(os.path.join(script_dir, 'model/dataset_encoder_energie.pkl'))
  
  def _load_scalers(self):
    self.scalers["mileage"] = joblib.load(os.path.join(script_dir, 'model/dataset_scaler_kilometer.pkl'))
    self.scalers["doors"] = joblib.load(os.path.join(script_dir, 'model/dataset_scaler_nombredeportes.pkl'))
    self.scalers["power"] = joblib.load(os.path.join(script_dir, 'model/dataset_scaler_power.pkl'))
    self.scalers["price"] = joblib.load(os.path.join(script_dir, 'model/dataset_scaler_price.pkl'))
    self.scalers["year"] = joblib.load(os.path.join(script_dir, 'model/dataset_scaler_year.pkl'))
    
  def _features_to_array(self, features: CarFeatures) -> np.ndarray:
    # Créer les DataFrames individuels avec les bons noms de colonnes
    year_df = pd.DataFrame([[features.year]], columns=['année'])
    mileage_df = pd.DataFrame([[features.mileage]], columns=['kilométragecompteur'])
    doors_df = pd.DataFrame([[features.doors]], columns=['nombredeportes'])
    power_df = pd.DataFrame([[features.power]], columns=['puissancefiscale'])
    
    transformed_data = {
      'carmodel': self.encoders['carmodel'].transform([features.carmodel])[0],
      'year': self.scalers['year'].transform(year_df)[0][0],
      'mileage': self.scalers['mileage'].transform(mileage_df)[0][0],
      'energy': self.encoders['energy'].transform([features.energy])[0],
      'gearbox': self.encoders['gearbox'].transform([features.gearbox])[0],
      'color': self.encoders['color'].transform([features.color])[0],
      'doors': self.scalers['doors'].transform(doors_df)[0][0],
      'first_hand': 1 if features.first_hand else 0,
      'power': self.scalers['power'].transform(power_df)[0][0],
      'metallic': 1 if features.metallic_color else 0
    }

    # Créer un DataFrame pour le modèle avec l'ordre correct des colonnes
    features_df = pd.DataFrame([list(transformed_data.values())], columns=list(transformed_data.keys()))
    return features_df.values

    
  def predict(self, car_data: Dict[str, Any]) -> float:
    """
    Predict the price of a car based on its features.
    
    Args:
      car_data (Dict[str, Any]): A dictionary containing the car features:
      {
        'carmodel': str,        # Ex: 'RENAULT TWINGO 3'
        'year': int,            # Ex: 2020
        'mileage': int,         # Ex: 27297
        'energy': str,          # Ex: 'essence'
        'gearbox': str,         # Ex: 'mécanique'
        'color': str,           # Ex: 'gris'
        'doors': int,           # Ex: 5
        'first_hand': bool,     # Ex: True
        'power': int,           # Ex: 5
        'metallic_color': bool  # Ex: False
      }

    Returns:
      float: price of the car
    """
    features = CarFeatures.from_dict(car_data)
    features_array = self._features_to_array(features)
    prediction = self.model.predict(features_array.reshape(1, -1))
    return float(self.scalers['price'].inverse_transform(prediction.reshape(-1, 1))[0][0])

if __name__ == '__main__':
  datestart = datetime.datetime.now()
  
  # Client initialization
  client = FrenchSecondHandCarsClient()
  
  # Data
  entry = {
    'carmodel': 'RENAULT TWINGO 3'.lower(),
    'year': 2020,
    'mileage': 27297,
    'energy': 'essence',
    'gearbox': 'mécanique',
    'color': 'gris',
    'doors': 5,
    'first_hand': True,
    'power': 5,
    'metallic_color': False
  }
  real_value = 11080
  
  # Predict
  prediction = client.predict(entry)
  
  # Show prediction vs real value
  print(f'\nPrediction: {prediction}')
  print(f'Real value: {real_value}')
  print(f'Precision: {round(100 - (abs(prediction - real_value) / real_value) * 100, 2)}%')
  
  dateend = datetime.datetime.now()
