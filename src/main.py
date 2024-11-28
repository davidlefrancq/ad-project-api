from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from client import FrenchSecondHandCarsClient
from typing import Dict, Any
import uvicorn

app = FastAPI()

# Configuration du CORS
app.add_middleware(
    CORSMiddleware,
    # Liste des origines autorisées (vous pouvez ajuster selon vos besoins)
    allow_origins=["http://localhost:5173", "http://57.128.24.53:8294"],  # Origine de votre app Vue.js
    allow_credentials=True,
    allow_methods=["*"],  # Autorise toutes les méthodes HTTP
    allow_headers=["*"],  # Autorise tous les headers
)

@app.get("/")
async def read_root():
  return {"status": "ok", "message": "API is working fine."}

@app.post("/predict")
async def predict_price(car_data: Dict[str, Any]):
  """Predict the price of a car based on its features.

  Args:
    car_data (Dict[str, Any]): A dictionary containing the car features in French:
    {
      'carmodel': str,                      # Ex: 'RENAULT TWINGO 3'
      'année': int,                         # Ex: 2020
      'kilométragecompteur': int,           # Ex: 27297
      'énergie': str,                       # Ex: 'essence'
      'boîtedevitesse': str,                # Ex: 'mécanique'
      'couleurextérieure': str,             # Ex: 'gris'
      'nombredeportes': int,                # Ex: 5
      'premièremain(déclaratif)': bool,     # Ex: True
      'puissancefiscale': int,              # Ex: 5
      'couleurextérieure_métallisée': bool  # Ex: False
    }

  Returns:
    float: price of the car
  """
  errors = []
  # Check data: carmodel
  if 'carmodel' not in car_data:
    errors.append("Missing 'carmodel' field.")
  # Check data: year
  if 'year' not in car_data:
    errors.append("Missing 'year' field.")
  # Check data: mileage
  if 'mileage' not in car_data:
    errors.append("Missing 'mileage' field.")
  # Check data: energy
  if 'energy' not in car_data:
    errors.append("Missing 'energy' field.")
  # Check data: gearbox
  if 'gearbox' not in car_data:
    errors.append("Missing 'gearbox' field.")
  # Check data: color
  if 'color' not in car_data:
    errors.append("Missing 'color' field.")
  # Check data: doors
  if 'doors' not in car_data:
    errors.append("Missing 'doors' field.")
  # Check data: first_hand
  if 'first_hand' not in car_data:
    errors.append("Missing 'first_hand' field.")
  # Check data: power
  if 'power' not in car_data:
    errors.append("Missing 'power' field.")
  # Check data: metallic_color
  if 'metallic_color' not in car_data:
    errors.append("Missing 'metallic_color' field.")
    
  if len(errors) > 0:
    raise HTTPException(status_code=400, detail=errors)
  
  # Initialisation du client
  client = FrenchSecondHandCarsClient()

  # dict_values['carmodel'] = car_data['carmodel']
  # dict_values['année'] = car_data['year']
  # dict_values['kilométragecompteur'] = car_data['mileage']
  # dict_values['énergie'] = car_data['energy']
  # dict_values['boîtedevitesse'] = car_data['gearbox']
  # dict_values['couleurextérieure'] = car_data['color']
  # dict_values['nombredeportes'] = car_data['doors']
  # dict_values['premièremain(déclaratif)'] = car_data['first_hand']
  # dict_values['puissancefiscale'] = car_data['power']
  # dict_values['couleurextérieure_métallisée'] = car_data['metallic_color']
  dict_values = {
    'carmodel': car_data['carmodel'],
    'année': car_data['year'],
    'kilométragecompteur': car_data['mileage'],
    'énergie': car_data['energy'],
    'boîtedevitesse': car_data['gearbox'],
    'couleurextérieure': car_data['color'],
    'nombredeportes': car_data['doors'],
    'premièremain(déclaratif)': car_data['first_hand'],
    'puissancefiscale': car_data['power'],
    'couleurextérieure_métallisée': car_data['metallic_color']    
  }
  
  # Prédiction
  try:
    prediction = client.predict(car_data)
    return {"price": prediction}
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
  uvicorn.run("main:app", host="0.0.0.0", port=80, reload=False)