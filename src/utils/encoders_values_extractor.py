import pickle
import os
import json

script_dir = os.path.dirname(__file__)

class EncodersValuesExtractor:
  def __init__(self, data_paths: dict, output_path: str):
    self.data_paths = data_paths
    self.output_path = output_path
    
  def extract_encoders_values(self):
    encoders_values = {}
    
    # Vérifier et charger les fichiers d'encoders
    for key, file_path in self.data_paths.items():
      if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
          encoder = pickle.load(file)
          encoders_values[key] = list(encoder.classes_)
      else:
        print(f"File not found: {file_path}")
    
    # Vérifier si le répertoire de sortie existe
        output_dir = os.path.dirname(self.output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)  # Créer le répertoire manquant
    
    # Sauvegarder les valeurs dans un fichier JSON
    with open(self.output_path, 'w', encoding='utf-8') as json_file:
      json.dump(encoders_values, json_file, indent=4, ensure_ascii=False)
    
    print(f"Encoders values saved in: {self.output_path}")

if __name__ == '__main__':
  encoder_dir = os.path.join(script_dir, '../model')
  
  encoder_files = {
    'carmodel': os.path.join(encoder_dir, 'dataset_encoder_carmodel.pkl'),
    'energy': os.path.join(encoder_dir, 'dataset_encoder_energie.pkl'),
    'gearbox': os.path.join(encoder_dir, 'dataset_encoder_boite.pkl'),
    'color': os.path.join(encoder_dir, 'dataset_encoder_couleur.pkl')
  }

  output_json_path = os.path.join(encoder_dir, 'dataset_encoders_values.json')
  
  encoders_values_extractor = EncodersValuesExtractor(encoder_files, output_json_path)
  encoders_values_extractor.extract_encoders_values()