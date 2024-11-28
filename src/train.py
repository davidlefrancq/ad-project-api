import os
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

script_dir = os.path.dirname(__file__)

class ModelTrainer:
  def __init__(self):
    self.models = {
      'LinearRegression': {
        'model': LinearRegression(),
        'params': {}  # Pas d'hyperparamètres à optimiser
      },
      'Ridge': {
        'model': Ridge(),
        'params': {}
      },
      'Lasso': {
        'model': Lasso(),
        'params': {}
      },
      'ElasticNet': {
        'model': ElasticNet(),
        'params': {}
      },
      'RandomForest': {
        'model': RandomForestRegressor(),
        'params': {}
      },
      'GradientBoosting': {
        'model': GradientBoostingRegressor(),
        'params': {}
      },
      'XGBRegressor': {
        'model': XGBRegressor(),
        'params': {}
      },
      'LGBMRegressor': {
        'model': LGBMRegressor(),
        'params': {}
      },
      'SVR': {
        'model': SVR(),
        'params': {}
      }
    }
    self._load_best_params()


  def _load_best_params(self):
    for model_name in self.models.keys():
      best_params_path = os.path.join(script_dir, f'model/{model_name}_best_params.pkl')
      if os.path.exists(best_params_path):
        self.models[model_name]['params'] = joblib.load(best_params_path)
        print(f'Best params for {model_name}:')
        print(self.models[model_name]['params'])
    
  def train(self):
    x_train_path = os.path.join(script_dir, 'data/x_train.npy')
    x_train = np.load(x_train_path, allow_pickle=True)
    
    y_train_path = os.path.join(script_dir, 'data/y_train.npy')
    y_train = np.load(y_train_path, allow_pickle=True)
    
    x_test_path = os.path.join(script_dir, 'data/x_test.npy')
    x_test = np.load(x_test_path, allow_pickle=True)
    
    y_test_path = os.path.join(script_dir, 'data/y_test.npy')
    y_test = np.load(y_test_path, allow_pickle=True)
        
    for model_name, model_data in self.models.items():
      model = model_data['model']
      model.set_params(**model_data['params'])
      model.fit(x_train, y_train)
      
      # Sauvegarde du modèle
      model_path = os.path.join(script_dir, f'model/{model_name}.pkl')
      joblib.dump(model, model_path)
      
      # Score du modèle
      score = model.score(x_test, y_test)
      print(f'{model_name} score: {score}')
      
      # Save score
      score_path = os.path.join(script_dir, f'model/{model_name}_score.txt')
      with open(score_path, 'w') as f:
        f.write(str(score))
        print(f'Score {score_path}')
        
if __name__ == '__main__':
  trainer = ModelTrainer()
  # trainer.train()