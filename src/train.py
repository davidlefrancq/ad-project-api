import os
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

script_dir = os.path.dirname(__file__)

class ModelTrainer:
  def __init__(self):
    self.best_params = {}
    self._load_best_params()
    self.models = {
      'linear': {
        'model': LinearRegression(),
        'params': {}
      },
      # 'ridge': {
      #   'model': Ridge(),
      #   'params': {
      #     'alpha': [0.1, 1.0, 10.0],
      #     'solver': ['auto', 'svd', 'cholesky']
      #   }
      # },
      # 'lasso': {
      #   'model': Lasso(),
      #   'params': {
      #     'alpha': [0.1, 1.0, 10.0],
      #     'selection': ['cyclic', 'random']
      #   }
      # },
      # 'rf': {
      #   'model': RandomForestRegressor(),
      #   'params': {
      #     'n_estimators': [100, 200],
      #     'max_depth': [10, 20, 30, None],
      #     'min_samples_split': [2, 5],
      #     'min_samples_leaf': [1, 2]
      #   }
      # },
      # 'svr': {
      #   'model': SVR(),
      #   'params': {
      #     'kernel': ['linear', 'rbf'],
      #     'C': [0.1, 1, 10],
      #     'epsilon': [0.1, 0.2, 0.5]
      #   }
      # }
    }

  def _load_best_params(self):
    best_params_path = os.path.join(script_dir, 'model/best_params.pkl')
    self.best_params = joblib.load(best_params_path)
    
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
      if self.best_params[model_name]:
        model_params = self.best_params[model_name]
        model.set_params(**model_params)
      model.fit(x_train, y_train)
      
      model_path = os.path.join(script_dir, f'model/{model_name}.pkl')
      joblib.dump(model, model_path)
      
      score = model.score(x_test, y_test)
      print(f'{model_name} score: {score}')
            
        
if __name__ == '__main__':
  trainer = ModelTrainer()
  trainer.train()