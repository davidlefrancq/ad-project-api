import os
import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

script_dir = os.path.dirname(__file__)

class HyperparameterOptimizer:
  def __init__(self):
    self.models = {
      'linear': {
        'model': LinearRegression(),
        'params': {}  # Pas d'hyperparamètres à optimiser
      },
      'ridge': {
        'model': Ridge(),
        'params': {
          'alpha': [0.1, 1.0, 10.0],
          'solver': ['auto', 'svd', 'cholesky']
        }
      },
      'lasso': {
        'model': Lasso(),
        'params': {
          'alpha': [0.1, 1.0, 10.0],
          'selection': ['cyclic', 'random']
        }
      },
      'rf': {
        'model': RandomForestRegressor(),
        'params': {
          'n_estimators': [100, 200],
          'max_depth': [10, 20, 30, None],
          'min_samples_split': [2, 5],
          'min_samples_leaf': [1, 2]
        }
      },
      'svr': {
        'model': SVR(),
        'params': {
          'kernel': ['linear', 'rbf'],
          'C': [0.1, 1, 10],
          'epsilon': [0.1, 0.2, 0.5]
        }
      }
    }
    self.best_models = {}
    self.best_params = None
    self.x_train = None
    self.y_train = None
    self.x_test = None
    self.y_test = None
    self._load_data()

  def _load_data(self):
    # Load x_train, y_train, x_test, y_test
    x_train_path = os.path.join(script_dir, 'data/x_train.npy')
    self.x_train = np.load(x_train_path, allow_pickle=True)
    
    y_train_path = os.path.join(script_dir, 'data/y_train.npy')
    self.y_train = np.load(y_train_path, allow_pickle=True)
    
    x_test_path = os.path.join(script_dir, 'data/x_test.npy')
    self.x_test = np.load(x_test_path, allow_pickle=True)
    
    y_test_path = os.path.join(script_dir, 'data/y_test.npy')
    self.y_test = np.load(y_test_path, allow_pickle=True)

  def _save_best_params(self):
    best_params_path = os.path.join(script_dir, 'model/best_params.pkl')
    joblib.dump(self.best_models, best_params_path)
    print(f"Best parameters saved to {best_params_path}")

  def optimize(self):
    datestart = datetime.datetime.now()
    results = []
    for name, model_info in self.models.items():
      print(f"\nOptimizing {name}...")
      
      # Créer et entraîner le GridSearchCV
      grid_search = GridSearchCV(
        estimator=model_info['model'],
        param_grid=model_info['params'],
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1  # Utiliser tous les cœurs disponibles
      )
      
      # Entraîner le modèle
      grid_search.fit(self.x_train, self.y_train)
            
      # Faire des prédictions
      train_pred = grid_search.predict(self.x_train)
      test_pred = grid_search.predict(self.x_test)
           
      result = {
        'model': name,
        'best_params': grid_search.best_params_,
        'train_mse': mean_squared_error(self.y_train, train_pred),
        'test_mse': mean_squared_error(self.y_test, test_pred),
        'train_r2': r2_score(self.y_train, train_pred),
        'test_r2': r2_score(self.y_test, test_pred)
      }
      results.append(result)
      
      print(f"Best parameters for {name}: {grid_search.best_params_}")
      print(f"Test R² score: {result['test_r2']:.4f}")
    
    # Créer un DataFrame avec tous les résultats
    results_df = pd.DataFrame(results)
    print("\nFinal Results:")
    print(results_df)
    
    # Sauvegarder les résultats
    results_path = os.path.join(script_dir, 'model/optimization_results.csv')
    results_df.to_csv(results_path, index=False)
  
    # Sauvegarder les meilleurs hyperparamètres
    self.best_params = results_df.loc[results_df['test_r2'].idxmax(), 'best_params']
    self._save_best_params()
    
    dateend = datetime.datetime.now()
    print('\nTime elapsed: ', dateend - datestart)
    
    return results_df

if __name__ == '__main__':
  optimizer = HyperparameterOptimizer()
  optimizer.optimize()