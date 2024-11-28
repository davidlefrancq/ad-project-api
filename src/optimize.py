import os
import datetime
import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score

script_dir = os.path.dirname(__file__)

class HyperparameterOptimizer:
  def __init__(self):
    self.results = []
    self.models = {
      'LinearRegression': {
        # Avantages : Simple, rapide, interprétable
        # Inconvénients : Ne capture pas les relations non-linéaires
        'model': LinearRegression(),
        'params': {}  # Pas d'hyperparamètres à optimiser
      },
      'Ridge': {
        # Avantages : Gère la multicolinéarité, évite le surapprentissage
        # Inconvénients : Reste linéaire
        'model': Ridge(),
        'params': {
          'alpha': [0.1, 1.0, 10.0],            # Valeurs de régularisation L2 à tester
          'solver': ['svd', 'cholesky']         # Différents algorithmes de résolution
        }
      },
      'Lasso': {
        # Avantages : Sélection de caractéristiques automatique
        # Inconvénients : Peut être instable avec des caractéristiques corrélées
        'model': Lasso(),
        'params': {
          'alpha': [0.1, 1.0, 10.0],            # Force de la régularisation L1
          'selection': ['cyclic', 'random']     # Méthode de mise à jour des coefficients
        }
      },
      'ElasticNet': {
        # Avantages : Combine les avantages de Ridge et Lasso
        # Inconvénients : Deux hyperparamètres à régler
        'model': ElasticNet(),
        'params': {
          'alpha': [0.1, 1.0, 10.0],            # Force globale de la régularisation
          'l1_ratio': [0.1, 0.5, 0.9]           # Ratio de régularisation L1
        }
      },
      'RandomForest': {
        # Avantages : Capture les relations non-linéaires, peu de paramètres à régler
        # Inconvénients : Moins interprétable, peut être lent sur de grandes données
        'model': RandomForestRegressor(),
        'params': {
          'n_estimators': [100, 200, 500],      # Nombre d'arbres
          'max_depth': [10, 20, 30, None],      # Profondeur maximale des arbres
          'min_samples_split': [2, 5],          # Échantillons min pour diviser un noeud
          'min_samples_leaf': [1, 2]            # Échantillons min dans une feuille
        }
      },
      'GradientBoosting': {
        # Avantages : Très performant, gère bien les outliers
        # Inconvénients : Sensible aux hyperparamètres
        'model': GradientBoostingRegressor(),
        'params': {
          'n_estimators': [100, 200],           # Nombre d'arbres séquentiels
          'learning_rate': [0.1, 0.5],          # Taux d'apprentissage: 0.1 : lent/stable ; 0.5 : rapide/risque de surapprentissage
          'max_depth': [3, 5],                  # Profondeur max des arbres
          'min_samples_split': [2, 5],          # Échantillons min pour diviser un noeud
          'min_samples_leaf': [1, 2]            # Échantillons min dans feuille
        }
      },
      'XGBRegressor': {
        # Avantages : Souvent le plus performant, optimisé
        # Inconvénients : Nombreux hyperparamètres à régler
        'model': XGBRegressor(),
        'params': {
          'n_estimators': [100, 200],           # Nombre d'arbres
          'learning_rate': [0.1, 0.5],          # Taux d'apprentissage
          'max_depth': [3, 5],                  # Profondeur max des arbres
          'min_child_weight': [1, 2],           # Poids minimum des feuilles
          'gamma': [0, 0.1, 0.2]                # Seuil de division des noeuds
        }
      },
      'LGBMRegressor': {
        # Avantages : Très rapide, performant sur de grandes données
        # Inconvénients : Peut être instable sur petits datasets
        'model': LGBMRegressor(),
        'params': {
          'n_estimators': [100, 200],           # Nombre d'arbres
          'max_depth': [3, 5, 7],               # Profondeur max des arbres (-1 = illimité)
          'learning_rate': [0.1, 0.01],         # Taux d'apprentissage
          'num_leaves': [10, 20, 31],           # Nombre de feuilles max
          
          # Paramètres de régularisation
          'min_child_samples': [5, 10, 20],        # Nombre min d'échantillons par feuille
          'reg_alpha': [0.1, 0.01],        # L1 regularization
          'reg_lambda': [0.1, 0.01],       # L2 regularization
          
          # Paramètres d'échantillonnage
          'subsample': [0.8],              # Ratio d'échantillons par arbre
          'colsample_bytree': [0.8],       # Ratio de features par arbre
          'min_split_gain': [0.01, 0.1, 0.2]    # Gain min pour diviser un noeud
        }
      }
    }
    self.best_model = None
    self.best_params = {}
    self.x_train = None
    self.y_train = None
    self.x_test = None
    self.y_test = None
    self._load_data()

  def _load_data(self):
    """
    Charge les données d'entraînement et de test
    """
    # X d'entrainement
    x_train_path = os.path.join(script_dir, 'data/x_train.npy')
    self.x_train = np.load(x_train_path, allow_pickle=True)
    
    # Y d'entrainement
    y_train_path = os.path.join(script_dir, 'data/y_train.npy')
    self.y_train = np.load(y_train_path, allow_pickle=True)
    
    # X de test
    x_test_path = os.path.join(script_dir, 'data/x_test.npy')
    self.x_test = np.load(x_test_path, allow_pickle=True)
    
    # Y de test
    y_test_path = os.path.join(script_dir, 'data/y_test.npy')
    self.y_test = np.load(y_test_path, allow_pickle=True)

  def _optimize_svr_parameters(self, trial, x_train, y_train):
    """
    Fonction d'optimisation des hyperparamètres du SVR
    
    Parameters:
    trial: Instance d'essai Optuna
    X_train: Features d'entraînement
    y_train: Target d'entraînement
    
    Returns:
    float: Score moyen de validation croisée (MSE négatif)
    """
    # Définition de l'espace de recherche des hyperparamètres
    params = {
        'kernel': trial.suggest_categorical('kernel', ['rbf']),  # Type de noyau
        'C': trial.suggest_loguniform('C', 1e-2, 1e2),          # Paramètre de régularisation
        'epsilon': trial.suggest_loguniform('epsilon', 1e-3, 1), # Marge de tolérance
        'gamma': trial.suggest_loguniform('gamma', 1e-3, 1)     # Coefficient du noyau RBF
    }
    
    # Création du modèle avec les paramètres suggérés
    model = SVR(**params)
    
    # Évaluation du modèle par validation croisée
    scores = cross_val_score(
        model, x_train, y_train, 
        cv=5,                           # 5-fold cross validation
        scoring='neg_mean_squared_error',# Métrique d'évaluation
        n_jobs=-1                       # Utilisation de tous les coeurs CPU
    )
    
    return -scores.mean()  # Retourne la MSE moyenne (négative car sklearn maximise)


  def _optimize_svr(self):
    """
    Recherche les hyperparamètres du SVR avec Optuna
    """
    # Avantages : Bon pour les relations non-linéaires complexes
    # Inconvénients : Lent sur de grands datasets, sensible à l'échelle des features
    
    # Initialisation de l'étude Optuna
    study = optuna.create_study(direction='minimize')  # On cherche à minimiser l'erreur

    # Lancement de l'optimisation
    study.optimize(
      lambda trial: self._optimize_svr_parameters(trial, self.x_train, self.y_train), 
      n_trials=50,  # Nombre d'essais d'optimisation
      show_progress_bar=True
    )

    # Récupération des meilleurs hyperparamètres trouvés
    best_params = study.best_params

    # Entraînement du modèle final avec les meilleurs paramètres
    final_model = SVR(**best_params)
    final_model.fit(self.x_train, self.y_train)

    # Prédictions sur les ensembles d'entraînement et de test
    train_pred = final_model.predict(self.x_train)
    test_pred = final_model.predict(self.x_test)

    # Calcul des métriques de performance
    train_mse = mean_squared_error(self.y_train, train_pred)  # Erreur quadratique moyenne sur train
    test_mse = mean_squared_error(self.y_test, test_pred)     # Erreur quadratique moyenne sur test
    train_r2 = r2_score(self.y_train, train_pred)             # Coefficient de détermination sur train
    test_r2 = r2_score(self.y_test, test_pred)                # Coefficient de détermination sur test
        
    # Sauvegarde des résultats
    result = {
      'model': 'SVR',
      'best_params': best_params,
      'train_mse': train_mse,
      'test_mse': test_mse,
      'train_r2': train_r2,
      'test_r2': test_r2
    }
    self.results.append(result)

  def optimize(self):
    """
    Optimise les hyperparamètres de chaque modèle
    """
    datestart = datetime.datetime.now()
    for name, model_info in self.models.items():
      print(f"\nOptimizing {name}...")
      
      # Créer et entraîner le GridSearchCV
      grid_search = GridSearchCV(
        estimator=model_info['model'],
        param_grid=model_info['params'],
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,  # Utiliser tous les coeurs disponibles
        verbose=0   # Pas de logs 
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
      self.results.append(result)
      
      # Add best hyperparameters to best_params
      self.best_params[name] = grid_search.best_params_
      
      print(f"Best parameters for {name}: {grid_search.best_params_}")
      print(f"Test R² score: {result['test_r2']:.4f}")
    
    # SRV Model
    self._optimize_svr()
    
    dateend = datetime.datetime.now()
    print('\nTime elapsed: ', dateend - datestart)

    # Best model
    best_model_name = max(self.results, key=lambda x: x['test_r2'])['model'] # Meilleur modèle selon le R² de test
    best_params = self.best_params[best_model_name]    
    
    # Résultats de chaque modèle
    for result in self.results:
      print("\nModel:", result['model'])
      print("Best Parameters:", result['best_params'])
      print("Train MSE:", result['train_mse'])
      print("Test MSE:", result['test_mse'])
      print("Train R²:", result['train_r2'])
      print("Test R²:", result['test_r2'])
      
      # Sauvegarde des best params
      result_path = os.path.join(script_dir, f"model/{result['model']}_best_params.pkl")
      joblib.dump(result['best_params'], result_path)
      
    # Meilleur modèle    
    print(f"\nBest model: {best_model_name}")
    print(f"Best parameters: {best_params}")
    
    # Sauvegarde des résultats
    results_df = pd.DataFrame(self.results)
    results_df.to_csv(os.path.join(script_dir, 'model/results.csv'), index=False)

    
if __name__ == '__main__':
  optimizer = HyperparameterOptimizer()
  optimizer.optimize()