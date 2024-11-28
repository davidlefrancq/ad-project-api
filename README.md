# ad_project_api

## Requirement
PDM: https://pdm-project.org/en/latest/#installation

## Start client
```bash
pdm run start
```

## Generate/Update requirements.txt
```bash
pdm export -o requirements.txt
```

# Models
## Hyperparams
### LinearRegression
```bash
Best Parameters: {}
Train MSE: 0.31038968179911086
Test MSE: 0.28111916236817835
Train R▒: 0.6915262521457641
Test R▒: 0.7115094082979001
```

### Ridge
```bash
Best Parameters: {'alpha': 10.0, 'solver': 'svd'}
Train MSE: 0.3104536940898909
Test MSE: 0.2814147807242112
Train R▒: 0.6914626349819101
Test R▒: 0.7112060383186661
```

### Lasso
```bash
Best Parameters: {'alpha': 0.1, 'selection': 'random'}
Train MSE: 0.37595871560972477
Test MSE: 0.349425325438467
Train R▒: 0.6263619545263865
Test R▒: 0.6414121398120209
```

### ElasticNet
```bash
Best Parameters: {'alpha': 0.1, 'l1_ratio': 0.1}
Train MSE: 0.32687010497178215
Test MSE: 0.3011319588334979
Train R▒: 0.6751475572328708
Test R▒: 0.6909718417896014
```

### RandomForest
```bash
Best Parameters: {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 500}
Train MSE: 0.023888373635184815
Test MSE: 0.16861953806988939
Train R▒: 0.9762590814788841
Test R▒: 0.8269589667935651
```

### GradientBoosting
```bash
Best Parameters: {'learning_rate': 0.1, 'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}
Train MSE: 0.02631742636502035
Test MSE: 0.13263461168830026
Train R▒: 0.9738450224967531
Test R▒: 0.8638874800145345
```

### XGBRegressor
```bash
Best Parameters: {'gamma': 0, 'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 2, 'n_estimators': 200}
Train MSE: 0.03342873651178149
Test MSE: 0.13419756197397872
Train R▒: 0.9667776081406759
Test R▒: 0.862283546476465
```

### LGBMRegressor
```bash
Best Parameters: {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 7, 'min_child_samples': 20, 'min_split_gain': 0.01, 'n_estimators': 200, 'num_leaves': 20, 'reg_alpha': 0.1, 'reg_lambda': 0.01, 'subsample': 0.8}
Train MSE: 0.05144396061334738
Test MSE: 0.12741872545566452
Train R▒: 0.9488735861228283
Test R▒: 0.8692401357809643
```

### SVR
```bash
Best Parameters: {'kernel': 'rbf', 'C': 31.72348496404067, 'epsilon': 0.08945058266848738, 'gamma': 0.004921155164830445}
Train MSE: 0.07529069422892809
Test MSE: 0.1661661950381002
Train R▒: 0.9251740505911004
Test R▒: 0.8294766407113683
```

## Best Model
### LGBMRegressor
```json
{
  'colsample_bytree': 0.8,
  'learning_rate': 0.1,
  'max_depth': 7,
  'min_child_samples': 20,
  'min_split_gain': 0.01,
  'n_estimators': 200,
  'num_leaves': 20,
  'reg_alpha': 0.1,
  'reg_lambda': 0.01,
  'subsample': 0.8
}
```