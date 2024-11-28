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
Train MSE: 44697066.291212834
Test MSE: 40482021.70661365
Train R²: 0.6915262521457672
Test R²: 0.7115094082979071
```

### Ridge
```bash
Best Parameters: {'alpha': 1.0, 'solver': 'svd'}
Train MSE: 44697161.96150772
Test MSE: 40482806.445573546
Train R²: 0.6915255918837664
Test R²: 0.7115038159436391
```

### Lasso
```bash
Best Parameters: {'alpha': 0.1, 'selection': 'cyclic'}
Train MSE: 44697066.51394036
Test MSE: 40482104.73198828
Train R²: 0.6915262506086284
Test R²: 0.7115088166268777
```

### ElasticNet
```bash
Best Parameters: {'alpha': 0.1, 'l1_ratio': 0.9}
Train MSE: 44724125.14439843
Test MSE: 40517802.30412927
Train R²: 0.6913395073200461
Test R²: 0.71125442187891
```

### RandomForest
```bash
Best Parameters: {'max_depth': 30, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 500}
Train MSE: 3469984.8672749237
Test MSE: 24181100.98840704
Train R²: 0.9760521366205146
Test R²: 0.8276760932862719
```

### GradientBoosting
```bash
Best Parameters: {'gamma': 0, 'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 2, 'n_estimators': 200}
Train MSE: 4813839.831129563
Test MSE: 19324862.487038948
Train R²: 0.9667776134432685
Test R²: 0.8622835328271998
```

### XGBRegressor
```bash
Best Parameters: {'gamma': 0, 'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 2, 'n_estimators': 200}
Train MSE: 4813839.831129563
Test MSE: 19324862.487038948
Train R²: 0.9667776134432685
Test R²: 0.8622835328271998
```

### LGBMRegressor
```bash
Best Parameters: {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 5, 'min_child_samples': 20, 'n_estimators': 200, 'num_leaves': 31, 'reg_alpha': 0.01, 'reg_lambda': 0.1, 'subsample': 0.8}
Train MSE: 8704684.945839759
Test MSE: 18497029.069285095
Train R²: 0.9399252118329431
Test R²: 0.8681829949205062
```

### SVR
```bash
Best Parameters: {'kernel': 'rbf', 'C': 97.81338812475256, 'epsilon': 0.0015374785628296513, 'gamma': 0.004180408469462122}
Train MSE: 149401365.99865252
Test MSE: 144951133.0329316
Train R²: -0.03108331548834964
Test R²: -0.03297800785792915
```
