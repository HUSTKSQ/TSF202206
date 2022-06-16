'''
Descripttion: 
version: 
Author: Shenqiang Ke
Date: 2022-06-16 23:16:19
LastEditors: Please set LastEditors
LastEditTime: 2022-06-16 23:19:51
'''
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV

def lgb_reg():
    other_params = {'boosting_type' : 'gbdt', \
        'objective': 'mae', 
        # 'n_estimators': 275, 
        'max_depth': 6, 
        'min_child_samples': 20, 
        'metric': 'mae', 
        'colsample_bytree': 0.95, 
        'subsample': 0.8,
        'num_leaves' : 40,
        'random_state': 2022}

    rds_params = {'n_estimators': range(100, 400, 10),\
        'min_child_weight': range(3, 20, 2),
        'colsample_bytree': np.arange(0.4, 1.0),
        'max_depth': range(5, 15, 2),
        'subsample': np.arange(0.5, 1.0, 0.1),
        'reg_lambda': np.arange(0.1, 1.0, 0.2),
        'reg_alpha': np.arange(0.1, 1.0, 0.2),
        'min_child_samples': range(10, 30)
        }

    model = lgb.LGBMRegressor(**other_params)


    optimized_GBM = RandomizedSearchCV(model , rds_params, n_iter=50, cv=5, n_jobs=-1)

    return optimized_GBM