from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier, plot_importance
import matplotlib.pyplot as plt
import joblib
import seaborn as sns


import lightgbm as lgb

from xgboost import plot_importance as xgb_plot_importance
from catboost import Pool



from utils import *



param_grid = {
    'n_estimators': [100, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 6],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1],
    'min_child_weight': [1, 3],
            }



def evaluate_and_save_results_boosting(df_train, df_test, model, dataset_file):
    target_col = TARGET_COLS[dataset_file]
    
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    preds = model.predict(X_test)

    results = {
        'accuracy': (accuracy_score(y_test, preds)),
        'f1': (f1_score(y_test, preds, average='weighted')),
        'precision': (precision_score(y_test, preds, average='weighted')),
        'recall': (recall_score(y_test, preds, average='weighted'))
    }

    return results





