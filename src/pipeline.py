import time
import os
import json
import random
import numpy as np
import pandas as pd


from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.feature_selection import VarianceThreshold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import KBinsDiscretizer

from category_encoders.target_encoder import TargetEncoder


from utils import *

TARGET_COLS_PIPELINE = {
    "modeldata.csv": "IsInsurable",
    "salary.csv": "Salary",
    "train.csv": "variety",
    "titanic.csv": "Survived",
    "wine.csv": "Wine"
}


RANDOM_SEEDS_PIPELINE = [42, 123]
# RANDOM_SEEDS_PIPELINE = [42]    # RANDOM_SEEDS_PIPELINE = [42, 123]
N_FOLDS_PIPELINE = 2    # N_FOLDS_PIPELINE = 5
CV_TIME_BUDGET_PIPELINE = 31

IMPUTE_STRATEGY_NUM_PIPELINE = "median"
# IMPUTE_STRATEGY_CAT_PIPELINE = "most_frequent"
# SCALER_TYPE_PIPELINE = "standard"
# ENCODER_TYPE_PIPELINE = "onehot"
HANDLE_UNKNOWN = "ignore"  
SPARSE_OUTPUT = False    
TEST_SIZE_PIPELINE = 0.2
RANDOM_STATE_PIPELINE = RANDOM_SEEDS_PIPELINE[0]
SCORING_METRICS_PIPELINE = ["accuracy", "f1", "precision", "recall"]

BINNING_PIPELINE_STRATEGY = "quantile"
N_BINS = 5

POLY_DEGREE_PIPELINE = 1
MAX_FEATURES_BEFORE_POLY = 50

VIF_MAX_FEATURES = 120
VIF_THRESHOLD_PIPELINE = 10
VIF_SAMPLE_SIZE = 3000
VIF_MAX_ITERATIONS = 20
MIN_FEATURES = 25

K_FEATURES_RATIO = 0.3


try:
    if K_FEATURES_PIPELINE_DICT:
        pass
except:
    K_FEATURES_PIPELINE_DICT = {}
    for data_name, data in data_dict_classification_only.items():
        n_features = data.shape[1]
        k = max(MIN_FEATURES, int(n_features * K_FEATURES_RATIO)) \
            if n_features > MIN_FEATURES else n_features
        K_FEATURES_PIPELINE_DICT[data_name] = k






class TimedTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformer, name=None, log_dict=None):
        self.transformer = transformer
        self.name = name if name else transformer.__class__.__name__
        self.log_dict = log_dict if log_dict is not None else {}

    def fit(self, X, y=None):
        start = time.time()
        self.transformer.fit(X, y)
        end = time.time()
        self.log_dict[f"{self.name}_fit_sec"] = end - start
        return self

    def transform(self, X):
        start = time.time()
        X_out = self.transformer.transform(X)
        end = time.time()
        self.log_dict[f"{self.name}_transform_sec"] = end - start
        return X_out

class StepTimer(BaseEstimator, TransformerMixin):
    def __init__(self, name, log_dict=None, transformer=None):
        self.transformer = transformer
        self.name = name
        self.log_dict = log_dict if log_dict is not None else {}

    def fit(self, X, y=None):
        start = time.time()
        if self.transformer is not None:
            self.transformer.fit(X, y)
        self.log_dict[self.name] = time.time() - start
        return self

    def transform(self, X):
        start = time.time()
        if self.transformer is not None:
            X_trans = self.transformer.transform(X)
        else:
            X_trans = X.copy()
        end = time.time()
        self.log_dict[self.name] = end - start
        return X_trans



class VIFSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=VIF_THRESHOLD_PIPELINE, max_features=VIF_MAX_FEATURES, sample_size=VIF_SAMPLE_SIZE, max_iterations=VIF_MAX_ITERATIONS, random_state=42):
        self.threshold = threshold
        self.max_features = max_features
        self.sample_size = sample_size
        self.max_iter = max_iterations
        self.random_state = random_state
        self.selected_features_ = None

    def approx_vif_from_corr(self, X):
        corr = X.corr().abs()
        vif_values = []
        for col in corr.columns:
            others = corr[col].drop(col)
            if len(others) > 0:
                r2 = others.max() ** 2
            else:
                r2 = 0
            if r2 < 1:
                vif = 1.0 / (1.0 - r2)
            else:
                vif = np.inf
            vif_values.append(vif)
        return pd.DataFrame({"variables": corr.columns, "VIF": vif_values})

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        
        if X_df.shape[0] > self.sample_size:
            X_sample = X_df.sample(self.sample_size, random_state=self.random_state)
            
        else:
            X_sample = X_df
            
        numeric_cols = X_sample.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if X_sample[c].nunique() > 1]
        
        if not numeric_cols or X_df.shape[1] <= 5:
            self.selected_features_ = X_df.columns.tolist()
            return self
            
        X_var = X_sample[numeric_cols].copy()
        
        if X_var.shape[1] > self.max_features:
            X_var = X_var[X_var.var().sort_values(ascending=False).head(self.max_features).index]
            
        features = list(X_var.columns)
        
        for i in range(self.max_iter):
            if len(features) <= 1:
                break
                
            vif_df = self.approx_vif_from_corr(X_var[features])
            
            if vif_df["VIF"].max() < self.threshold:
                break
                
            drop_feat = vif_df.loc[vif_df["VIF"].idxmax(), "variables"]
            features.remove(drop_feat)
            
        self.selected_features_ = features
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.selected_features_]





def split_data_to_folds(data_dict_classification_only, n_folds=N_FOLDS_PIPELINE):
    pipeline_data_dict_ = {}

    for dataset_file, df in data_dict_classification_only.items():
        y = df[TARGET_COLS_PIPELINE[dataset_file]]
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        pipeline_data_dict_[dataset_file] = {}

        for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(df, y)):
            train_df = df.iloc[train_idx].copy()
            valid_df = df.iloc[valid_idx].copy()

            fold_name = f"fold_{fold_idx + 1}"
            pipeline_data_dict_[dataset_file][fold_name] = {
                "train": train_df,
                "val": valid_df
            }

    return pipeline_data_dict_


try:
    if pipeline_data_dict_:
        pass
except:
    pipeline_data_dict_ = split_data_to_folds(data_dict_classification_only)


def align_categories_in_folds_pipeline(pipeline_data_dict_):
    pipeline_data_dict = {}

    for dataset_file, folds in pipeline_data_dict_.items():
        all_cats = {}

        for fold_name, fold_data in folds.items():
            df_train = fold_data["train"]
            for col in df_train.select_dtypes(include=["object", "category"]).columns:
                all_cats.setdefault(col, set()).update(df_train[col].dropna().unique())

        pipeline_data_dict[dataset_file] = {}

        for fold_name, fold_data in folds.items():
            train_df = fold_data["train"].copy()
            valid_df = fold_data["val"].copy()

            for col, cats in all_cats.items():
                if col in train_df.columns:
                    train_df[col] = train_df[col].astype("category").cat.set_categories(sorted(list(cats)))
                if col in valid_df.columns:
                    valid_df[col] = valid_df[col].astype("category").cat.set_categories(sorted(list(cats)))

            pipeline_data_dict[dataset_file][fold_name] = {
                "train": train_df,
                "val": valid_df
            }

    return pipeline_data_dict


try:
    if pipeline_data_dict:
        pass
except:
    pipeline_data_dict = align_categories_in_folds_pipeline(pipeline_data_dict_)







def pipeline_builder(dataset_name, X=None, time_log=None, enable_steps=None):
    if enable_steps is None:
        enable_steps = {}

    num_basic = Pipeline([
        ("imputer", SimpleImputer(strategy=enable_steps.get("impute_strategy_num_pipeline", "mean"))),
        ("scaler", StandardScaler())
    ])

    cat_encoder = OneHotEncoder(
        handle_unknown=enable_steps.get("handle_unknown", "ignore"),
        sparse_output=enable_steps.get("sparse_output", False)
    )

    preprocessor = ColumnTransformer([
        ("num_basic", num_basic, make_column_selector(dtype_include=["int","float"])),
        ("cat", cat_encoder, make_column_selector(dtype_include=["object","category"]))
    ], remainder="drop", verbose_feature_names_out=False)

    steps = []

    if enable_steps.get("preprocessing", True):
        steps.append(("preprocessing", StepTimer("preprocessing", time_log, transformer=preprocessor)))

    if enable_steps.get("nan_guard_before_vif", True):
        steps.append(("nan_guard_before_vif", StepTimer("nan_guard_before_vif", time_log,
                                                       transformer=SimpleImputer(strategy="constant", fill_value=0))))

    if enable_steps.get("vif", True) and X is not None:
        if X.shape[0] > 20:
            steps.append(("vif", StepTimer("vif", time_log, transformer=VIFSelector())))
        else:
            steps.append(("vif", StepTimer("vif", time_log, transformer=VIFSelector(threshold=1e6))))

    if enable_steps.get("binner", True):
        steps.append(("binner", StepTimer("binner", time_log,
                                          transformer=KBinsDiscretizer(
                                              n_bins=enable_steps.get("n_bins", N_BINS),
                                              encode="ordinal",
                                              strategy=enable_steps.get("binning_strategy", "uniform")
                                          ))))

    if enable_steps.get("nan_guard_before_poly", True):
        steps.append(("nan_guard_before_poly", StepTimer("nan_guard_before_poly", time_log,
                                                         transformer=SimpleImputer(strategy="constant", fill_value=0))))

    if enable_steps.get("poly", True) and X is not None and X.shape[0] > 20:
        if X.shape[1] <= MAX_FEATURES_BEFORE_POLY: 
            steps.append(("poly", StepTimer("poly", time_log,
                                            transformer=PolynomialFeatures(degree=enable_steps.get("poly_degree", 2),
                                                                          include_bias=False))))

    if enable_steps.get("selector", True) and X is not None:
        k_orig = enable_steps.get("k_features_dict", {}).get(dataset_name, X.shape[1])
        if X.shape[0] > 20:
            k_reduced = min(k_orig, X.shape[1], 200) 
        else:
            k_reduced = X.shape[1]
        steps.append(("selector", StepTimer("selector", time_log, transformer=SelectKBest(f_classif, k=k_reduced))))

    return Pipeline(steps)


def run_pipeline_on_folds_with_control(pipeline_data_dict, enable_steps=None):
    pipeline_controlled_dict = {}

    for dataset_name, folds in pipeline_data_dict.items():
        target = TARGET_COLS_PIPELINE[dataset_name]
        pipeline_controlled_dict[dataset_name] = {}

        for fold_name, fold_data in folds.items():
            X_train = fold_data["train"].drop(columns=[target])
            y_train = fold_data["train"][target]
            X_val = fold_data["val"].drop(columns=[target])
            y_val = fold_data["val"][target]

            pipeline_time_log = {}
            pipeline = pipeline_builder(dataset_name=dataset_name, X=X_train,
                                        time_log=pipeline_time_log, enable_steps=enable_steps)

            X_train_p = pipeline.fit_transform(X_train, y_train)
            X_val_p = pipeline.transform(X_val)

            feature_names = [f"f_{i}" for i in range(X_train_p.shape[1])]
            X_train_p = pd.DataFrame(X_train_p, columns=feature_names)
            X_val_p = pd.DataFrame(X_val_p, columns=feature_names)

            X_train_p[target] = y_train.values
            X_val_p[target] = y_val.values

            pipeline_controlled_dict[dataset_name][fold_name] = {
                "train": X_train_p,
                "val": X_val_p,
                "pipeline_time": pipeline_time_log
            }

    return pipeline_controlled_dict
