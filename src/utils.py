import os, sys
import pandas as pd
import numpy as np
import tempfile
import shutil
import re

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.model_selection import StratifiedKFold, KFold

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
PROJECT_FILE_PATH =  os.getcwd()
THIS_FILE_LOCATION = os.path.join(PROJECT_FILE_PATH, "src")
THIS_FILE_PATH = os.path.join(PROJECT_FILE_PATH, "src")

from settings import *






def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss 
    return round(mem_bytes / (1024 * 1024), 3)
    




def evaluate_and_save_results_general(df_train, df_test, automl, dataset_name, framework='generic'):

    target_col = TARGET_COLS[dataset_name]
    task_type = tasks_dict_classification_only[dataset_name]

    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col].values

    if framework == 'h2o':
        import h2o
        from h2o.automl import H2OAutoML
        
        h2o_test = h2o.H2OFrame(X_test)
        preds_ds = automl.predict(h2o_test)
        preds_array = preds_ds.as_data_frame().values

        if task_type == 'binary':
            preds = (preds_array[:, 0] > 0.5).astype(int)
        elif task_type == 'multiclass':
            preds = preds_array.argmax(axis=1)
        else:
            preds = preds_array.flatten()

    else:
        preds_ds = automl.predict(X_test)
        if hasattr(preds_ds, 'data'):
            preds_array = np.array(preds_ds.data)
        else:
            preds_array = np.array(preds_ds)

        if task_type == 'binary':
            if preds_array.ndim > 1:
                preds = (preds_array[:, 0] > 0.5).astype(int)
            else:
                preds = (preds_array > 0.5).astype(int)
        elif task_type == 'multiclass':
            if preds_array.ndim > 1:
                preds = preds_array.argmax(axis=1)
            else:
                preds = preds_array
        else:
            preds = preds_array.flatten()

    results = {
        'accuracy': (accuracy_score(y_test, preds)),
        'f1': (f1_score(y_test, preds, average='weighted')),
        'precision': (precision_score(y_test, preds, average='weighted')),
        'recall': (recall_score(y_test, preds, average='weighted'))
    }

    return results







def clean_column_names(df):
    df = df.copy()
    df.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in df.columns]
    return df





def import_from_local_file(path,file_type):
        """
            This function is used to import data from local file system.
            The function takes file path and the file type as arguments/inputs and returns a 
            dataframe (Formats: .csv,.tsv, .xlsx, .json) as output.
            Modules:json, json_normalize
        """
        if file_type == 'CSV':
            try:
                df = pd.read_csv(path)
                return clean_column_names(df)
            except :
                print("Oops!File doesn't exist")
        elif file_type == 'TSV':
            try:
                df = pd.read_table(path)
                return clean_column_names(df)
            except :
                print("Oops!File doesn't exist")
        elif file_type == 'XLSX':
            try:
                df = pd.read_excel(path)
                return clean_column_names(df)
            except :
                print("Oops!File doesn't exist")
        elif file_type == 'JSON':
            try:
                df = pd.read_json(path,orient = 'columns')
                return clean_column_names(df)
            except :
                print("Oops!File doesn't exist")
        else:
            pass 



            

def EDA(df):
    
    b = pd.DataFrame()
    
    b['Missing value'] = df.isnull().sum()
    b['Missing Percentage']= (b['Missing value']/df.shape[0])*100
    b['N unique value'] = df.nunique()
    b['dtype'] = df.dtypes
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    b['max'] = np.nan
    b['min'] = np.nan
    b['mean'] = np.nan
    b['std_dev'] = np.nan
    
    b.loc[numeric_cols, 'max'] = df[numeric_cols].max()
    b.loc[numeric_cols, 'min'] = df[numeric_cols].min()
    b.loc[numeric_cols, 'mean'] = df[numeric_cols].mean()
    b.loc[numeric_cols, 'std_dev'] = df[numeric_cols].std()
    
    b.loc[ (b.dtype == 'object'), 'max'] = np.nan
    b.loc[ (b.dtype == 'object'), 'min'] = np.nan
    b.loc[ (b.dtype == 'object'), 'mean'] = np.nan
    b.loc[ (b.dtype == 'object'), 'std_dev'] = np.nan
    
    b.loc[ (b.dtype == 'int64'), 'dtype'] = 'Numeric'
    b.loc[(b.dtype =='float64'),'dtype'] ='Numeric'

    b.loc[ (b.dtype == 'bool'), 'max'] = np.nan
    b.loc[ (b.dtype == 'bool'), 'min'] = np.nan
    b.loc[ (b.dtype == 'bool'), 'mean'] = np.nan
    b.loc[ (b.dtype == 'bool'), 'std_dev'] = np.nan
    
    b.loc[ (b.dtype == 'category'), 'max'] = np.nan
    b.loc[ (b.dtype == 'category'), 'min'] = np.nan
    b.loc[ (b.dtype == 'category'), 'mean'] = np.nan
    b.loc[ (b.dtype == 'category'), 'std_dev'] = np.nan
    
    b.loc[ (b.dtype == 'datetime64'), 'max'] = np.nan
    b.loc[ (b.dtype == 'datetime64'), 'min'] = np.nan
    b.loc[ (b.dtype == 'datetime64'), 'mean'] = np.nan
    b.loc[ (b.dtype == 'datetime64'), 'std_dev'] = np.nan
    
    b.loc[ (b.dtype == 'timedelta[ns]'), 'max'] = np.nan
    b.loc[ (b.dtype == 'timedelta[ns]'), 'min'] = np.nan
    b.loc[ (b.dtype == 'timedelta[ns]'), 'mean'] = np.nan
    b.loc[ (b.dtype == 'timedelta[ns]'), 'std_dev'] = np.nan

            #print(type(self.b))
    b.dtype = b.dtype.astype(str)
    return b,df






def loop_data_files(DATA_FILE_PATH, n_files=None, names=None):
    files = sorted(os.listdir(DATA_FILE_PATH))
    data_dict = {}
    eda_dict = {}

    if names:
        for name in names:
            full_path = os.path.join(DATA_FILE_PATH, name)
            if os.path.exists(full_path):
                df = import_from_local_file(full_path, name.split('.')[-1].upper())
                if df is not None:
                    data_dict[name] = df
                    eda_dict[name] = EDA(df)[0]
            else:
                print(f"file '{name}' doesn't exist")
        if n_files and len(names) < n_files:
            remaining_files = [file for file in files if file not in data_dict]
            for file in remaining_files[: n_files - len(names)]:
                full_path = os.path.join(DATA_FILE_PATH, file)
                if os.path.exists(full_path):
                    df = import_from_local_file(full_path, file.split('.')[-1].upper())
                    if df is not None:
                        data_dict[file] = df
                        eda_dict[file] = EDA(df)[0]
                else:
                    print(f"file '{file}' doesn't exist")
    else:
        selected_files = files if n_files is None else files[:n_files]
        for file in selected_files:
            full_path = os.path.join(DATA_FILE_PATH, file)
            if os.path.exists(full_path):
                df = import_from_local_file(full_path, file.split('.')[-1].upper())
                if df is not None:
                    data_dict[file] = df
                    eda_dict[file] = EDA(df)[0]
            else:
                print(f"file '{file}' doesn't exist")

    return data_dict, eda_dict



try:
    if data_dict:
        pass
except:
    data_dict, eda_dict = loop_data_files(DATA_PATH)



def preprocess_features_dict(data_dict):
    processed_dict = {}
    for name, df in data_dict.items():
        df_copy = df.copy()
        for col in df_copy.columns:
            if df_copy[col].dtype == 'bool':
                df_copy[col] = df_copy[col].astype(int)
            elif df_copy[col].dtype.name in ['object', 'category']:
                df_copy[col] = df_copy[col].astype('category')
            elif np.issubdtype(df_copy[col].dtype, np.datetime64):
                df_copy[col+'_year'] = df_copy[col].dt.year
                df_copy[col+'_month'] = df_copy[col].dt.month
                df_copy[col+'_day'] = df_copy[col].dt.day
                df_copy[col+'_weekday'] = df_copy[col].dt.weekday
                df_copy = df_copy.drop(columns=[col])
        for col in df_copy.select_dtypes(include=['object']).columns:
            df_copy[col] = df_copy[col].astype('category')
        processed_dict[name] = df_copy
    return processed_dict



try:
    if data_dict_processed:
        pass
except:
    data_dict_processed = preprocess_features_dict(data_dict)




def build_categories_dict(data_dict_processed):
    categories_dict = {}

    for name, df in data_dict_processed.items():
        col_categories = {}
        for col in df.select_dtypes(include=['category']).columns:
            col_categories[col] = df[col].cat.categories
        categories_dict[name] = col_categories

    return categories_dict


try:
    if categories_dict:
        pass
except:
    categories_dict = build_categories_dict(data_dict_processed)





def detect_task_type(df, target_col, numeric_reg_threshold=20, unique_ratio_threshold=0.1):
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame")
    s = df[target_col].dropna()
    n_unique = s.nunique()
    total = len(s)
    if n_unique == 0:
        return 'unknown'
    unique_ratio = n_unique / total
    if n_unique <= 2:
        return 'binary'
    if pd.api.types.is_numeric_dtype(s):
        if n_unique >= numeric_reg_threshold or unique_ratio > unique_ratio_threshold:
            return 'regression'
        else:
            return 'multiclass'
    else:
        if n_unique == 2:
            return 'binary'
        else:
            return 'multiclass'


try:
    if tasks_dict:
        pass
except:
    tasks_dict = {}
    for dataset_name, target_col in TARGET_COLS.items():
        if dataset_name in data_dict_processed:
            df = data_dict_processed[dataset_name]
            task_type = detect_task_type(df, target_col)
            tasks_dict[dataset_name] = task_type










def keep_classification_only(data_dict_processed, tasks_dict):
    data_dict_classification_only = {
        name: df for name, df in data_dict_processed.items() if tasks_dict.get(name) in ['binary', 'multiclass']}
    tasks_dict_classification_only = {
        name: task for name, task in tasks_dict.items() if task in ['binary', 'multiclass']}
    
    return data_dict_classification_only, tasks_dict_classification_only


try:
    if data_dict_classification_only:
        pass
except:
    data_dict_classification_only, tasks_dict_classification_only = keep_classification_only(data_dict_processed, tasks_dict)










def label_encode_targets(data_dict_processed, TARGET_COLS, tasks_dict):
    encoders_dict = {}
    for dataset_name, df in data_dict_classification_only.items():
        target_col = TARGET_COLS[dataset_name]
        task_type = tasks_dict[dataset_name]

        if task_type in ['binary', 'multiclass']:
            encoder = LabelEncoder()
            df[target_col] = encoder.fit_transform(df[target_col].astype(str))
            data_dict_classification_only[dataset_name] = df
            encoders_dict[dataset_name] = encoder  
    return data_dict_classification_only, encoders_dict


            
try:
    if encoders_dict:
        pass
except:
    data_dict_classification_only, encoders_dict = label_encode_targets(data_dict_classification_only, TARGET_COLS, tasks_dict_classification_only)













def split_datasets(data_dict_processed, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    splits_dict_ = {}
    for key, df in data_dict_processed.items():
        target_col = TARGET_COLS[key]
        df_clean = df[df[target_col].notna()].copy()
        train_df, test_df = train_test_split(df_clean, test_size=test_size, random_state=random_state)
        feature_cols = [c for c in df.columns if c != target_col]

        numeric_cols = train_df[feature_cols].select_dtypes(include=['number']).columns
        categorical_cols = train_df[feature_cols].select_dtypes(exclude=['number']).columns

        if len(numeric_cols) > 0:
            imputer_num = SimpleImputer(strategy='mean')
            train_df[numeric_cols] = imputer_num.fit_transform(train_df[numeric_cols])
            test_df[numeric_cols] = imputer_num.transform(test_df[numeric_cols])

        if len(categorical_cols) > 0:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            train_df[categorical_cols] = imputer_cat.fit_transform(train_df[categorical_cols])
            test_df[categorical_cols] = imputer_cat.transform(test_df[categorical_cols])
            train_df[categorical_cols] = train_df[categorical_cols].astype('category')
            test_df[categorical_cols] = test_df[categorical_cols].astype('category')

        splits_dict_[key] = {"train": train_df, "test": test_df}
    return splits_dict_


try:
    if splits_dict_:
        pass
except:
    splits_dict_ = split_datasets(data_dict_classification_only)





def align_categories_in_splits(splits_dict_):
    splits_dict = {}  
    for dataset_name, split in splits_dict_.items():
        df_train = split["train"]
        df_test = split["test"]

        all_levels = {}
        for col in df_train.select_dtypes(include=['category']).columns:
            train_levels = list(df_train[col].cat.categories)
            test_levels = list(df_test[col].cat.categories)
            combined = set(train_levels) | set(test_levels)
            all_levels[col] = combined

        for col, cats in all_levels.items():
            df_train[col] = df_train[col].cat.set_categories(sorted(list(cats)))
            df_test[col] = df_test[col].cat.set_categories(sorted(list(cats)))

        splits_dict[dataset_name] = {"train": df_train, "test": df_test}

    return splits_dict



try:
    if splits_dict:
        pass
except:
    splits_dict = align_categories_in_splits(splits_dict_)







def create_stratified_folds(data_dict_processed, n_splits=N_FOLDS, random_state=RANDOM_STATE):
    folds_dict_ = {}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for dataset_name, df in data_dict_processed.items():
        target_col = TARGET_COLS[dataset_name]
        df_clean = df[df[target_col].notna()].reset_index(drop=True)
        feature_cols = [c for c in df.columns if c != target_col]

        numeric_cols = df_clean[feature_cols].select_dtypes(include=['number']).columns
        categorical_cols = df_clean[feature_cols].select_dtypes(exclude=['number']).columns

        fold_data = {}
        for i, (train_idx, val_idx) in enumerate(skf.split(df_clean[feature_cols], df_clean[target_col]), start=1):
            train_df = df_clean.iloc[train_idx].copy()
            val_df = df_clean.iloc[val_idx].copy()

            if len(numeric_cols) > 0:
                imputer_num = SimpleImputer(strategy='mean')
                train_df[numeric_cols] = imputer_num.fit_transform(train_df[numeric_cols])
                val_df[numeric_cols] = imputer_num.transform(val_df[numeric_cols])

            if len(categorical_cols) > 0:
                imputer_cat = SimpleImputer(strategy='most_frequent')
                train_df[categorical_cols] = imputer_cat.fit_transform(train_df[categorical_cols])
                val_df[categorical_cols] = imputer_cat.transform(val_df[categorical_cols])
                train_df[categorical_cols] = train_df[categorical_cols].astype('category')
                val_df[categorical_cols] = val_df[categorical_cols].astype('category')

            fold_data[f"fold_{i}"] = {"train": train_df, "val": val_df}

        folds_dict_[dataset_name] = fold_data

    return folds_dict_


try:
    if folds_dict_:
        pass
except:
    folds_dict_ = create_stratified_folds(data_dict_classification_only)




def align_categories_in_folds(folds_dict):
    for dataset_name, folds in folds_dict.items():
        all_levels = {}

        for fold_name, split in folds.items():
            df_train = split["train"]
            df_val = split["val"]

            for col in df_train.select_dtypes(include=['category']).columns:
                train_levels = list(df_train[col].cat.categories)
                val_levels = list(df_val[col].cat.categories)
                combined = set(train_levels) | set(val_levels)
                if col not in all_levels:
                    all_levels[col] = combined
                else:
                    all_levels[col] |= combined

        for fold_name, split in folds.items():
            df_train = split["train"]
            df_val = split["val"]

            for col, cats in all_levels.items():
                df_train[col] = df_train[col].cat.set_categories(sorted(list(cats)))
                df_val[col] = df_val[col].cat.set_categories(sorted(list(cats)))

            folds_dict[dataset_name][fold_name]["train"] = df_train
            folds_dict[dataset_name][fold_name]["val"] = df_val

    return folds_dict



try:
    if folds_dict:
        pass
except:
    folds_dict = align_categories_in_folds(folds_dict_)















def print_dict_structure(d):
    nested_dicts = []

    print("{")
    for k, v in d.items():
        print(f"  {repr(k)}: {type(v).__name__}")
        if isinstance(v, dict):
            nested_dicts.append((k, v))
    print("}")

    for k, sub_d in nested_dicts:
        print(f"\nkey: {k}")
        print_dict_structure(sub_d)


def show_dfs_from_dict(d, index=None):
    keys = list(d.keys())

    if index is None or index >= len(keys):
        items = keys
    else:
        items = [keys[index]]

    for k in items:
        v = d[k]
        if isinstance(v, dict):
            for sub_k in v:
                print(f"{k} → {sub_k}")
                display(v[sub_k])
        else:
            print(k)
            display(v)






AUTOML = ["autogluon", "lightautoml", "h2o", "flaml"]
BOOSTINGS = ["lightgbm", "xgboost", "catboost"]
DATASET_NAMES = list(data_dict_classification_only.keys())

