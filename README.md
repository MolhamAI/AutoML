# AutoML
Auto ML repo


# Reproducible AutoML Benchmarking Framework for Tabular Datasets

## A research-oriented AutoML framework for fair comparison of modern AutoML systems and boosting models under leakage-free experimental settings

This project implements a reproducible experimental framework for benchmarking modern AutoML systems and state-of-the-art boosting models on multiple tabular datasets. 
The goal is to provide fair, leakage-free, and statistically sound comparisons suitable for academic research and empirical evaluation.

The project includes implementations of multiple AutoML frameworks (AutoGluon, LightAutoML, H2O AutoML, FLAML) and boosting-based models (LightGBM, XGBoost, CatBoost), all evaluated under identical data splits, random seeds, and cross-validation protocols. 
Experiments are conducted using stratified 5-fold cross-validation with repeated runs, while ensuring that all preprocessing and feature engineering steps are fitted exclusively on training folds to prevent data leakage.

The framework automatically logs performance metrics (Accuracy, Precision, Recall, F1-score), runtime statistics, model sizes, and statistical significance tests, and generates publication-quality tables and figures to support future research and paper writing.
