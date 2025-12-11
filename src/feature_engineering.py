from pipeline import *

ABLATION_TIME_BUDGET = 30


ablation_dict = {
    "no_feature_engineering": {
        "preprocessing": True,
        "nan_guard_before_vif": False,  
        "vif": False,
        "binner": False,
        "nan_guard_before_poly": False, 
        "poly": False,
        "selector": True, 
    },

    "polynomial_only": {
        "preprocessing": True,
        "nan_guard_before_vif": False,
        "vif": False,
        "binner": False,
        "nan_guard_before_poly": True,  
        "poly": True,
        "selector": True,
        "poly_degree": POLY_DEGREE_PIPELINE,
        "max_features_before_poly": MAX_FEATURES_BEFORE_POLY
    },

    "binning_only": {
        "preprocessing": True,
        "nan_guard_before_vif": False,
        "vif": False,
        "binner": True,
        "nan_guard_before_poly": False,
        "poly": False,
        "selector": True,
        "n_bins": N_BINS,
        "binning_strategy": BINNING_PIPELINE_STRATEGY
    },

    "polynomial_and_binning": {
        "preprocessing": True,
        "nan_guard_before_vif": False,
        "vif": False,
        "binner": True,
        "nan_guard_before_poly": True,
        "poly": True,
        "selector": True,
        "poly_degree": POLY_DEGREE_PIPELINE,
        "max_features_before_poly": MAX_FEATURES_BEFORE_POLY,
        "n_bins": N_BINS,
        "binning_strategy": BINNING_PIPELINE_STRATEGY
    },

    "vif_and_binning": {
        "preprocessing": True,
        "nan_guard_before_vif": True, 
        "vif": True,
        "binner": True,
        "nan_guard_before_poly": False,
        "poly": False,
        "selector": True,
        "n_bins": N_BINS,
        "binning_strategy": BINNING_PIPELINE_STRATEGY,
        "VIF_threshold": VIF_THRESHOLD_PIPELINE,
        "VIF_max_features": VIF_MAX_FEATURES,
        "VIF_sample_size": VIF_SAMPLE_SIZE,
        "VIF_max_iterations": VIF_MAX_ITERATIONS
    }
}
