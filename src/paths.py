import os, sys
import inspect

def get_paths():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    root_path         = os.path.abspath(os.path.join(base_dir, ".."))
    current_run_path  = os.getcwd()
    current_file_path = base_dir

    paths = {
        "PROJECT_ROOT"    : root_path       ,
        "CURRENT_RUN_PATH": current_run_path,
        
        "DATA_PATH"       : os.path.join(root_path, "data"       ),
        "COMPARISONS_PATH": os.path.join(root_path, "comparisons"),
        "NOTEBOOKS_PATH"  : os.path.join(root_path, "notebooks"  ),
        "OUTPUTS_PATH"    : os.path.join(root_path, "outputs"    ),
        "SRC_PATH"        : os.path.join(root_path, "src"        ),
        "REPORTS_PATH"    : os.path.join(root_path, "reports"    ),

        "AUTOML_COMPARISON_PATH": os.path.join(root_path, "notebooks", "automl_comparison"),
        
        "METRICS_PATH"                    : os.path.join(root_path, "outputs", "metrics"                                                   ),
        "AUTOML_METRICS_PATH"             : os.path.join(root_path, "outputs", "metrics", "automl_metrics"                                 ),        
        "BOOSTINGS_PATH"                  : os.path.join(root_path, "outputs", "metrics", "boostings_metrics"                              ),
        "BOOSTINGS_TUNED_PATH"            : os.path.join(root_path, "outputs", "metrics", "boostings_metrics_tuned"                        ),
        "CV_FOLDS_METRICS_PATH"           : os.path.join(root_path, "outputs", "metrics", "cv_folds_metrics"                               ),
        "PIPELINE_METRICS_PATH"           : os.path.join(root_path, "outputs", "metrics", "pipeline_metrics"                               ),
        "PIPELINE_LOG_TIMES_PATH"         : os.path.join(root_path, "outputs", "metrics", "pipeline_metrics", "pipeline_log_times"         ),
        "ABLATION_METRICS_PATH"           : os.path.join(root_path, "outputs", "metrics", "pipeline_metrics", "ablation_metrics"           ),   
        "PIPELINE_DATAFRAMES_METRICS_PATH": os.path.join(root_path, "outputs", "metrics", "pipeline_metrics", "pipeline_dataframes_metrics"),

        "ABLATION_MODELS_FOR_SHAP_PATH": os.path.join(root_path, "outputs", "ablation_models_for_shap"),
        
        "PLOTS_PATH"             : os.path.join(root_path, "outputs", "plots"                      ),
        "ABLATION_PLOTS_PATH"    : os.path.join(root_path, "outputs", "plots", "ablation_polts"    ),
        "FEATURE_IMPORTANCE_PATH": os.path.join(root_path, "outputs", "plots", "feature_importance"),

        "FEATURE_IMPORTANCE_SHAP_PATH": os.path.join(root_path, "outputs", "figures", "shap", "feature_importance"),
        "PER_FEATURE_SHAP_PATH"       : os.path.join(root_path, "outputs", "figures", "shap", "per_feature"       ),
        "SAMPLES_SHAP_PATH"           : os.path.join(root_path, "outputs", "figures", "shap", "samples"           ),

        "RUNTIME_MEMORY_TASK1_PATH": os.path.join(root_path, "reports", "runtime_memory_task1"),
        "RUNTIME_MEMORY_TASK2_PATH": os.path.join(root_path, "reports", "runtime_memory_task2"),
        "RUNTIME_MEMORY_TASK3_PATH": os.path.join(root_path, "reports", "runtime_memory_task3"),

        "RUNTIME_MEMORY_TASK2_BOOSTINGS_PATH"      : os.path.join(root_path, "reports", "runtime_memory_task2", "boosting"      ),
        "RUNTIME_MEMORY_TASK2_BOOSTINGS_TUNED_PATH": os.path.join(root_path, "reports", "runtime_memory_task2", "boosting_tuned"),
        

        "TASK1_COMPARISONS_PATH": os.path.join(root_path, "comparisons", "task1_comparisons"),
        "TASK2_COMPARISONS_PATH": os.path.join(root_path, "comparisons", "task2_comparisons"),
        "TASK3_COMPARISONS_PATH": os.path.join(root_path, "comparisons", "task3_comparisons"),
        "TASK5_COMPARISONS_PATH": os.path.join(root_path, "comparisons", "task5_comparisons"),

    }

    caller_globals = inspect.currentframe().f_back.f_globals

    for key, value in paths.items():
        caller_globals[key] = value

    return tuple(paths.values())
