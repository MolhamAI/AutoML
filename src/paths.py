from pathlib import Path
import inspect
import os

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_paths(project_root=None):
    try:
        base_dir = Path(__file__).resolve().parent
    except NameError:
        base_dir = Path(os.getcwd())

    root_path = Path(project_root) if project_root else base_dir.parent

    paths_dict = {
        "PROJECT_ROOT"    : root_path,
        "CURRENT_RUN_PATH": Path(os.getcwd()),
        "DATA_PATH"       : ensure_dir(root_path / "data"),
        "COMPARISONS_PATH": ensure_dir(root_path / "comparisons"),
        "NOTEBOOKS_PATH"  : ensure_dir(root_path / "notebooks"),
        "OUTPUTS_PATH"    : ensure_dir(root_path / "outputs"),
        "SRC_PATH"        : ensure_dir(root_path / "src"),
        "REPORTS_PATH"    : ensure_dir(root_path / "reports"),
        "AUTOML_COMPARISON_PATH": ensure_dir(root_path / "notebooks" / "automl_comparison"),
        "METRICS_PATH"                    : ensure_dir(root_path / "outputs" / "metrics"),
        "AUTOML_METRICS_PATH"             : ensure_dir(root_path / "outputs" / "metrics" / "automl_metrics"),
        "BOOSTINGS_PATH"                  : ensure_dir(root_path / "outputs" / "metrics" / "boostings_metrics"),
        "BOOSTINGS_TUNED_PATH"            : ensure_dir(root_path / "outputs" / "metrics" / "boostings_metrics_tuned"),
        "CV_FOLDS_METRICS_PATH"           : ensure_dir(root_path / "outputs" / "metrics" / "cv_folds_metrics"),
        "PIPELINE_METRICS_PATH"           : ensure_dir(root_path / "outputs" / "metrics" / "pipeline_metrics"),
        "PIPELINE_LOG_TIMES_PATH"         : ensure_dir(root_path / "outputs" / "metrics" / "pipeline_metrics" / "pipeline_log_times"),
        "ABLATION_METRICS_PATH"           : ensure_dir(root_path / "outputs" / "metrics" / "pipeline_metrics" / "ablation_metrics"),
        "PIPELINE_DATAFRAMES_METRICS_PATH": ensure_dir(root_path / "outputs" / "metrics" / "pipeline_metrics" / "pipeline_dataframes_metrics"),
        "ABLATION_MODELS_FOR_SHAP_PATH": ensure_dir(root_path / "outputs" / "ablation_models_for_shap"),
        "PLOTS_PATH"             : ensure_dir(root_path / "outputs" / "plots"),
        "ABLATION_PLOTS_PATH"    : ensure_dir(root_path / "outputs" / "plots" / "ablation_polts"),
        "FEATURE_IMPORTANCE_PATH": ensure_dir(root_path / "outputs" / "plots" / "feature_importance"),
        "FIGURES_PATH"                  : ensure_dir(root_path / "outputs" / "figures"),
        "FEATURE_IMPORTANCE_SHAP_PATH"  : ensure_dir(root_path / "outputs" / "figures" / "shap" / "feature_importance"),
        "PER_FEATURE_SHAP_PATH"         : ensure_dir(root_path / "outputs" / "figures" / "shap" / "per_feature"),
        "SAMPLES_SHAP_PATH"             : ensure_dir(root_path / "outputs" / "figures" / "shap" / "samples"),
        "RUNTIME_MEMORY_TASK1_PATH": ensure_dir(root_path / "reports" / "runtime_memory_task1"),
        "RUNTIME_MEMORY_TASK2_PATH": ensure_dir(root_path / "reports" / "runtime_memory_task2"),
        "RUNTIME_MEMORY_TASK3_PATH": ensure_dir(root_path / "reports" / "runtime_memory_task3"),
        "RUNTIME_MEMORY_TASK2_BOOSTINGS_PATH"      : ensure_dir(root_path / "reports" / "runtime_memory_task2" / "boosting"),
        "RUNTIME_MEMORY_TASK2_BOOSTINGS_TUNED_PATH": ensure_dir(root_path / "reports" / "runtime_memory_task2" / "boosting_tuned"),
        "TASK1_COMPARISONS_PATH": ensure_dir(root_path / "comparisons" / "task1_comparisons"),
        "TASK2_COMPARISONS_PATH": ensure_dir(root_path / "comparisons" / "task2_comparisons"),
        "TASK3_COMPARISONS_PATH": ensure_dir(root_path / "comparisons" / "task3_comparisons"),
        "TASK5_COMPARISONS_PATH": ensure_dir(root_path / "comparisons" / "task5_comparisons"),
    }

    caller_globals = inspect.currentframe().f_back.f_globals
    for key, value in paths_dict.items():
        caller_globals[key] = value

    return tuple(paths_dict.values())


