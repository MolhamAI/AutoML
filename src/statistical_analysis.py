import os, sys
from scipy.stats import ttest_rel, wilcoxon



sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
PROJECT_FILE_PATH =  os.getcwd()
THIS_FILE_LOCATION = os.path.join(PROJECT_FILE_PATH, "src")
THIS_FILE_PATH = os.path.join(PROJECT_FILE_PATH, "src")

from utils import *



N_ITERATIONS = 1000
CONFIDENCE_LEVEL = 0.95

def bootstrap_confidence_interval(values, n_iterations=N_ITERATIONS, confidence_level=CONFIDENCE_LEVEL):
    bootstrap_means = []
    size = len(values)
    for _ in range(n_iterations):
        sample = np.random.choice(values, size=size, replace=True)
        bootstrap_means.append(np.mean(sample))
    lower_bound = np.percentile(bootstrap_means, (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(bootstrap_means, (1 + confidence_level) / 2 * 100)
    return lower_bound, upper_bound









def compare_model_with_baseline(df_model, df_baseline, metric_names):
    rows = []
    for metric in metric_names:
        m = df_model[metric].astype(float).values
        b = df_baseline[metric].astype(float).values
        p_t = ttest_rel(m, b).pvalue
        try:
            p_w = wilcoxon(m, b).pvalue
        except:
            p_w = np.nan
        rows.append({
            "metric": metric,
            "p_ttest": p_t,
            "p_wilcoxon": p_w
        })
    return pd.DataFrame(rows)




def load_all_results(metrics_dir, datasets, frameworks):
    results = {d: {} for d in datasets}

    for file in os.listdir(metrics_dir):
        if not file.endswith("_cv_summary.csv"):
            continue
        
        parts = file.split("_")
        framework = parts[0]                  
        dataset = parts[1] + ".csv"          
        
        if dataset in datasets and framework in frameworks:
            file_path = os.path.join(metrics_dir, file)
            df = pd.read_csv(file_path)
            results[dataset][framework] = df

    return results




def build_statistical_summary(all_results, baseline_model_name, metric_names, frameworks):
    rows = []
    for dataset_name, models in all_results.items():
        if baseline_model_name not in models:
            continue
        baseline_df = models[baseline_model_name]
        for fw in frameworks:
            if fw not in models or fw == baseline_model_name:
                continue
            model_df = models[fw]
            for metric in metric_names:
                m = model_df[metric].astype(float).values
                b = baseline_df[metric].astype(float).values
                mean_m = m.mean()
                std_m = m.std()
                ci_low, ci_high = bootstrap_confidence_interval(m)
                p_t = ttest_rel(m, b).pvalue
                try:
                    p_w = wilcoxon(m, b).pvalue
                except:
                    p_w = np.nan
                rows.append({
                    "dataset": dataset_name,
                    "model": fw,
                    "baseline": baseline_model_name,
                    "metric": metric,
                    "mean": mean_m,
                    "std": std_m,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_ttest": p_t,
                    "p_wilcoxon": p_w
                })
    return pd.DataFrame(rows)





def to_mean_std(value):

    parts = value.split("+")
    mean = parts[0].strip()
    std = parts[1].strip()

    return mean, std



    



























