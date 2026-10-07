import argparse

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument(
    "--metric",
    choices=["roc_auc", "cohens_d", "perm_p"],
    default="perm_p",
    help="roc_auc: AUC strength; cohens_d: signed effect size; perm_p: time-shift permutation p-value of the AUC",
)
parser.add_argument("--n-perm", type=int, default=1000, help="Permutations for --metric perm_p")
parser.add_argument("--min-positive", type=int, default=5, help="Minimum windows per class to score a pair")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
P_CLIP = 0.5
rng = np.random.default_rng(args.seed)


def auc_from_ranks(ranks, labels):
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def cohens_d(values, labels):
    pos, neg = values[labels == 1], values[labels == 0]
    pooled = np.sqrt(
        ((len(pos) - 1) * pos.var(ddof=1) + (len(neg) - 1) * neg.var(ddof=1)) / (len(pos) + len(neg) - 2)
    )
    return (pos.mean() - neg.mean()) / pooled if pooled > 0 else np.nan


def permutation_p(values, labels, n_perm):
    # circular time shifts keep the autocorrelation of both the labels and the neural feature
    ranks = rankdata(values)
    observed = abs(auc_from_ranks(ranks, labels) - 0.5)
    n = len(labels)
    shifts = rng.integers(1, n, size=n_perm)
    null = np.array([abs(auc_from_ranks(ranks, np.roll(labels, shift)) - 0.5) for shift in shifts])
    return (1 + np.sum(null >= observed)) / (1 + n_perm)


df_lounge = pd.read_csv("/scratch/timonmerk/nbu_lfp_preproc/clean/vlm_label_try/lounge.csv")
df_sleep = pd.read_csv("/scratch/timonmerk/nbu_lfp_preproc/clean/vlm_label_try/sleep.csv")
# there is a column "chunk_path" with the following format: TRBD001_20260219_091142.23511990_chunk_0000, TRBD001_20260219_091142.23511990_chunk_0001
# I want to extract time, each subsequent chunk is 10 seconds part of the recording, it should be added to the 20260219_091142 time 
def add_time_column(df):
    df["time_offset_chunk"] = df["chunk_path"].str.split("_").str[-1].str.replace("chunk_", "").astype(int) * 15
    df["time_group"] = df["chunk_path"].str.split(".").str[0].apply(lambda x: x[len("TRBD001_"):])
    df["time_group"] = pd.to_datetime(df["time_group"], format="%Y%m%d_%H%M%S")
    df["time"] = df["time_group"] + pd.to_timedelta(df["time_offset_chunk"], unit="s")
    return df

df_lounge = add_time_column(df_lounge)
df_sleep = add_time_column(df_sleep)

# merge the two based on time, cols in lounge should be lounge_ and cols in sleep should be sleep_
df_merged = pd.merge(df_lounge, df_sleep, on="time", how="outer", suffixes=("_lounge", "_sleep"))
# set time to be the index
df_merged = df_merged.set_index("time")
# resample the merged data by 10s
# drop "chunk_path_lounge" and "chunk_path_sleep"
df_merged = df_merged.drop(columns=["chunk_path_lounge", "chunk_path_sleep"])
df_merged = df_merged.resample("10s").mean()
# set time zone to CT
df_merged = df_merged.tz_localize("US/Central")

# now get neural data
df_neural = pd.read_csv("/scratch/timonmerk/nbu_lfp_preproc/clean/analysis_results/daytime_specparam_10s_features.csv")
df_neural = df_neural.query("date == '2026-02-19'")
#rename datetime to be time
df_neural = df_neural.rename(columns={"datetime": "time"})
df_neural["time"] = pd.to_datetime(df_neural["time"])
df_neural = df_neural.set_index("time")

# there is a column "hemisphere" in the neural data
# I want to have the columns for each hemisphere to be prefixed with "hemisphere_", and so low_gamma_band_power_sum_left and low_gamma_band_power_sum_right
# so that there are more columns and less rows (pivoting)

df_neural = df_neural.pivot_table(index="time", columns="hemisphere", values=df_neural.columns.difference(["date", "hemisphere"]), aggfunc="mean")
# join the indexes index1_index2
df_neural.columns = [f"{col[0]}_{col[1]}" for col in df_neural.columns]
# drop columns visit_left and madrs_left, and rename visit_right to be visit and madrs_right to be madrs
df_neural = df_neural.drop(columns=["visit_left", "madrs_left"])
df_neural = df_neural.rename(columns={"visit_right": "visit", "madrs_right": "madrs"})

# resample both by 10s
df_neural = df_neural.resample("10s").mean()

# merge the two: 
df_merged = pd.merge(df_merged, df_neural, left_index=True, right_index=True, how="outer")

# see if there are not NaN values in (alpha_band_power_sum_left or alpha_band_power_sum_right) and (Resting_lounge or Resting_sleep)
 
mask = (
    df_merged[["alpha_band_power_sum_left", "alpha_band_power_sum_right"]]
    .notna().any(axis=1)
    &
    df_merged[["Resting_lounge", "Resting_sleep"]]
    .notna().any(axis=1)
)

df_merged_data = df_merged.loc[mask]

# ok, for each of those neural features: alpha_band_power_(left or right), low_beta_band_power_sum_(left or right), delta, theta, low_gamma and high_gamma
# separate, compute the AUC for each with respect to the VLM labels
# separate for lounge and sleep, based on Fully_Visible_lounge, Fully_Visible_sleep
fband_list = ["delta", "theta", "alpha", "low_beta", "high_beta", "low_gamma", "high_gamma"]
neural_feature_list = [f"{fband}_band_power_sum_{side}" for fband in fband_list for side in ["left", "right"]] + \
    ["aperiodic_exponent_left", "aperiodic_exponent_right", "aperiodic_offset_left", "aperiodic_offset_right"]

behav_columns = df_merged_data.columns.difference(neural_feature_list + ["visit", "madrs", "time_group_lounge", "time_group_sleep", "time_offset_chunk_lounge", "time_offset_chunk_sleep"])

l_rocs = []
for environment in ["lounge", "sleep"]:
    df_env = df_merged_data.query(f"Fully_Visible_{environment} == 1").sort_index()
    for feature in neural_feature_list:
        for behav in behav_columns:
            # remove nan values
            df_env_filtered = df_env.dropna(subset=[behav, feature])
            if df_env_filtered.empty:
                continue
            labels = (df_env_filtered[behav].to_numpy() > 0).astype(int)
            values = df_env_filtered[feature].to_numpy(dtype=float)
            n_pos = int(labels.sum())
            if min(n_pos, len(labels) - n_pos) < args.min_positive:
                continue
            row = {
                "environment": environment,
                "feature": feature,
                "behavior": behav,
                "n_positive": n_pos,
                "n_total": len(labels),
            }
            if args.metric == "roc_auc":
                row["auc"] = roc_auc_score(labels, values)
            elif args.metric == "cohens_d":
                row["cohens_d"] = cohens_d(values, labels)
            else:
                row["auc"] = roc_auc_score(labels, values)
                row["perm_p"] = permutation_p(values, labels, args.n_perm)
            l_rocs.append(row)

df_rocs = pd.DataFrame(l_rocs)
if args.metric == "roc_auc":
    df_rocs["plot_value"] = df_rocs["auc"].apply(lambda x: max(x, 1 - x))
    heatmap_kwargs = {"cmap": "coolwarm", "cbar_kws": {"label": "AUC Strength"}}
    title = "AUC of Neural Features in Different Environments"
    output_name = "auc_neural_features.pdf"
elif args.metric == "cohens_d":
    df_rocs["plot_value"] = df_rocs["cohens_d"]
    heatmap_kwargs = {"cmap": "coolwarm", "center": 0, "cbar_kws": {"label": "Cohen's d (behavior present - absent)"}}
    title = "Cohen's d of Neural Features in Different Environments"
    output_name = "cohens_d_neural_features.pdf"
else:
    # p-values above P_CLIP are clipped to P_CLIP, not hidden
    df_rocs["plot_value"] = df_rocs["perm_p"].clip(upper=P_CLIP)
    heatmap_kwargs = {
        "cmap": "viridis_r",
        "vmin": 0,
        "vmax": 0.1,
        "fmt": ".3f",
        "cbar_kws": {"label": f"Permutation p (clipped at {P_CLIP})"},
    }
    title = f"Time-shift permutation p of AUC ({args.n_perm} permutations)"
    output_name = "perm_p_neural_features.pdf"

# now make a behavior time neural matrix to plot
# sort feature columns by left and right within plot
df_pivot = df_rocs.pivot_table(index="feature", columns="behavior", values="plot_value")

# sort the columns by left and right
plt.figure(figsize=(10, 10))
# sort the columns by left and right
df_pivot = df_pivot.reindex(
    columns=sorted(
        df_pivot.columns,
        key=lambda x: (
            "_".join(x.split("_")[:-1]),
            {"left": 0, "right": 1}.get(x.split("_")[-1].lower(), 2)
        )
    )
)
# set annot fontsize to be 5, set color bar label
sns.heatmap(df_pivot, annot=True, annot_kws={"size": 5}, **heatmap_kwargs)
plt.title(title)
plt.tight_layout()
plt.savefig(f"clean/vlm_label_try/{output_name}")


plt.figure(figsize=(10, 10))
