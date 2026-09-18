"""
===============================================================================
COSINE SIMILARITY STATISTICAL ANALYSIS
===============================================================================

Purpose:
    Performs descriptive and cross-model inferential analyses of cosine
    similarity across datasets, models, and image resolutions.

Input:
    outputs/<RUN_NAME>/csv/<dataset>_cosine.csv

    Contains the pairwise cosine similarity between each image's base-resolution
    embedding and its embedding at each configured resolution.

Workflow:
    1. Load each dataset's cosine-similarity CSV using pandas.read_csv().

    2. Compute descriptive statistics for cosine similarity using the custom
       compute_similarity_stats() function from Funcs_stats.py.

    3. Exclude the native/base resolution from inferential analyses because
       each embedding is compared with itself, producing Cosine_Sim = 1.0 for
       every image. It remains included in descriptive statistics.

    4. Validate the dataset structure and required resolution columns using the
       custom validate_dataset() function from Funcs_stats.py.

    5. Convert the wide-format data to long format using the custom
       reshape_to_long() function from Funcs_stats.py.

    6. Perform a two-way repeated-measures ANOVA using the custom
       run_two_way_rm_anova() function from Funcs_stats.py to test the effects
       of model, resolution, and their interaction on Cosine_Sim.

    7. Test normality of paired differences using the custom
       check_paired_diff_normality() function from Funcs_stats.py.

    8. Test normality of ANOVA residuals using the custom
       check_anova_residual_normality() function from Funcs_stats.py.

    9. Perform post-hoc comparisons following the ANOVA using the custom
       run_posthoc_tests() function from Funcs_stats.py.

   10. Combine results across datasets using pandas.concat() and save each
       analysis table as a separate CSV using DataFrame.to_csv().

Output:
    outputs/<RUN_NAME>/csv/
        all_cosine_stats.csv
        anova_summary.csv
        paired_diff_normality_results.csv
        anova_residual_normality_results.csv
        posthoc_results.csv

Custom functions:
    Funcs_stats.py:
        compute_similarity_stats()
        validate_dataset()
        reshape_to_long()
        run_two_way_rm_anova()
        check_paired_diff_normality()
        check_anova_residual_normality()
        run_posthoc_tests()

Public/external functions:
    pandas.read_csv()
    pandas.concat()
    DataFrame.to_csv()
    os.path.isfile()

Configuration:
    DATASETS and RUN_NAME are imported from config.py.
    The analyzed metric is cosine similarity (Cosine_Sim).

===============================================================================
"""

import os
import warnings
import pandas as pd

from Funcs_vis import (
    plot_metric_curve_multi,
    plot_metric_heatmap_multi,
    plot_stability_vs_retrieval_multi,
    plot_effect_sizes_multi,
    plot_gee_odds_ratios,
    plot_path_c_to_cprime,
    plot_aid_overview,
)

from config import DATASETS, RUN_NAME

warnings.filterwarnings("ignore")


def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_dir = os.path.join(BASE_DIR, "outputs", RUN_NAME)
    csv_dir = os.path.join(run_dir, "csv")
    figure_dir = os.path.join(run_dir, "figures")

    if not os.path.isdir(csv_dir):
        print(f"'{csv_dir}' not found. Run the previous pipeline stages first.")
        return

    os.makedirs(figure_dir, exist_ok=True)

    dataset_names = list(DATASETS.keys())

    cosine_stats_all = pd.read_csv(os.path.join(csv_dir, "all_cosine_stats.csv"))
    posthoc_all = pd.read_csv(os.path.join(csv_dir, "posthoc_results.csv"))
    retrieval_accuracy_all = pd.read_csv(os.path.join(csv_dir, "retrieval_accuracy_stats.csv"))

    def split_by_dataset_model(df):
        result = {}
        for dataset_name in dataset_names:
            subset = df[df["Dataset"] == dataset_name]
            result[dataset_name] = {
                model: subset[subset["Model"] == model].copy()
                for model in subset["Model"].unique()
            }
        return result

    def build_retrieval_stats(rename_map):
        result = {}
        for dataset_name in dataset_names:
            subset = retrieval_accuracy_all[retrieval_accuracy_all["Dataset"] == dataset_name]
            result[dataset_name] = {
                model: subset[subset["Model"] == model].copy().rename(columns=rename_map)
                for model in subset["Model"].unique()
            }
        return result

    cosine_stats_by_dataset = split_by_dataset_model(cosine_stats_all)

    # 1. Cosine similarity curve
    plot_metric_curve_multi(
        data_by_dataset=cosine_stats_by_dataset,
        metric_name="mean", ylabel="Cosine Similarity",
        title="Cosine Similarity vs Image Resolution",
        save_path=os.path.join(figure_dir, "Figure_S1.png"),
        show_ci=True,
    )

    # 2. Cosine heatmap
    heatmap_by_dataset = {
        dataset_name: cosine_stats_all[cosine_stats_all["Dataset"] == dataset_name]
        for dataset_name in dataset_names
    }
    plot_metric_heatmap_multi(
        df_by_dataset=heatmap_by_dataset, metric_name="mean",
        title="Cosine Similarity Heatmap",
        save_path=os.path.join(figure_dir, "Figure_S5.png"),
    )

    # 3. Cosine effect sizes
    posthoc_by_dataset = {
        dataset_name: posthoc_all[posthoc_all["Dataset"] == dataset_name]
        for dataset_name in dataset_names
    }
    plot_effect_sizes_multi(
        posthoc_by_dataset=posthoc_by_dataset,
        title="Cosine Similarity Effect Sizes",
        save_path=os.path.join(figure_dir, "Figure_S6.png"),
    )

    # 4. Retrieval accuracy curve (Top-1)
    retrieval_stats = build_retrieval_stats({
        "Top1_Acc": "mean", "Top1_CI_Lower": "ci_lower", "Top1_CI_Upper": "ci_upper",
    })
    plot_metric_curve_multi(
        data_by_dataset=retrieval_stats,
        metric_name="mean", ylabel="Top-1 Retrieval Accuracy",
        title="Top-1 Retrieval Accuracy vs Image Resolution",
        save_path=os.path.join(figure_dir, "Figure_S2.png"),
        show_ci=True,
    )

    # 5. Retrieval accuracy curve (Top-5)
    retrieval_top5_stats = build_retrieval_stats({
        "Top5_Acc": "mean", "Top5_CI_Lower": "ci_lower", "Top5_CI_Upper": "ci_upper",
    })
    plot_metric_curve_multi(
        data_by_dataset=retrieval_top5_stats,
        metric_name="mean", ylabel="Top-5 Retrieval Accuracy",
        title="Top-5 Retrieval Accuracy vs Image Resolution",
        save_path=os.path.join(figure_dir, "Figure_S3.png"),
        show_ci=True,
    )

    # 6. Mean retrieval rank curve
    retrieval_mean_rank_stats = build_retrieval_stats({"Mean_Rank": "mean"})
    plot_metric_curve_multi(
        data_by_dataset=retrieval_mean_rank_stats,
        metric_name="mean", ylabel="Mean Retrieval Rank (lower is better)",
        title="Mean Retrieval Rank vs Image Resolution",
        save_path=os.path.join(figure_dir, "Figure_S4.png"),
        show_ci=False,
    )

    # 7. Stability vs retrieval
    plot_stability_vs_retrieval_multi(
        cosine_by_dataset=cosine_stats_by_dataset,
        retrieval_by_dataset=retrieval_stats,
        save_path=os.path.join(figure_dir, "Figure_3.png"),
    )

    model_order = ["ResNet18", "VGG16", "ViT"]

    gee_full_all = pd.read_csv(os.path.join(csv_dir, "retrieval_gee_summary_full.csv"))
    plot_gee_odds_ratios(
        gee_full_all, dataset_order=dataset_names, model_order=model_order,
        save_path=os.path.join(figure_dir, "Figure_1.png"),
    )

    mediation_all = pd.read_csv(os.path.join(csv_dir, "mediation_analysis.csv"))
    plot_path_c_to_cprime(
        mediation_all, dataset_order=dataset_names, model_order=model_order,
        save_path=os.path.join(figure_dir, "Figure_2.png"),
    )

    plot_aid_overview(
        csv_dir, dataset_names=dataset_names, model_order=model_order,
        save_path=os.path.join(figure_dir, "Aid_overview.png"),
        max_resolution=None,
    )

    plot_aid_overview(
        csv_dir, dataset_names=dataset_names, model_order=model_order,
        save_path=os.path.join(figure_dir, "Aid_overview_96px.png"),
        max_resolution=96,
    )

    print("=" * 60), print("VISUALIZATION COMPLETE"), print("=" * 60)


if __name__ == "__main__":
    main()