"""
===============================================================================
RETRIEVAL STATISTICAL ANALYSIS
===============================================================================

Purpose:
    Combines the previously generated retrieval-rank and cosine-similarity
    data and performs the statistical analyses used to evaluate the
    relationship between embedding stability, image resolution, and retrieval
    performance.

Inputs:
    outputs/<RUN_NAME>/csv/<dataset>_retrieval.csv
        Contains the retrieval rank for each image, model, and resolution.

    outputs/<RUN_NAME>/csv/<dataset>_cosine.csv
        Contains the cosine similarity between each image's base-resolution
        embedding and its embedding at each resolution.

Workflow:
    1. Load the retrieval-rank and cosine-similarity CSV files using
       pandas.read_csv().

    2. For each model, combine retrieval ranks and cosine similarities into a
       long-format analysis table using the custom build_long_table() function
       imported from Funcs_retrieval_stats.py.

    3. Create an inferential-analysis dataset by excluding the native/base
       resolution. At native resolution, the query and gallery embeddings are
       identical by construction, producing Cosine_Sim = 1 and Top-1 = True
       for every image; these values are therefore retained for descriptive
       statistics but excluded from inferential tests.

    4. Compute descriptive retrieval-performance statistics using the custom
       compute_retrieval_stats() function imported from
       Funcs_retrieval_stats.py.

    5. Fit the primary GEE model using the custom fit_gee() function from
       Funcs_retrieval_stats.py, testing retrieval performance as a function
       of standardized embedding stability (Stability_z) and standardized
       image resolution (Resolution_z).

    6. Fit a resolution-only GEE model using fit_gee() from
       Funcs_retrieval_stats.py to assess the effect of resolution without
       embedding stability as a predictor.

    7. Fit an alternative GEE model using log-transformed standardized
       resolution (Resolution_log_z) with fit_gee_log_resolution() from
       Funcs_retrieval_stats.py, both with and without Stability_z.

    8. Check predictor collinearity using the custom check_collinearity()
       function imported from Funcs_retrieval_stats.py.

    9. Calculate the correlation between embedding stability and retrieval
       performance separately at each resolution using the custom
       per_resolution_correlation() function from Funcs_retrieval_stats.py.

   10. Perform mediation analysis using the custom
       run_mediation_analysis() function from Funcs_retrieval_stats.py.

   11. Compare ResNet18 and ViT retrieval outcomes using McNemar's test through
       the custom run_mcnemar_resnet_vs_vit() function from
       Funcs_retrieval_stats.py.

   12. Convert GEE model results into pandas DataFrames using the local
       gee_result_to_df() helper function defined in this script.

   13. Combine results across datasets and models using pandas.concat() and
       save each analysis result as a separate CSV file using
       DataFrame.to_csv().

Output:
    outputs/<RUN_NAME>/csv/
        retrievals_ranks.csv
        retrieval_accuracy_stats.csv
        retrieval_gee_summary_full.csv
        retrieval_gee_summary_resolution_only.csv
        retrieval_gee_summary_full_logres.csv
        retrieval_gee_summary_resolution_only_logres.csv
        retrieval_collinearity.csv
        retrieval_per_resolution_correlation.csv
        mediation_analysis.csv
        resnet18_vs_vit_mcnemar.csv

Custom functions used:
    Funcs_retrieval_stats.py:
        build_long_table()
        compute_retrieval_stats()
        fit_gee()
        fit_gee_log_resolution()
        check_collinearity()
        per_resolution_correlation()
        run_mediation_analysis()
        run_mcnemar_resnet_vs_vit()

    This script:
        gee_result_to_df()

Public/external functions used:
    pandas.read_csv()
    pandas.DataFrame()
    pandas.concat()
    DataFrame.to_csv()
    os.path.isfile()

Statistical library:
    statsmodels.api is imported in this script but no statsmodels function is
    called directly here; the statistical model fitting is delegated to the
    custom functions imported from Funcs_retrieval_stats.py.

===============================================================================
"""

import os
import warnings
import pandas as pd
import statsmodels.api as sm
from Funcs_retrieval_stats import (
    build_long_table,
    compute_retrieval_stats,
    fit_gee,
    fit_gee_log_resolution,
    check_collinearity,
    per_resolution_correlation,
    run_mediation_analysis,
    run_mcnemar_resnet_vs_vit,
)
from config import DATASETS, RUN_NAME
warnings.filterwarnings("ignore")


def gee_result_to_df(result):
    return pd.DataFrame({
        "coef": result.params,
        "std_err": result.bse,
        "p_value": result.pvalues,
    }).rename_axis("Parameter").reset_index()


def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_dir = os.path.join(BASE_DIR, "outputs", RUN_NAME)
    csv_dir = os.path.join(run_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    model_names = ["ResNet18", "VGG16", "ViT"]

    collected = {
        "retrievals_ranks": [],
        "retrieval_accuracy_stats": [],
        "retrieval_gee_summary_full": [],
        "retrieval_gee_summary_resolution_only": [],
        "retrieval_gee_summary_full_logres": [],
        "retrieval_gee_summary_resolution_only_logres": [],
        "retrieval_collinearity": [],
        "retrieval_per_resolution_correlation": [],
        "mediation_analysis": [],
        "resnet18_vs_vit_mcnemar": [],
    }

    for dataset_name, dataset_cfg in DATASETS.items():

        print("=" * 60)
        print(f"DATASET: {dataset_name}")
        print("=" * 60)

        sizes = dataset_cfg["sizes"]
        resolution_names = {str(s) for s in sizes}
        base_resolution = dataset_cfg["base_resolution"]

        retrieval_file = os.path.join(csv_dir, f"{dataset_name}_retrieval.csv")
        cosine_file = os.path.join(csv_dir, f"{dataset_name}_cosine.csv")

        if not os.path.isfile(retrieval_file) or not os.path.isfile(cosine_file):
            print(f"Skipping '{dataset_name}' (data not found).")
            continue

        retrieval_all = pd.read_csv(retrieval_file)
        retrieval_all.rename(columns=lambda c: int(c) if c in resolution_names else c, inplace=True)

        cosine_all = pd.read_csv(cosine_file)
        cosine_all.rename(columns=lambda c: int(c) if c in resolution_names else c, inplace=True)

        long_dfs_by_model = {}
        for model_name in model_names:

            print("-" * 60), print(f"Processing {model_name}"), print("-" * 60)

            retrieval_df = retrieval_all[retrieval_all["Model"] == model_name].copy()
            cosine_df = cosine_all[cosine_all["Model"] == model_name].copy()

            if retrieval_df.empty or cosine_df.empty:
                print(f"Skipping '{model_name}' (no rows).")
                continue

            long_df = build_long_table(retrieval_df, cosine_df, sizes, save_csv=False)

            # Exclude native resolution from inferential tests: at that
            # resolution the query embedding IS the gallery embedding, so
            # Cosine_Sim = 1.0 and Top1 = True for every image by
            # construction — not a measured outcome. Keep full long_df
            # for the descriptive accuracy table and saved CSV only.
            inferential_df = long_df[long_df["Resolution"] != base_resolution].copy()

            long_dfs_by_model[model_name] = inferential_df

            accuracy_stats = compute_retrieval_stats(long_df, save_csv=False)

            gee_full = fit_gee(
                inferential_df, predictors=("Stability_z", "Resolution_z"),
                save_csv=False, step_label="STEP 2a",
            )
            gee_full_df = gee_result_to_df(gee_full)

            gee_res_only = fit_gee(
                inferential_df, predictors=("Resolution_z",),
                save_csv=False, step_label="STEP 2b",
            )
            gee_res_only_df = gee_result_to_df(gee_res_only)

            gee_full_logres = fit_gee_log_resolution(
                inferential_df, predictors=("Stability_z", "Resolution_log_z"),
                save_csv=False, step_label="STEP 2c",
            )
            gee_full_logres_df = gee_result_to_df(gee_full_logres)

            gee_res_only_logres = fit_gee_log_resolution(
                inferential_df, predictors=("Resolution_log_z",),
                save_csv=False, step_label="STEP 2d",
            )
            gee_res_only_logres_df = gee_result_to_df(gee_res_only_logres)

            collinearity = check_collinearity(inferential_df, save_csv=False)

            correlation = per_resolution_correlation(inferential_df, save_csv=False)
            mediation = run_mediation_analysis(inferential_df)
            mediation.insert(0, "Model", model_name)
            mediation.insert(0, "Dataset", dataset_name)
            collected["mediation_analysis"].append(mediation)

            long_df.insert(0, "Dataset", dataset_name)
            for df in (accuracy_stats, gee_full_df, gee_res_only_df,
                       gee_full_logres_df, gee_res_only_logres_df,
                       collinearity, correlation):
                df.insert(0, "Model", model_name)
                df.insert(0, "Dataset", dataset_name)

            collected["retrievals_ranks"].append(long_df)
            collected["retrieval_accuracy_stats"].append(accuracy_stats)
            collected["retrieval_gee_summary_full"].append(gee_full_df)
            collected["retrieval_gee_summary_resolution_only"].append(gee_res_only_df)
            collected["retrieval_gee_summary_full_logres"].append(gee_full_logres_df)
            collected["retrieval_gee_summary_resolution_only_logres"].append(gee_res_only_logres_df)
            collected["retrieval_collinearity"].append(collinearity)
            collected["retrieval_per_resolution_correlation"].append(correlation)

            print(f"{model_name} complete.\n")

        if "ResNet18" in long_dfs_by_model and "ViT" in long_dfs_by_model:
            mcnemar_df = run_mcnemar_resnet_vs_vit(
                long_dfs_by_model["ResNet18"], long_dfs_by_model["ViT"]
            )
            mcnemar_df.insert(0, "Dataset", dataset_name)
            collected["resnet18_vs_vit_mcnemar"].append(mcnemar_df)

    for output_name, dfs in collected.items():
        if not dfs:
            print(f"[NO DATA] {output_name}.csv")
            continue

        combined = pd.concat(dfs, ignore_index=True)
        output_path = os.path.join(csv_dir, f"{output_name}.csv")
        combined.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")

    print("=" * 60), print("RETRIEVAL STATISTICAL ANALYSIS COMPLETE"), print("=" * 60)


if __name__ == "__main__":
    main()