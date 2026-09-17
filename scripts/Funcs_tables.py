"""
===============================================================================
STATISTICAL TABLE EXPORT FUNCTIONS
===============================================================================

Purpose:
    Provides functions for collecting statistical results from CSV files,
    formatting them into publication-ready tables, and exporting the tables
    as Excel workbooks.

Functions:
    export_all_tables()
        Loads the statistical analysis results, formats the main and
        supplementary tables, calculates GEE odds ratios and confidence
        intervals, and exports all tables to Excel files.

Outputs:
    Table 1
        Collinearity statistics including Pearson correlation and VIF.

    Tables S1-S11
        Descriptive statistics, repeated-measures ANOVA, post-hoc comparisons,
        retrieval accuracy, GEE analyses, log-resolution sensitivity analysis,
        mediation analysis, McNemar tests, per-resolution correlations, and
        normality diagnostics.

Note:
    Table S7 optionally incorporates bootstrap confidence intervals for
    mediation shrinkage when the corresponding bootstrap results file exists.

===============================================================================
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import norm


def export_all_tables(csv_dir, table_dir):
    os.makedirs(table_dir, exist_ok=True)

    collinearity = pd.read_csv(os.path.join(csv_dir, "retrieval_collinearity.csv"))
    descriptive = pd.read_csv(os.path.join(csv_dir, "all_cosine_stats.csv"))
    anova = pd.read_csv(os.path.join(csv_dir, "anova_summary.csv"))
    posthoc = pd.read_csv(os.path.join(csv_dir, "posthoc_results.csv"))
    retrieval_accuracy = pd.read_csv(os.path.join(csv_dir, "retrieval_accuracy_stats.csv"))
    gee_full = pd.read_csv(os.path.join(csv_dir, "retrieval_gee_summary_full.csv"))
    gee_res_only = pd.read_csv(os.path.join(csv_dir, "retrieval_gee_summary_resolution_only.csv"))
    gee_log_full = pd.read_csv(os.path.join(csv_dir, "retrieval_gee_summary_full_logres.csv"))
    gee_log_res_only = pd.read_csv(os.path.join(csv_dir, "retrieval_gee_summary_resolution_only_logres.csv"))
    mediation = pd.read_csv(os.path.join(csv_dir, "mediation_analysis.csv"))
    per_res_corr = pd.read_csv(os.path.join(csv_dir, "retrieval_per_resolution_correlation.csv"))
    mcnemar = pd.read_csv(os.path.join(csv_dir, "resnet18_vs_vit_mcnemar.csv"))
    anova_norm = pd.read_csv(os.path.join(csv_dir, "anova_residual_normality_results.csv"))
    paired_norm = pd.read_csv(os.path.join(csv_dir, "paired_diff_normality_results.csv"))

    # Table 1
    collinearity_wide = collinearity.pivot(
        index=["Dataset", "Model"], columns="metric", values="value"
    ).reset_index()
    collinearity_wide = collinearity_wide.rename(
        columns={"Pearson_r": "Pearson r", "VIF_Resolution_z": "VIF"}
    )
    collinearity_wide = collinearity_wide[["Dataset", "Model", "Pearson r", "VIF"]]
    collinearity_wide = collinearity_wide.round({"Pearson r": 3, "VIF": 2})

    # Table S1
    descriptive = descriptive[
        ["Dataset", "Model", "resolution", "mean", "std", "ci_lower", "ci_upper"]
    ].round({"mean": 4, "std": 4, "ci_lower": 4, "ci_upper": 4})

    # Table S2
    anova = anova.round(4)

    # Table S3
    posthoc = posthoc[
        ["Dataset", "Resolution", "Comparison", "Mean_Difference",
         "CI_Lower", "CI_Upper", "Holm_p", "Cohens_dz", "Significant"]
    ].round(4)

    # Table S4
    retrieval_accuracy = retrieval_accuracy.round(4)

    # Table S5
    z = norm.ppf(0.975)
    gee_full.insert(2, "GEE_Model", "Full")
    gee_res_only.insert(2, "GEE_Model", "Resolution-only")
    gee = pd.concat([gee_full, gee_res_only], ignore_index=True)
    gee["Odds_Ratio"] = np.exp(gee["coef"])
    gee["OR_CI_Lower"] = np.exp(gee["coef"] - z * gee["std_err"])
    gee["OR_CI_Upper"] = np.exp(gee["coef"] + z * gee["std_err"])
    gee = gee.round({
        "coef": 4, "std_err": 4, "p_value": 4,
        "Odds_Ratio": 3, "OR_CI_Lower": 3, "OR_CI_Upper": 3
    })

    # Table S6
    gee_log_full.insert(2, "GEE_Model", "Full")
    gee_log_res_only.insert(2, "GEE_Model", "Resolution-only")
    gee_log = pd.concat([gee_log_full, gee_log_res_only], ignore_index=True)
    gee_log["Odds_Ratio"] = np.exp(gee_log["coef"])
    gee_log["OR_CI_Lower"] = np.exp(gee_log["coef"] - z * gee_log["std_err"])
    gee_log["OR_CI_Upper"] = np.exp(gee_log["coef"] + z * gee_log["std_err"])
    gee_log = gee_log.round({
        "coef": 4, "std_err": 4, "p_value": 4,
        "Odds_Ratio": 3, "OR_CI_Lower": 3, "OR_CI_Upper": 3
    })

    # Table S7
    mediation = mediation.drop(columns=["Mediation_pattern_consistent"])
    bootstrap_file = os.path.join(csv_dir, "mediation_shrinkage_bootstrap.csv")

    if os.path.isfile(bootstrap_file):
        bootstrap_ci = pd.read_csv(bootstrap_file)
        mediation = mediation.merge(bootstrap_ci, on=["Dataset", "Model"], how="left")
        mediation = mediation.round({
            "Shrinkage_pct": 2,
            "Shrinkage_pct_CI_Lower": 2,
            "Shrinkage_pct_CI_Upper": 2
        })
    else:
        print(
            "NOTE: mediation_shrinkage_bootstrap.csv not found — "
            "TableS7 will not include shrinkage % confidence intervals. "
            "Run 8_main_bootstrap_shrinkage.py to generate it."
        )

    # Table S10
    anova_norm = anova_norm.round({"Skewness": 4, "Excess_Kurtosis": 4})

    # Table S11
    paired_norm = paired_norm.round({"Skewness": 4, "Excess_Kurtosis": 4})

    collinearity_wide.to_excel(os.path.join(table_dir, "Table1_Collinearity.xlsx"), index=False)
    descriptive.to_excel(os.path.join(table_dir, "TableS1_Descriptive_Statistics.xlsx"), index=False)
    anova.to_excel(os.path.join(table_dir, "TableS2_RM_ANOVA.xlsx"), index=False)
    posthoc.to_excel(os.path.join(table_dir, "TableS3_PostHoc_Comparisons.xlsx"), index=False)
    retrieval_accuracy.to_excel(os.path.join(table_dir, "TableS4_Retrieval_Accuracy.xlsx"), index=False)
    gee.to_excel(os.path.join(table_dir, "TableS5_GEE_Retrieval_Odds_Ratios.xlsx"), index=False)
    gee_log.to_excel(os.path.join(table_dir, "TableS6_GEE_Log_Resolution_Sensitivity.xlsx"), index=False)
    mediation.to_excel(os.path.join(table_dir, "TableS7_Mediation_Analysis.xlsx"), index=False)
    per_res_corr.to_excel(os.path.join(table_dir, "TableS8_Per_Resolution_Correlation.xlsx"), index=False)
    mcnemar.to_excel(os.path.join(table_dir, "TableS9_McNemar_Results.xlsx"), index=False)
    anova_norm.to_excel(os.path.join(table_dir, "TableS10_ANOVA_Residual_Normality.xlsx"), index=False)
    paired_norm.to_excel(os.path.join(table_dir, "TableS11_Paired_Diff_Normality.xlsx"), index=False)

    print("All tables exported.")