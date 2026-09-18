"""
===============================================================================
COSINE SIMILARITY STATISTICAL ANALYSIS FUNCTIONS
===============================================================================

Purpose:
    Provides functions for validating, reshaping, summarizing, and statistically
    analyzing cosine-similarity data across models and image resolutions.

Functions:
    compute_similarity_stats()
        Calculates descriptive statistics and confidence intervals for each
        model and resolution.

    validate_dataset()
        Checks the input data for required columns, missing values, duplicate
        image/model combinations, complete model coverage, and balance.

    reshape_to_long()
        Converts the cosine-similarity data from wide format to long format,
        with one row per image, model, and resolution.

    check_anova_residual_normality()
        Fits the Model × Resolution factorial model and evaluates its residuals
        using skewness and excess kurtosis.

    run_two_way_rm_anova()
        Performs a two-way repeated-measures ANOVA testing the effects of model,
        resolution, and their interaction on cosine similarity.

    check_paired_diff_normality()
        Evaluates the normality of paired cosine-similarity differences between
        model pairs at each resolution using skewness and excess kurtosis.

    run_posthoc_tests()
        Performs paired t-tests between model pairs at each resolution when the
        Model × Resolution interaction is significant, with Holm correction.

    run_full_statistical_analysis()
        Runs the complete cosine-similarity statistical pipeline, from dataset
        validation and reshaping through ANOVA, normality checks, and post-hoc
        comparisons.

===============================================================================
"""

import os
import pingouin as pg
import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests
from scipy import stats as sstats
from statsmodels.formula.api import ols


def compute_similarity_stats(combined_df, sizes, confidence=0.95):

    def compute_single_stats(df, model_name):

        n = len(df)
        t_crit = sstats.t.ppf((1 + confidence) / 2, df=n - 1)

        rows = []

        for s in sizes:

            vals = df[s]

            mean = vals.mean()
            std = vals.std(ddof=1)
            sem = std / np.sqrt(n)
            margin = t_crit * sem

            rows.append({
                "Model": model_name,
                "resolution": s,
                "n": n,
                "mean": mean,
                "std": std,
                "sem": sem,
                "ci_lower": mean - margin,
                "ci_upper": mean + margin,
            })

        return pd.DataFrame(rows)

    stats_tables = {}

    all_tables = []

    for model_name, model_df in combined_df.groupby("Model"):

        stats_df = compute_single_stats(model_df, model_name)

        stats_tables[model_name] = stats_df

        all_tables.append(stats_df)

    stats_tables["All"] = pd.concat(
        all_tables,
        ignore_index=True
    )

    return stats_tables

def validate_dataset(combined_df, sizes):

    print("=" * 60)
    print("STEP 1 - VALIDATING DATASET")
    print("=" * 60)

    # ---------------------------------------------------------
    # Check required columns
    # ---------------------------------------------------------
    required_columns = {"image_id", "Model"} | set(sizes)

    missing_columns = required_columns - set(combined_df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    print("✓ Required columns present.")

    # ---------------------------------------------------------
    # Check missing values
    # ---------------------------------------------------------
    if combined_df[list(required_columns)].isnull().any().any():

        missing = combined_df[list(required_columns)].isnull().sum()

        raise ValueError(
            "Dataset contains missing values:\n"
            f"{missing[missing > 0]}"
        )

    print("✓ No missing values.")

    # ---------------------------------------------------------
    # Check duplicate Image × Model rows
    # ---------------------------------------------------------
    duplicates = combined_df.duplicated(
        subset=["image_id", "Model"]
    )

    if duplicates.any():

        raise ValueError(
            f"Found {duplicates.sum()} duplicate "
            "(image_id, Model) rows."
        )

    print("✓ No duplicate Image × Model rows.")

    # ---------------------------------------------------------
    # Check every image has every model
    # ---------------------------------------------------------
    expected_models = {"ResNet18", "VGG16", "ViT"}

    counts = combined_df.groupby("image_id")["Model"].nunique()

    bad_images = counts[counts != len(expected_models)]

    if not bad_images.empty:

        raise ValueError(
            f"{len(bad_images)} image(s) do not contain "
            "all three models."
        )

    print("✓ Every image contains all models.")

    # ---------------------------------------------------------
    # Check each model has same number of images
    # ---------------------------------------------------------
    model_counts = combined_df["Model"].value_counts()

    if model_counts.nunique() != 1:

        raise ValueError(
            "Models contain different numbers of images:\n"
            f"{model_counts}"
        )

    print("✓ Balanced dataset.")

    print("Dataset validation completed successfully.\n")

def reshape_to_long(combined_df, sizes, value_name="Value"):

    print("=" * 60)
    print("STEP 2 - RESHAPING DATA")
    print("=" * 60)

    long_df = combined_df.melt(
        id_vars=["image_id", "Model"],
        value_vars=sizes,
        var_name="Resolution",
        value_name=value_name
    )

    # Make sure resolutions are numeric
    long_df["Resolution"] = long_df["Resolution"].astype(int)

    # Sort for readability and reproducibility
    long_df = long_df.sort_values( by=["image_id", "Model", "Resolution"] ).reset_index(drop=True)

    print(f"Original shape : {combined_df.shape}")
    print(f"Long shape     : {long_df.shape}")
    print("Data successfully converted to long format.\n")

    return long_df

def check_anova_residual_normality(long_df, value_name="Value", save_csv=True, output_file="outputs/anova_residual_normality_results.csv"):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Create categorical factors
    df = long_df.copy()
    df["Model"] = df["Model"].astype("category")
    df["Resolution"] = df["Resolution"].astype("category")

    # Fit the Model × Resolution factorial model
    model = ols(
        f"{value_name} ~ C(Model) * C(Resolution)",
        data=df
    ).fit()

    residuals = model.resid

    # Descriptive normality diagnostics
    skew = sstats.skew(residuals)
    kurtosis = sstats.kurtosis(residuals)

    # Shapiro-Wilk is not practical for N=336,000,
    # so use skewness/kurtosis for the large sample.
    results = pd.DataFrame([{
        "N": len(residuals),
        "Skewness": skew,
        "Excess_Kurtosis": kurtosis
    }])

    if save_csv:
        results.to_csv(output_file, index=False)

    return results

def run_two_way_rm_anova(long_df, dv="Value", alpha=0.05, save_csv=True, output_file="outputs/anova_summary.csv"):
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print("=" * 60), print("STEP 3 - TWO-WAY REPEATED-MEASURES ANOVA"), print("=" * 60)

    # ---------------------------------------------------------
    # Run repeated-measures ANOVA
    # ---------------------------------------------------------

    anova = pg.rm_anova( data=long_df, dv=dv, within=["Model", "Resolution"],
        subject="image_id", correction="auto", detailed=True, effsize="ng2")

    # ---------------------------------------------------------
    # Use Greenhouse-Geisser corrected p-values when available
    # ---------------------------------------------------------

    if "p_GG_corr" in anova.columns:
        anova["Final_p"] = anova["p_GG_corr"].fillna(anova["p_unc"])
    else:
        anova["Final_p"] = anova["p_unc"]

    # ---------------------------------------------------------
    # Print ANOVA table
    # ---------------------------------------------------------

    print("\nANOVA Results\n")
    print(anova)

    # ---------------------------------------------------------
    # Report sphericity information (if available)
    # ---------------------------------------------------------

    print("\nSphericity / Greenhouse-Geisser Information")

    if "eps" in anova.columns:
        print(anova[["Source", "eps"]])

    if "p_GG_corr" in anova.columns:
        print("\nGreenhouse-Geisser corrected p-values are being used when necessary.")
    else:
        print("\nNo Greenhouse-Geisser correction was applied.")

    # ---------------------------------------------------------
    # Find Model × Resolution interaction
    # ---------------------------------------------------------

    interaction_row = anova[
        anova["Source"].astype(str).str.contains("Model", case=False)
        &
        anova["Source"].astype(str).str.contains("Resolution", case=False)
    ]

    if interaction_row.empty:
        raise ValueError(
            "Could not locate the Model × Resolution interaction in the ANOVA table."
        )

    interaction_p = interaction_row.iloc[0]["Final_p"]

    interaction_significant = interaction_p < alpha

    print("\n" + "-" * 60)

    if interaction_significant:

        print(f"Model × Resolution interaction is SIGNIFICANT (p = {interaction_p:.6g})")
        print("Proceed to post-hoc comparisons.")

    else:

        print(f"Model × Resolution interaction is NOT significant (p = {interaction_p:.6g})")
        print("All models appear to degrade similarly.")
        print("Post-hoc comparisons are unnecessary.")

    print("-" * 60)

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    if save_csv:
        anova.to_csv(output_file, index=False)
        print(f"\nANOVA table saved to: {output_file}")

    print("\nSTEP 3 COMPLETE\n")

    return {
        "anova": anova,
        "interaction_significant": interaction_significant,
        "interaction_p": interaction_p
    }

def check_paired_diff_normality(long_df, value_name="Value", skew_threshold=2.0, kurtosis_threshold=7.0, save_csv=True, output_file="outputs/paired_diff_normality_results.csv"):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    results = []
    resolutions = sorted(long_df["Resolution"].unique())
    comparisons = [("ResNet18", "VGG16"), ("ResNet18", "ViT"), ("VGG16", "ViT")]

    for resolution in resolutions:
        subset = long_df[long_df["Resolution"] == resolution]
        wide = subset.pivot(index="image_id", columns="Model", values=value_name)

        for model1, model2 in comparisons:
            diff = wide[model1] - wide[model2]
            skew = sstats.skew(diff)
            kurtosis = sstats.kurtosis(diff)  # excess kurtosis (scipy default, fisher=True)

            results.append({
                "Resolution": resolution,
                "Comparison": f"{model1} vs {model2}",
                "N": len(diff),
                "Skewness": skew,
                "Excess_Kurtosis": kurtosis,
                "Pass": (abs(skew) <= skew_threshold and abs(kurtosis) <= kurtosis_threshold)
            })

    results_df = pd.DataFrame(results)
    if save_csv:
        results_df.to_csv(output_file, index=False)
    return results_df

def run_posthoc_tests(long_df, anova_results, value_name="Value", alpha=0.05, save_csv=True, output_file="outputs/posthoc_results.csv"):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print("=" * 60)
    print("STEP 5 - POST-HOC ANALYSIS")
    print("=" * 60)

    if not anova_results["interaction_significant"]:
        print("Interaction not significant.")
        print("Skipping post-hoc analysis.\n")
        return pd.DataFrame()

    results = []
    resolutions = sorted(long_df["Resolution"].unique())

    for resolution in resolutions:
        print(f"\nResolution {resolution}")

        subset = long_df[long_df["Resolution"] == resolution]
        wide = subset.pivot(index="image_id", columns="Model", values=value_name)

        comparisons = [("ResNet18", "VGG16"), ("ResNet18", "ViT"), ("VGG16", "ViT")]

        for model1, model2 in comparisons:
            x = wide[model1].values
            y = wide[model2].values
            N = len(x)

            t_stat, p = sstats.ttest_rel(x, y)

            diff = x - y
            mean_diff = np.mean(diff)
            std_diff = np.std(diff, ddof=1)
            sem = std_diff / np.sqrt(N)
            t_crit = sstats.t.ppf(0.975, df=N - 1)
            ci_lower = mean_diff - t_crit * sem
            ci_upper = mean_diff + t_crit * sem

            dz = np.nan if std_diff == 0 else mean_diff / std_diff

            results.append({
                "Resolution": resolution,
                "Comparison": f"{model1} vs {model2}",
                "Mean_Difference": mean_diff,
                "CI_Lower": ci_lower,
                "CI_Upper": ci_upper,
                "t": t_stat,
                "df": N - 1,
                "Raw_p": p,
                "Cohens_dz": dz
            })

    results = pd.DataFrame(results)

    reject, corrected_p, _, _ = multipletests(results["Raw_p"], alpha=alpha, method="holm")
    results["Holm_p"] = corrected_p
    results["Significant"] = reject

    results = results.sort_values(by=["Resolution", "Comparison"]).reset_index(drop=True)

    if save_csv:
        results.to_csv(output_file, index=False)
        print(f"\nSaved to {output_file}")

    print("\nSTEP 5 COMPLETE\n")

    return results

def run_full_statistical_analysis(combined_df, sizes, value_name="Value", results_dir="outputs"):

    normality_file = os.path.join(results_dir, "paired_diff_normality_results.csv")
    anova_normality_file = os.path.join(results_dir, "anova_residual_normality_results.csv")
    anova_file = os.path.join(results_dir, "anova_summary.csv")
    posthoc_file = os.path.join(results_dir, "posthoc_results.csv")
    sphericity_file = os.path.join(results_dir, "sphericity_results.csv")

    validate_dataset(combined_df, sizes)
    long_df = reshape_to_long(combined_df, sizes, value_name=value_name)
    anova_results = run_two_way_rm_anova(long_df, dv=value_name, output_file=anova_file)
    normality_results = check_paired_diff_normality(long_df, value_name=value_name, output_file=normality_file)
    anova_normality_results = check_anova_residual_normality(long_df, value_name=value_name, output_file=anova_normality_file)
    posthoc_results = run_posthoc_tests(long_df, anova_results, value_name=value_name, output_file=posthoc_file)

    print("=" * 60)
    print("STATISTICAL ANALYSIS COMPLETE")
    print("=" * 60)

    return {"normality": normality_results, "anova_normality": anova_normality_results, "anova": anova_results, "posthoc": posthoc_results}