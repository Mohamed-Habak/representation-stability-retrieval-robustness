"""
===============================================================================
RETRIEVAL STATISTICAL ANALYSIS FUNCTIONS
===============================================================================

Purpose:
    Provides functions for preparing retrieval/stability data and performing
    the statistical analyses used to evaluate retrieval performance.

Functions:
    build_long_table()
        Combines retrieval ranks and cosine similarity into a long-format table
        and derives Top-1 and Top-5 retrieval outcomes.

    fit_gee()
        Fits a GEE logistic regression using standardized stability and/or
        resolution as predictors of retrieval success.

    fit_gee_log_resolution()
        Performs the same GEE analysis using log-transformed resolution as a
        sensitivity analysis.

    check_collinearity()
        Measures the relationship between resolution and stability using
        Pearson correlation and variance inflation factors (VIF).

    compute_retrieval_stats()
        Calculates Top-1/Top-5 accuracy, Wilson confidence intervals, mean rank,
        and median rank for each resolution.

    per_resolution_correlation()
        Measures the relationship between cosine stability and retrieval
        success separately at each resolution using point-biserial correlation,
        with Holm correction for multiple comparisons.

    run_full_retrieval_analysis()
        Runs the main retrieval statistical analysis pipeline by combining the
        data-preparation, descriptive, GEE, collinearity, and correlation
        analyses.

    path_a_resolution_predicts_stability()
        Tests whether image resolution predicts embedding stability.

    path_c_resolution_alone()
        Tests the effect of resolution on retrieval success without stability.

    path_b_and_cprime_full_model()
        Tests stability and resolution together to obtain the stability effect
        and the remaining resolution effect.

    bootstrap_shrinkage_ci()
        Estimates a bootstrap confidence interval for the reduction in the
        resolution effect after accounting for embedding stability.

    run_mediation_analysis()
        Combines the three mediation paths and calculates the resulting
        shrinkage in the resolution effect.

    run_mcnemar_resnet_vs_vit()
        Performs paired McNemar tests comparing ResNet18 and ViT Top-1
        retrieval outcomes at each resolution, with Holm correction.

===============================================================================
"""

import os
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import pointbiserialr
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar


def build_long_table(retrieval_df, cosine_df, sizes, save_csv=True, output_file="outputs/retrievals_ranks.csv"):

    print("=" * 60), print("STEP 1 - BUILDING LONG-FORMAT RETRIEVAL/STABILITY TABLE"), print("=" * 60)

    rank_long = retrieval_df.melt( id_vars=["image_id", "Model"], value_vars=sizes, var_name="Resolution", value_name="Rank")
    cosine_long = cosine_df.melt( id_vars=["image_id", "Model"], value_vars=sizes, var_name="Resolution", value_name="Cosine_Sim"    )

    rank_long["Resolution"] = rank_long["Resolution"].astype(int)
    cosine_long["Resolution"] = cosine_long["Resolution"].astype(int)

    long_df = rank_long.merge(cosine_long, on=["image_id", "Model", "Resolution"], how="inner")

    long_df["Top1"] = long_df["Rank"] == 0
    long_df["Top5"] = long_df["Rank"] < 5

    # GEE requires rows sorted by group (image_id) into contiguous blocks —
    # melt+merge order them by resolution first, which silently breaks the
    # clustered covariance estimation. Sort here, matching reshape_to_long
    # in Funcs_stats.py.
    long_df = long_df.sort_values(by=["image_id", "Model", "Resolution"]).reset_index(drop=True)

    if save_csv:
            long_df.to_csv(output_file, index=False)
            print(f"\nSaved to {output_file}")

    return long_df

def fit_gee(long_df, predictors=("Stability_z", "Resolution_z"), outcome="Top1", save_csv=True, output_file="outputs/retrieval_gee_summary.csv", step_label="STEP 2"):

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print("=" * 60), print(f"{step_label} - GEE: {' + '.join(predictors)} -> {outcome}"), print("=" * 60)

    df = long_df.copy()
    df["Resolution_z"] = (df["Resolution"] - df["Resolution"].mean()) / df["Resolution"].std()
    df["Stability_z"] = (df["Cosine_Sim"] - df["Cosine_Sim"].mean()) / df["Cosine_Sim"].std()
    df[outcome] = df[outcome].astype(int)

    formula = f"{outcome} ~ " + " + ".join(predictors)

    model = smf.gee(formula, groups="image_id", data=df, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable())
    result = model.fit(cov_type="robust")

    if save_csv:
        summary_df = pd.DataFrame({ "coef": result.params, "std_err": result.bse, "p_value": result.pvalues,})
        summary_df.to_csv(output_file)
        print(f"\nGEE summary saved to: {output_file}")

    print(f"\n{step_label} COMPLETE\n")

    return result

def fit_gee_log_resolution(long_df, predictors=("Stability_z", "Resolution_log_z"), outcome="Top1", save_csv=True, output_file="outputs/retrieval_gee_summary.csv", step_label="STEP 2 (log-res sensitivity)"):

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print("=" * 60), print(f"{step_label} - GEE: {' + '.join(predictors)} -> {outcome}"), print("=" * 60)

    df = long_df.copy()
    df["Resolution_log"] = np.log(df["Resolution"])
    df["Resolution_log_z"] = (df["Resolution_log"] - df["Resolution_log"].mean()) / df["Resolution_log"].std()
    df["Stability_z"] = (df["Cosine_Sim"] - df["Cosine_Sim"].mean()) / df["Cosine_Sim"].std()
    df[outcome] = df[outcome].astype(int)

    formula = f"{outcome} ~ " + " + ".join(predictors)

    model = smf.gee(formula, groups="image_id", data=df, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable())
    result = model.fit(cov_type="robust")

    if save_csv:
        summary_df = pd.DataFrame({"coef": result.params, "std_err": result.bse, "p_value": result.pvalues})
        summary_df.to_csv(output_file)
        print(f"\nGEE summary saved to: {output_file}")

    print(f"\n{step_label} COMPLETE\n")

    return result

def check_collinearity(long_df, save_csv=True, output_file="outputs/retrieval_collinearity.csv"):

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print("=" * 60), print("STEP - COLLINEARITY: Resolution_z vs Stability_z"), print("=" * 60)

    df = long_df.copy()
    df["Resolution_z"] = (df["Resolution"] - df["Resolution"].mean()) / df["Resolution"].std()
    df["Stability_z"] = (df["Cosine_Sim"] - df["Cosine_Sim"].mean()) / df["Cosine_Sim"].std()

    pearson_r = df["Resolution_z"].corr(df["Stability_z"])

    X = sm.add_constant(df[["Resolution_z", "Stability_z"]])
    vif_resolution = variance_inflation_factor(X.values, 1)
    vif_stability = variance_inflation_factor(X.values, 2)

    results = pd.DataFrame({
        "metric": ["Pearson_r", "VIF_Resolution_z", "VIF_Stability_z"],
        "value": [pearson_r, vif_resolution, vif_stability],
    })

    if save_csv:
        results.to_csv(output_file, index=False)
        print(f"\nSaved to {output_file}")

    print("\nCOLLINEARITY CHECK COMPLETE\n")

    return results

def compute_retrieval_stats(long_df, confidence=0.95, save_csv=True, output_file="outputs/retrieval_accuracy_stats.csv"):

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print("=" * 60), print("STEP - RETRIEVAL ACCURACY BY RESOLUTION"), print("=" * 60)

    from statsmodels.stats.proportion import proportion_confint

    rows = []
    for res, g in long_df.groupby("Resolution"):
        n = len(g)
        top1_count = int(g["Top1"].sum())
        top5_count = int(g["Top5"].sum())

        top1_acc = top1_count / n
        top5_acc = top5_count / n

        top1_ci_low, top1_ci_high = proportion_confint(top1_count, n, alpha=1 - confidence, method="wilson")
        top5_ci_low, top5_ci_high = proportion_confint(top5_count, n, alpha=1 - confidence, method="wilson")

        rows.append({
            "Resolution": res,
            "N": n,
            "Top1_Acc": top1_acc,
            "Top1_CI_Lower": top1_ci_low,
            "Top1_CI_Upper": top1_ci_high,
            "Top5_Acc": top5_acc,
            "Top5_CI_Lower": top5_ci_low,
            "Top5_CI_Upper": top5_ci_high,
            "Mean_Rank": g["Rank"].mean(),
            "Median_Rank": g["Rank"].median(),
        })

    results = pd.DataFrame(rows).sort_values("Resolution", ascending=False).reset_index(drop=True)

    if save_csv:
        results.to_csv(output_file, index=False)
        print(f"\nSaved to {output_file}")

    print("\nRETRIEVAL ACCURACY STATS COMPLETE\n")

    return results

def per_resolution_correlation(long_df, outcome="Top1", save_csv=True, output_file="outputs/retrieval_per_resolution_correlation.csv"):

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print("=" * 60), print("STEP 3 - PER-RESOLUTION STABILITY-RETRIEVAL CORRELATION"), print("=" * 60)

    rows = []
    for res, g in long_df.groupby("Resolution"):
        if g[outcome].nunique() < 2:
            print(f"Resolution {res}: no variance in {outcome}, skipping.")
            continue
        r, p = pointbiserialr(g[outcome].astype(int), g["Cosine_Sim"])
        rows.append({"Resolution": res, "r": r, "Raw_p": p, "N": len(g)})

    results = pd.DataFrame(rows).sort_values("Resolution", ascending=False).reset_index(drop=True)

    if len(results):
        results["Holm_p"] = multipletests(results["Raw_p"], method="holm")[1]

    if save_csv:
        results.to_csv(output_file, index=False)
        print(f"\nSaved to {output_file}")

    print("\nSTEP 3 COMPLETE\n")

    return results

def run_full_retrieval_analysis(retrieval_df, cosine_df, sizes, results_dir="outputs"):
    rank_file = os.path.join(results_dir, "retrievals_ranks.csv")
    accuracy_file = os.path.join(results_dir, "retrieval_accuracy_stats.csv")
    gee_full_file = os.path.join(results_dir, "retrieval_gee_summary_full.csv")
    gee_res_only_file = os.path.join(results_dir, "retrieval_gee_summary_resolution_only.csv")
    collinearity_file = os.path.join(results_dir, "retrieval_collinearity.csv")
    corr_file = os.path.join(results_dir, "retrieval_per_resolution_correlation.csv")

    long_df = build_long_table(retrieval_df, cosine_df, sizes, output_file=rank_file)

    accuracy_stats = compute_retrieval_stats(long_df, output_file=accuracy_file)

    # Full model: does stability predict retrieval, controlling for resolution?
    gee_full = fit_gee(
        long_df, predictors=("Stability_z", "Resolution_z"),
        output_file=gee_full_file, step_label="STEP 2a",
    )

    # Resolution-only model: does resolution predict retrieval on its own?
    # Needed alongside the full model to support a mediation reading —
    # a large effect here that shrinks in the full model is the expected
    # signature of resolution acting through stability, not around it.
    gee_res_only = fit_gee(
        long_df, predictors=("Resolution_z",),
        output_file=gee_res_only_file, step_label="STEP 2b",
    )

    collinearity = check_collinearity(long_df, output_file=collinearity_file)

    corr_results = per_resolution_correlation(long_df, output_file=corr_file)

    print("=" * 60), print("RETRIEVAL STATISTICAL ANALYSIS COMPLETE"), print("=" * 60)

    return {
        "long_df": long_df,
        "accuracy_stats": accuracy_stats,
        "gee_full": gee_full,
        "gee_resolution_only": gee_res_only,
        "collinearity": collinearity,
        "correlation": corr_results,
    }

def path_a_resolution_predicts_stability(df):
    X = sm.add_constant(df["Resolution_z"])
    model = sm.OLS(df["Stability_z"], X).fit(
        cov_type="cluster", cov_kwds={"groups": df["image_id"]}
    )
    return {
        "PathA_coef": model.params["Resolution_z"],
        "PathA_p": model.pvalues["Resolution_z"],
        "PathA_r2": model.rsquared,
    }

def path_c_resolution_alone(df):
    result = smf.gee(
        "Top1 ~ Resolution_z", groups="image_id", data=df,
        family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable(),
    ).fit()
    return {
        "PathC_resolution_coef": result.params["Resolution_z"],
        "PathC_resolution_p": result.pvalues["Resolution_z"],
    }

def path_b_and_cprime_full_model(df):
    result = smf.gee(
        "Top1 ~ Stability_z + Resolution_z", groups="image_id", data=df,
        family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable(),
    ).fit()
    return {
        "PathB_stability_coef": result.params["Stability_z"],
        "PathB_stability_p": result.pvalues["Stability_z"],
        "PathCprime_resolution_coef": result.params["Resolution_z"],
        "PathCprime_resolution_p": result.pvalues["Resolution_z"],
    }

def bootstrap_shrinkage_ci(long_df, n_boot=200, ci=0.95, seed=42, outcome="Top1", checkpoint_file=None, progress_label=""):

    df = long_df.copy()
    df["Resolution_z"] = (df["Resolution"] - df["Resolution"].mean()) / df["Resolution"].std()
    df["Stability_z"] = (df["Cosine_Sim"] - df["Cosine_Sim"].mean()) / df["Cosine_Sim"].std()
    df[outcome] = df[outcome].astype(int)

    unique_ids = df["image_id"].unique()
    n_images = len(unique_ids)
    grouped = {img_id: g for img_id, g in df.groupby("image_id")}

    # ------------------------------------------------------------
    # Resume from checkpoint, if one exists
    # ------------------------------------------------------------
    shrinkage_draws = []
    start_b = 0

    if checkpoint_file and os.path.isfile(checkpoint_file) and os.path.getsize(checkpoint_file) > 0:
        existing = pd.read_csv(checkpoint_file)
        if not existing.empty:
            start_b = int(existing["draw_index"].max()) + 1
            shrinkage_draws = existing["shrinkage_pct"].dropna().tolist()
            print(f"  Resuming {progress_label}: {start_b}/{n_boot} draws already attempted "
                  f"({len(shrinkage_draws)} usable so far).")

    checkpoint_fh = None
    if checkpoint_file:
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        write_header = start_b == 0
        checkpoint_fh = open(checkpoint_file, "a", buffering=1)  # line-buffered
        if write_header:
            checkpoint_fh.write("draw_index,shrinkage_pct\n")
            checkpoint_fh.flush()

    b = start_b - 1  # keeps b defined for the interrupt message if range() below is empty

    try:
        for b in range(start_b, n_boot):
            rng = np.random.default_rng(seed + b)
            sampled_ids = rng.choice(unique_ids, size=n_images, replace=True)

            pieces = []
            for new_id, orig_id in enumerate(sampled_ids):
                piece = grouped[orig_id].copy()
                piece["image_id"] = new_id  # re-key: avoids merging repeated draws into one cluster
                pieces.append(piece)

            boot_df = pd.concat(pieces, ignore_index=True)

            value = np.nan
            try:
                path_c = smf.gee(
                    f"{outcome} ~ Resolution_z", groups="image_id", data=boot_df,
                    family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable(),
                ).fit().params["Resolution_z"]

                path_cprime = smf.gee(
                    f"{outcome} ~ Stability_z + Resolution_z", groups="image_id", data=boot_df,
                    family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable(),
                ).fit().params["Resolution_z"]

                if path_c != 0:
                    value = (path_c - path_cprime) / path_c * 100
                    shrinkage_draws.append(value)

            except Exception:
                pass  # non-converging draw: recorded as blank below, excluded from the CI

            if checkpoint_fh:
                checkpoint_fh.write(f"{b},{'' if np.isnan(value) else value}\n")
                checkpoint_fh.flush()
                os.fsync(checkpoint_fh.fileno())

            print(f"  Bootstrap {b + 1}/{n_boot} complete")

    except KeyboardInterrupt:
        print(f"\n  Interrupted after draw {b + 1}/{n_boot} for {progress_label}.")
        print(f"  Progress saved to: {checkpoint_file}")
        print("  Re-run the script and it will pick up from here.")
        raise

    finally:
        if checkpoint_fh:
            checkpoint_fh.close()

    shrinkage_draws = np.array(shrinkage_draws)
    lower_pct = (1 - ci) / 2 * 100
    upper_pct = (1 + ci) / 2 * 100

    return {
        "Shrinkage_pct_CI_Lower": np.percentile(shrinkage_draws, lower_pct),
        "Shrinkage_pct_CI_Upper": np.percentile(shrinkage_draws, upper_pct),
        "Shrinkage_pct_n_boot_success": len(shrinkage_draws),
    }

def run_mediation_analysis(long_df):

    df = long_df.copy()
    df["Resolution_z"] = (df["Resolution"] - df["Resolution"].mean()) / df["Resolution"].std()
    df["Stability_z"] = (df["Cosine_Sim"] - df["Cosine_Sim"].mean()) / df["Cosine_Sim"].std()
    df["Top1"] = df["Top1"].astype(int)

    row = {"N": len(df)}
    row.update(path_a_resolution_predicts_stability(df))
    row.update(path_c_resolution_alone(df))
    row.update(path_b_and_cprime_full_model(df))

    c = row["PathC_resolution_coef"]
    cprime = row["PathCprime_resolution_coef"]
    row["Shrinkage_abs"] = c - cprime
    row["Shrinkage_pct"] = (c - cprime) / c * 100 if c != 0 else np.nan

    row["Mediation_pattern_consistent"] = bool(
        row["PathA_p"] < 0.05
        and row["PathB_stability_p"] < 0.05
        and row["Shrinkage_pct"] > 0
    )

    return pd.DataFrame([row])

def run_mcnemar_resnet_vs_vit(resnet_long_df, vit_long_df):

    resnet = resnet_long_df[["image_id", "Resolution", "Top1"]].copy()
    resnet["Top1"] = resnet["Top1"].astype(bool)
    resnet = resnet.rename(columns={"Top1": "ResNet18_Top1"})

    vit = vit_long_df[["image_id", "Resolution", "Top1"]].copy()
    vit["Top1"] = vit["Top1"].astype(bool)
    vit = vit.rename(columns={"Top1": "ViT_Top1"})

    merged = pd.merge(resnet, vit, on=["image_id", "Resolution"], how="inner")

    results = []

    for resolution in sorted(merged["Resolution"].unique()):
        subset = merged[merged["Resolution"] == resolution]
        r = subset["ResNet18_Top1"]
        v = subset["ViT_Top1"]

        a = ((r == True) & (v == True)).sum()
        b = ((r == True) & (v == False)).sum()
        c = ((r == False) & (v == True)).sum()
        d = ((r == False) & (v == False)).sum()

        test = mcnemar([[a, b], [c, d]], exact=True)

        resnet_acc = r.mean()
        vit_acc = v.mean()

        results.append({
            "Resolution": resolution,
            "N": len(subset),
            "Both_Correct": a,
            "ResNet_only": b,
            "ViT_only": c,
            "Both_Wrong": d,
            "ResNet18_Top1": resnet_acc,
            "ViT_Top1": vit_acc,
            "Difference_ViT_minus_ResNet": vit_acc - resnet_acc,
            "McNemar_p": test.pvalue,
        })

    results = pd.DataFrame(results)

    reject, corrected_p, _, _ = multipletests(results["McNemar_p"], method="holm")
    results["McNemar_p_Holm"] = corrected_p
    results["Significant_Holm"] = reject

    results["Better_Model"] = "Tie"
    results.loc[results["ViT_Top1"] > results["ResNet18_Top1"], "Better_Model"] = "ViT"
    results.loc[results["ResNet18_Top1"] > results["ViT_Top1"], "Better_Model"] = "ResNet18"

    return results

