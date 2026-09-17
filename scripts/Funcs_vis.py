"""
===============================================================================
STATISTICAL RESULTS VISUALIZATION FUNCTIONS
===============================================================================

Purpose:
    Provides functions for generating publication-ready figures that visualize
    embedding stability, retrieval performance, effect sizes, GEE odds ratios,
    mediation effects, and analysis-of-influence diagnostics across datasets
    and models.

Functions:
    plot_metric_curve_multi()
        Plots a metric across image resolutions for multiple datasets and
        models, optionally including confidence intervals.

    plot_metric_heatmap_multi()
        Creates heatmaps showing metric values across models and resolutions
        for multiple datasets.

    plot_stability_vs_retrieval_multi()
        Plots embedding stability and Top-1 retrieval accuracy side by side
        across resolutions.

    plot_effect_sizes_multi()
        Plots paired model-comparison effect sizes across image resolutions.

    plot_gee_odds_ratios()
        Creates a forest plot of GEE odds ratios and 95% confidence intervals
        for embedding stability, grouped by dataset and model.

    plot_path_c_to_cprime()
        Visualizes the reduction in the resolution coefficient from the
        resolution-only model (Path C) to the model adjusted for stability
        (Path C′), including the percentage shrinkage.

    _load_cosine_for_aid()
        Loads and aggregates cosine-similarity data for analysis-of-influence
        plots, organized by model and resolution.

    _load_retrieval_for_aid()
        Loads retrieval results and calculates Top-1 accuracy and mean retrieval
        rank by model and resolution for analysis-of-influence plots.

    _plot_aid_metric()
        Plots a single analysis-of-influence metric across image resolutions.

    plot_aid_overview()
        Generates a combined overview of embedding stability, Top-1 retrieval
        accuracy, and mean retrieval rank across datasets and resolutions.

===============================================================================
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, NullFormatter
from matplotlib.lines import Line2D
import os

COLORS = {
    "ResNet18": "tab:blue",
    "VGG16": "tab:green",
    "ViT": "tab:red"
}


def plot_metric_curve_multi(data_by_dataset, metric_name, ylabel, title, save_path, show_ci=True):
    dataset_names = list(data_by_dataset.keys())
    n = len(dataset_names)

    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5), squeeze=False)
    axes = axes[0]

    for ax, dataset_name in zip(axes, dataset_names):
        for model_name, df in data_by_dataset[dataset_name].items():
            res_col = "resolution" if "resolution" in df.columns else "Resolution"
            df = df.sort_values(res_col, ascending=False)
            color = COLORS.get(model_name)

            ax.plot(df[res_col], df[metric_name], marker="o", linewidth=2, color=color, label=model_name)

            if show_ci:
                ax.fill_between(df[res_col], df["ci_lower"], df["ci_upper"], color=color, alpha=0.18)

        ax.set_xlabel("Image Resolution (pixels)")
        ax.set_ylabel(ylabel)
        ax.set_title(dataset_name)
        ax.invert_xaxis()
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=1000)
    plt.close()

def plot_metric_heatmap_multi(df_by_dataset, metric_name, title, save_path, cmap="viridis", annotate=True):
    dataset_names = list(df_by_dataset.keys())
    n = len(dataset_names)

    fig, axes = plt.subplots(n, 1, figsize=(10, 3.8 * n), squeeze=False)
    axes = axes[:, 0]

    for ax, dataset_name in zip(axes, dataset_names):
        df = df_by_dataset[dataset_name]

        heatmap = df.pivot(index="Model", columns="resolution", values=metric_name)
        heatmap = heatmap.reindex(sorted(heatmap.columns, reverse=True), axis=1)

        im = ax.imshow(heatmap.values, aspect="auto", cmap=cmap, interpolation="nearest")

        ax.set_xticks(np.arange(len(heatmap.columns)))
        ax.set_xticklabels(heatmap.columns)
        ax.set_yticks(np.arange(len(heatmap.index)))
        ax.set_yticklabels(heatmap.index)
        ax.set_xlabel("Image Resolution (pixels)")
        ax.set_ylabel("Model")
        ax.set_title(dataset_name)

        if annotate:
            for i in range(heatmap.shape[0]):
                for j in range(heatmap.shape[1]):
                    value = heatmap.iloc[i, j]
                    ax.text(j, i, f"{value:.3f}", ha="center", va="center",
                            fontsize=8, color="white" if value < heatmap.values.mean() else "black")

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric_name)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=1000, bbox_inches="tight")
    plt.close()

def plot_stability_vs_retrieval_multi(cosine_by_dataset, retrieval_by_dataset, save_path):
    dataset_names = list(cosine_by_dataset.keys())
    n = len(dataset_names)

    fig, axes = plt.subplots(n, 2, figsize=(13, 5 * n), squeeze=False)

    for row, dataset_name in enumerate(dataset_names):

        ax = axes[row, 0]
        for model, df in cosine_by_dataset[dataset_name].items():
            res_col = "resolution" if "resolution" in df.columns else "Resolution"
            df = df.sort_values(res_col, ascending=False)
            color = COLORS.get(model)
            ax.plot(df[res_col], df["mean"], marker="o", linewidth=2, color=color, label=model)
            ax.fill_between(df[res_col], df["ci_lower"], df["ci_upper"], alpha=0.18, color=color)
        ax.set_title(f"{dataset_name} — (A) Representation Stability")
        ax.set_xlabel("Resolution")
        ax.set_ylabel("Cosine Similarity")
        ax.invert_xaxis()
        ax.grid(alpha=.3)

        ax = axes[row, 1]
        for model, df in retrieval_by_dataset[dataset_name].items():
            res_col = "resolution" if "resolution" in df.columns else "Resolution"
            df = df.sort_values(res_col, ascending=False)
            color = COLORS.get(model)
            ax.plot(df[res_col], df["mean"], marker="o", linewidth=2, color=color, label=model)
            ax.fill_between(df[res_col], df["ci_lower"], df["ci_upper"], alpha=0.18, color=color)
        ax.set_title(f"{dataset_name} — (B) Retrieval Accuracy (Top-1)")
        ax.set_xlabel("Resolution")
        ax.set_ylabel("Top-1 Accuracy")
        ax.invert_xaxis()
        ax.grid(alpha=.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(save_path, dpi=1000, bbox_inches="tight")
    plt.close()

def plot_effect_sizes_multi(posthoc_by_dataset, title, save_path):
    comparison_colors = {
        "ResNet18 vs VGG16": "tab:blue",
        "ResNet18 vs ViT": "tab:red",
        "VGG16 vs ViT": "tab:green",
    }

    dataset_names = list(posthoc_by_dataset.keys())
    n = len(dataset_names)

    fig, axes = plt.subplots(1, n, figsize=(8 * n, 5), squeeze=False)
    axes = axes[0]

    for ax, dataset_name in zip(axes, dataset_names):
        posthoc_df = posthoc_by_dataset[dataset_name]

        for comparison in posthoc_df["Comparison"].unique():
            df = posthoc_df[posthoc_df["Comparison"] == comparison].sort_values("Resolution", ascending=False)
            ax.plot(df["Resolution"], df["Cohens_dz"], marker="o", linewidth=2,
                    label=comparison, color=comparison_colors.get(comparison))

        ax.axhline(0.2, color="gray", linestyle="--", linewidth=1)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.axhline(0.8, color="gray", linestyle="--", linewidth=1)

        ax.text(posthoc_df["Resolution"].max(), 0.22, "Small", fontsize=8, va="bottom")
        ax.text(posthoc_df["Resolution"].max(), 0.52, "Medium", fontsize=8, va="bottom")
        ax.text(posthoc_df["Resolution"].max(), 0.82, "Large", fontsize=8, va="bottom")

        ax.invert_xaxis()
        ax.grid(alpha=.3)
        ax.set_xlabel("Image Resolution")
        ax.set_ylabel("Cohen's $d_z$")
        ax.set_title(dataset_name)
        ax.legend()

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=1000, bbox_inches="tight")
    plt.close()

def plot_gee_odds_ratios(gee_full_df, dataset_order, model_order, save_path):

    df = gee_full_df[
        gee_full_df["Parameter"].astype(str).str.strip() == "Stability_z"
    ].copy()

    z = 1.959963984540054  # norm.ppf(0.975)
    df["Odds_Ratio"] = np.exp(df["coef"])
    df["OR_CI_Lower"] = np.exp(df["coef"] - z * df["std_err"])
    df["OR_CI_Upper"] = np.exp(df["coef"] + z * df["std_err"])

    model_colors = {"ResNet18": "#1f77b4", "VGG16": "#2ca02c", "ViT": "#d62728"}

    # ------------------------------------------------------------
    # Positions: same pattern as the original — models within a dataset
    # get consecutive integer y's counting down, with a one-row gap
    # between dataset blocks. E.g. for 2 datasets x 3 models:
    # ChestMNIST: 5, 4, 3 | gap | STL10: 1, 0, -1
    # ------------------------------------------------------------
    positions = {}
    block_tops = {}  # top y-position of each dataset's block, for labels
    y = (len(dataset_order) * len(model_order)) + (len(dataset_order) - 1) - 2

    for d_idx, dataset in enumerate(dataset_order):
        block_tops[dataset] = y
        for model in model_order:
            positions[(dataset, model)] = y
            y -= 1
        if d_idx < len(dataset_order) - 1:
            y -= 1  # extra gap row between dataset blocks

    top_y = block_tops[dataset_order[0]]
    bottom_y = min(positions.values())

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------
    total_span = top_y - bottom_y + 1
    fig, ax = plt.subplots(figsize=(7.5, 0.65 * total_span + 1))

    for _, row in df.iterrows():
        key = (row["Dataset"], row["Model"])
        if key not in positions:
            continue
        y = positions[key]
        or_value, ci_low, ci_high = row["Odds_Ratio"], row["OR_CI_Lower"], row["OR_CI_Upper"]
        color = model_colors.get(row["Model"], "gray")

        ax.errorbar(
            or_value, y,
            xerr=[[or_value - ci_low], [ci_high - or_value]],
            fmt="o", color=color, markersize=7, capsize=4,
            capthick=1.3, elinewidth=1.5, linewidth=1.5, zorder=3
        )

    ax.axvline(1, color="black", linestyle="--", linewidth=1.0, zorder=1)

    # Dataset group labels + separators
    for d_idx, dataset in enumerate(dataset_order):
        ax.text(
            0.03, block_tops[dataset] + 0.55, dataset,
            transform=ax.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=10.5, fontweight="bold"
        )
        if d_idx < len(dataset_order) - 1:
            separator_y = block_tops[dataset] - len(model_order)
            ax.axhline(separator_y, color="black", linewidth=0.7, alpha=0.35)

    # Y axis
    yticks = [positions[(d, m)] for d in dataset_order for m in model_order]
    yticklabels = [m for d in dataset_order for m in model_order]

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylabel("Model", fontsize=11)

    # X axis
    ax.set_xscale("log")
    x_min = max(1, df["OR_CI_Lower"].min() * 0.5)
    x_max = df["OR_CI_Upper"].max() * 1.5
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Odds Ratio", fontsize=11)

    # Explicit, human-readable tick values (not bare powers of 10)
    candidate_ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    xticks = [t for t in candidate_ticks if x_min <= t <= x_max]

    ax.set_xticks(xticks)
    ax.set_xticklabels([str(t) for t in xticks])

    ax.xaxis.set_minor_locator(LogLocator(base=10, subs="auto"))
    ax.xaxis.set_minor_formatter(NullFormatter())

    ax.grid(axis="x", which="major", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_ylim(bottom_y - 0.5, top_y + 0.8)

    # Compact OR/CI labels
    for _, row in df.iterrows():
        key = (row["Dataset"], row["Model"])
        if key not in positions:
            continue
        y = positions[key]
        or_value, ci_low, ci_high = row["Odds_Ratio"], row["OR_CI_Lower"], row["OR_CI_Upper"]

        ax.text(
            1.015, y, f"{or_value:.2f} ({ci_low:.2f}\u2013{ci_high:.2f})",
            transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8.5
        )

    ax.text(
        1.015, top_y + 0.75, "OR (95% CI)",
        transform=ax.get_yaxis_transform(), ha="left", va="bottom",
        fontsize=9, fontweight="bold"
    )

    fig.subplots_adjust(left=0.16, right=0.82, top=0.93, bottom=0.14)
    fig.savefig(save_path, dpi=1000, bbox_inches="tight")
    plt.close()

def plot_path_c_to_cprime(mediation_df, dataset_order, model_order, save_path):

    df = mediation_df.copy()
    df["Model"] = pd.Categorical(df["Model"], categories=model_order, ordered=True)
    df["Dataset"] = pd.Categorical(df["Dataset"], categories=dataset_order, ordered=True)
    df = df.sort_values(["Dataset", "Model"]).reset_index(drop=True)

    model_colors = {"ResNet18": "blue", "VGG16": "green", "ViT": "red"}

    fig, ax = plt.subplots(figsize=(10, 1.1 * len(df) + 2))

    y_positions = list(range(len(df)))

    for y, (_, row) in zip(y_positions, df.iterrows()):
        path_c = float(row["PathC_resolution_coef"])
        path_cprime = float(row["PathCprime_resolution_coef"])
        shrinkage_pct = (path_c - path_cprime) / path_c * 100 if path_c != 0 else np.nan
        color = model_colors.get(row["Model"], "gray")

        ax.plot([path_c, path_cprime], [y, y], color=color, linewidth=2.0, zorder=1)
        ax.scatter(path_c, y, s=85, color=color, marker="o", zorder=3)
        ax.scatter(path_cprime, y, s=85, facecolors="white", edgecolors=color, linewidths=2, marker="o", zorder=3)

        midpoint = (path_c + path_cprime) / 2
        ax.annotate(
            f"{shrinkage_pct:.1f}%", xy=(midpoint, y), xytext=(0, 10),
            textcoords="offset points", ha="center", va="bottom", fontsize=9
        )

    labels = [f"{row['Dataset']} \u2014 {row['Model']}" for _, row in df.iterrows()]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()

    ax.set_xlabel("Resolution coefficient", fontsize=11, labelpad=8)

    x_all = pd.concat([df["PathC_resolution_coef"], df["PathCprime_resolution_coef"]])
    ax.set_xlim(min(-0.5, x_all.min() - 0.5), x_all.max() + 0.8)

    n_models = len(model_order)
    for i in range(n_models, len(df), n_models):
        ax.axhline(i - 0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.set_axisbelow(True)

    legend_elements = [
        Line2D([0], [0], marker="o", color=model_colors[m], markerfacecolor=model_colors[m],
               markersize=7, linewidth=2, label=m)
        for m in model_order if m in model_colors
    ] + [
        Line2D([0], [0], marker="o", color="black", markerfacecolor="black",
               markersize=7, linestyle="None", label="Path C"),
        Line2D([0], [0], marker="o", color="black", markerfacecolor="white",
               markersize=7, linestyle="None", label="Path C\u2032"),
    ]

    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.1),
              ncol=len(legend_elements), frameon=False, fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.25, right=0.97, top=0.95, bottom=0.18)
    fig.savefig(save_path, dpi=1000, bbox_inches="tight")
    plt.close()

def _load_cosine_for_aid(path, model_order, max_resolution=None):
    df = pd.read_csv(path)
    resolution_cols = [c for c in df.columns if c not in ("Model", "label", "image_id")]

    for col in resolution_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    stability = df.groupby("Model")[resolution_cols].mean()
    stability.columns = [int(c) for c in resolution_cols]

    resolutions = sorted(stability.columns, reverse=True)
    if max_resolution is not None:
        resolutions = [r for r in resolutions if r <= max_resolution]

    stability = stability.reindex(model_order).dropna(how="all")[resolutions]
    return stability, resolutions

def _load_retrieval_for_aid(path, model_order, max_resolution=None):
    df = pd.read_csv(path)
    resolution_cols = [c for c in df.columns if c not in ("Model", "label", "image_id")]

    for col in resolution_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    top1, mean_rank = {}, {}

    for model in model_order:
        subset = df[df["Model"] == model]
        if subset.empty:
            continue
        top1[model] = (subset[resolution_cols] == 0).mean() * 100
        mean_rank[model] = subset[resolution_cols].mean() + 1

    top1_df = pd.DataFrame(top1).T
    mean_rank_df = pd.DataFrame(mean_rank).T

    top1_df.columns = [int(c) for c in resolution_cols]
    mean_rank_df.columns = [int(c) for c in resolution_cols]

    resolutions = sorted(top1_df.columns, reverse=True)
    if max_resolution is not None:
        resolutions = [r for r in resolutions if r <= max_resolution]

    return top1_df[resolutions], mean_rank_df[resolutions], resolutions

def _plot_aid_metric(ax, data, resolutions, title, ylabel, model_order, higher_is_better=True):
    for model in model_order:
        if model not in data.index:
            continue
        values = data.loc[model, resolutions]
        ax.plot(resolutions, values, marker="o", linewidth=2, label=model)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Resolution (px)")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xticks(resolutions)
    ax.set_xticklabels([str(r) for r in resolutions])
    ax.minorticks_off()
    ax.grid(True, alpha=0.25)
    ax.legend()

    direction = "Higher = better" if higher_is_better else "Lower = better"
    ax.text(0.02, 0.03, direction, transform=ax.transAxes, fontsize=9, alpha=0.7)

def plot_aid_overview(csv_dir, dataset_names, model_order, save_path, max_resolution=None):

    results = {}

    for dataset_name in dataset_names:
        cosine_path = os.path.join(csv_dir, f"{dataset_name}_cosine.csv")
        retrieval_path = os.path.join(csv_dir, f"{dataset_name}_retrieval.csv")

        stability, stability_res = _load_cosine_for_aid(cosine_path, model_order, max_resolution)
        top1, mean_rank, retrieval_res = _load_retrieval_for_aid(retrieval_path, model_order, max_resolution)

        if stability_res != retrieval_res:
            print(f"WARNING: stability/retrieval resolution mismatch for {dataset_name}")

        results[dataset_name] = {
            "stability": stability, "top1": top1, "mean_rank": mean_rank, "resolutions": retrieval_res
        }

    n = len(dataset_names)
    fig, axes = plt.subplots(3, n, figsize=(9 * n, 15), squeeze=False)

    for col, dataset_name in enumerate(dataset_names):
        r = results[dataset_name]
        _plot_aid_metric(axes[0, col], r["stability"], r["resolutions"],
                          f"{dataset_name} \u2014 Embedding Stability", "Mean cosine similarity", model_order, True)
        _plot_aid_metric(axes[1, col], r["top1"], r["resolutions"],
                          f"{dataset_name} \u2014 Top-1 Retrieval", "Top-1 accuracy (%)", model_order, True)
        _plot_aid_metric(axes[2, col], r["mean_rank"], r["resolutions"],
                          f"{dataset_name} \u2014 Mean Retrieval Rank", "Mean rank (1 = best)", model_order, False)

    suffix = f" (\u2264 {max_resolution}px)" if max_resolution else ""
    fig.suptitle(f"Stability and Retrieval Performance Across Resolution{suffix}",
                 fontsize=20, fontweight="bold")

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(save_path, dpi=1000, bbox_inches="tight")
    plt.close()