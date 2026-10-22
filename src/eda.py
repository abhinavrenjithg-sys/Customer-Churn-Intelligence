"""Exploratory Data Analysis (EDA) module for Customer Churn Prediction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Aesthetic styling configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
COLOR_CHURN = "#E11D48"     # Rose / Vibrant Crimson
COLOR_RETAINED = "#2563EB"  # Deep Royal Blue
COLOR_PALETTE = [COLOR_RETAINED, COLOR_CHURN]
FIG_DPI = 300


def set_plot_aesthetics() -> None:
    """Configures global matplotlib/seaborn formatting parameters."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 1.0,
        "grid.color": "#F1F5F9",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "medium",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.autolayout": True,
    })


def plot_churn_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    """Generates and saves the overall target class distribution."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=FIG_DPI)
    counts = df["Churn"].value_counts()
    percentages = (df["Churn"].value_counts(normalize=True) * 100).round(1)

    labels = [f"Retained (No)\n{counts['No']:,} ({percentages['No']}%)",
              f"Churned (Yes)\n{counts['Yes']:,} ({percentages['Yes']}%)"]
    bars = ax.bar(["Retained", "Churned"], counts.values, color=[COLOR_RETAINED, COLOR_CHURN], width=0.55, edgecolor="none", alpha=0.9)

    for bar, pct, cnt in zip(bars, percentages.values, counts.values):
        height = bar.get_height()
        ax.annotate(f"{cnt:,}\n({pct}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10.5, fontweight="bold")

    ax.set_title("Customer Churn Distribution (Class Imbalance)", pad=15)
    ax.set_ylabel("Number of Customers")
    ax.set_ylim(0, max(counts.values) * 1.18)
    out_file = output_dir / "churn_distribution.png"
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_churn_by_contract(df: pd.DataFrame, output_dir: Path) -> Path:
    """Visualizes churn rate across contract categories."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=FIG_DPI)
    contract_churn = pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100
    contract_churn = contract_churn.reindex(["Month-to-month", "One year", "Two year"])

    x = np.arange(len(contract_churn))
    width = 0.35

    ax.bar(x - width/2, contract_churn["No"], width, label="Retained (No)", color=COLOR_RETAINED, alpha=0.85)
    ax.bar(x + width/2, contract_churn["Yes"], width, label="Churned (Yes)", color=COLOR_CHURN, alpha=0.85)

    for i in x:
        ax.text(i - width/2, contract_churn.loc[contract_churn.index[i], "No"] + 1.5,
                f"{contract_churn.loc[contract_churn.index[i], 'No']:.1f}%", ha="center", fontsize=9.5, fontweight="bold")
        ax.text(i + width/2, contract_churn.loc[contract_churn.index[i], "Yes"] + 1.5,
                f"{contract_churn.loc[contract_churn.index[i], 'Yes']:.1f}%", ha="center", fontsize=9.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(contract_churn.index)
    ax.set_title("Churn Rate by Contract Commitment Type", pad=15)
    ax.set_ylabel("Percentage within Contract Group (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=True, facecolor="white")

    out_file = output_dir / "churn_by_contract.png"
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_churn_by_tenure_cohorts(df: pd.DataFrame, output_dir: Path) -> Path:
    """Visualizes churn across tenure brackets (cohorts)."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=FIG_DPI)
    bins = [0, 12, 24, 36, 48, 60, 72]
    labels = ["0-12m", "13-24m", "25-36m", "37-48m", "49-60m", "61-72m"]
    df_copy = df.copy()
    df_copy["TenureCohort"] = pd.cut(df_copy["tenure"], bins=bins, labels=labels, include_lowest=True)

    crosstab = pd.crosstab(df_copy["TenureCohort"], df_copy["Churn"], normalize="index") * 100

    crosstab.plot(kind="bar", stacked=True, color=[COLOR_RETAINED, COLOR_CHURN], ax=ax, alpha=0.88, edgecolor="none", width=0.6)
    ax.set_title("Customer Retention & Churn by Tenure Cohorts", pad=15)
    ax.set_xlabel("Customer Tenure Range")
    ax.set_ylabel("Composition (%)")
    ax.set_ylim(0, 105)
    ax.legend(["Retained (No)", "Churned (Yes)"], frameon=True, facecolor="white", loc="upper right")
    plt.xticks(rotation=0)

    out_file = output_dir / "churn_by_tenure.png"
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_churn_by_internet_service(df: pd.DataFrame, output_dir: Path) -> Path:
    """Visualizes churn rate segmented by internet service technology."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=FIG_DPI)
    ct = pd.crosstab(df["InternetService"], df["Churn"], normalize="index") * 100

    x = np.arange(len(ct))
    width = 0.35
    ax.bar(x - width/2, ct["No"], width, label="Retained (No)", color=COLOR_RETAINED, alpha=0.85)
    ax.bar(x + width/2, ct["Yes"], width, label="Churned (Yes)", color=COLOR_CHURN, alpha=0.85)

    for i in x:
        ax.text(i - width/2, ct.iloc[i]["No"] + 1.5, f"{ct.iloc[i]['No']:.1f}%", ha="center", fontsize=9.5, fontweight="bold")
        ax.text(i + width/2, ct.iloc[i]["Yes"] + 1.5, f"{ct.iloc[i]['Yes']:.1f}%", ha="center", fontsize=9.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(ct.index)
    ax.set_title("Churn Rate by Internet Service Category", pad=15)
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=True, facecolor="white")

    out_file = output_dir / "churn_by_internet_service.png"
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_churn_by_payment_method(df: pd.DataFrame, output_dir: Path) -> Path:
    """Visualizes churn proportions by billing/payment method."""
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=FIG_DPI)
    ct = pd.crosstab(df["PaymentMethod"], df["Churn"], normalize="index") * 100
    ct = ct.sort_values(by="Yes", ascending=True)

    bars_no = ax.barh(ct.index, ct["No"], label="Retained (No)", color=COLOR_RETAINED, alpha=0.85, height=0.55)
    bars_yes = ax.barh(ct.index, ct["Yes"], left=ct["No"], label="Churned (Yes)", color=COLOR_CHURN, alpha=0.85, height=0.55)

    for i, (idx, row) in enumerate(ct.iterrows()):
        ax.text(row["No"] / 2, i, f"{row['No']:.1f}%", va="center", ha="center", color="white", fontweight="bold", fontsize=9)
        ax.text(row["No"] + (row["Yes"] / 2), i, f"{row['Yes']:.1f}%", va="center", ha="center", color="white", fontweight="bold", fontsize=9)

    ax.set_title("Churn Breakdown by Payment Method (100% Stacked)", pad=15)
    ax.set_xlabel("Proportion (%)")
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right", frameon=True, facecolor="white")

    out_file = output_dir / "churn_by_payment_method.png"
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_monthly_charges_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    """Visualizes kernel density estimation and boxplots for monthly charges."""
    fig, (ax_box, ax_kde) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [0.28, 0.72]}, dpi=FIG_DPI
    )

    sns.boxplot(data=df, x="MonthlyCharges", y="Churn", hue="Churn", palette=COLOR_PALETTE, ax=ax_box, legend=False)
    ax_box.set(xlabel="")
    ax_box.set_title("Monthly Charges ($) Distribution by Churn Status", pad=12)

    sns.kdeplot(data=df[df["Churn"] == "No"], x="MonthlyCharges", label="Retained (No)", fill=True, color=COLOR_RETAINED, alpha=0.35, ax=ax_kde)
    sns.kdeplot(data=df[df["Churn"] == "Yes"], x="MonthlyCharges", label="Churned (Yes)", fill=True, color=COLOR_CHURN, alpha=0.35, ax=ax_kde)

    ax_kde.set_xlabel("Monthly Charges (USD)")
    ax_kde.set_ylabel("Density")
    ax_kde.legend(frameon=True, facecolor="white")

    out_file = output_dir / "monthly_charges_by_churn.png"
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_tenure_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    """Visualizes customer tenure distribution by churn outcome."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=FIG_DPI)
    sns.histplot(data=df, x="tenure", hue="Churn", multiple="dodge", bins=24, palette=COLOR_PALETTE, shrink=0.85, alpha=0.85, ax=ax)
    ax.set_title("Customer Tenure Distribution (Months) by Churn Status", pad=15)
    ax.set_xlabel("Tenure (Months)")
    ax.set_ylabel("Customer Count")
    ax.legend(["Churned (Yes)", "Retained (No)"], frameon=True, facecolor="white")

    out_file = output_dir / "tenure_distribution_by_churn.png"
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_correlation_matrix(df: pd.DataFrame, output_dir: Path) -> Path:
    """Computes and visualizes Pearson correlation matrix for numerical features."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=FIG_DPI)
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    corr_df = df[num_cols].copy()
    if "Churn" in df.columns:
        corr_df["Churn_Binary"] = (df["Churn"] == "Yes").astype(int)

    corr = corr_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.7,
        cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
        ax=ax,
    )
    ax.set_title("Numerical Features Correlation Matrix", pad=15)

    out_file = output_dir / "numerical_correlation_matrix.png"
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file


def generate_all_eda_figures(df: pd.DataFrame, output_dir: Optional[Path | str] = None) -> list[Path]:
    """Generates and persists all required EDA charts.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset loaded via data_loader.
    output_dir : Optional[Path | str]
        Target directory to save figures. Defaults to `reports/figures/`.

    Returns
    -------
    list[Path]
        Paths of generated figure files.
    """
    set_plot_aesthetics()
    if output_dir is None:
        target_dir = Path(__file__).resolve().parent.parent / "reports" / "figures"
    else:
        target_dir = Path(output_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating EDA figures in %s", target_dir)

    figure_paths = [
        plot_churn_distribution(df, target_dir),
        plot_churn_by_contract(df, target_dir),
        plot_churn_by_tenure_cohorts(df, target_dir),
        plot_churn_by_internet_service(df, target_dir),
        plot_churn_by_payment_method(df, target_dir),
        plot_monthly_charges_distribution(df, target_dir),
        plot_tenure_distribution(df, target_dir),
        plot_correlation_matrix(df, target_dir),
    ]

    logger.info("Successfully generated %d EDA figures.", len(figure_paths))
    return figure_paths
