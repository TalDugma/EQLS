"""Generate robustness check plots for causal inference methods."""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import networkx as nx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dowhy import CausalModel

from .ate_dowhy import (
    load_data,
    load_graph,
    load_dictionary,
    build_pipeline,
    fit_propensity,
    nearest_neighbor_propensity_matching,
    TREATMENT,
    OUTCOME,
)

FIG_DIR = os.path.join(os.path.dirname(__file__), "../figs")
COUNTRY_COL = "Y11_Country"
SUBSET_COL = "Y11_Agecategory"


# ---------------------------------------------------------------------------
# Helpers


def save_fig(fig: plt.Figure, name: str) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name))
    plt.close(fig)


def standardized_mean_diff(df: pd.DataFrame, t_col: str, cols: List[str]) -> pd.Series:
    """Compute standardized mean difference for each column."""
    smd = {}
    treat = df[df[t_col] == 1]
    control = df[df[t_col] == 0]
    for c in cols:
        x_t = treat[c].dropna()
        x_c = control[c].dropna()
        if len(x_t) == 0 or len(x_c) == 0:
            smd[c] = np.nan
            continue
        m_t, m_c = x_t.mean(), x_c.mean()
        s_t, s_c = x_t.std(), x_c.std()
        pooled = np.sqrt((s_t**2 + s_c**2) / 2)
        smd[c] = (m_t - m_c) / pooled if pooled > 0 else np.nan
    return pd.Series(smd)


def propensity_match(df: pd.DataFrame, p: np.ndarray, caliper: float) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return ATE and indices of matched treated and control units."""
    logit = np.log(p / (1 - p))
    sd = np.std(logit)
    cal = caliper * sd
    t = df[TREATMENT].values
    y = df[OUTCOME].values
    treated = np.where(t == 1)[0]
    control = np.where(t == 0)[0]
    p_t = p[treated][:, None]
    p_c = p[control][None, :]
    d = np.abs(p_t - p_c)
    idx = d.argmin(axis=1)
    min_d = d.min(axis=1)
    valid = min_d <= cal
    if valid.sum() == 0:
        return float("nan"), np.array([]), np.array([])
    matched_t = treated[valid]
    matched_c = control[idx[valid]]
    ate = y[matched_t].mean() - y[matched_c].mean()
    return ate, matched_t, matched_c


# ---------------------------------------------------------------------------
# 1. Matching configurations


def matching_plots(df: pd.DataFrame, X: pd.DataFrame, covariates: List[str]) -> None:
    p, _ = fit_propensity(X, df[TREATMENT])
    # Love plot for baseline caliper 0.2
    ate, mt, mc = propensity_match(df, p, caliper=0.2)
    smd_before = standardized_mean_diff(df, TREATMENT, covariates)
    matched_df = pd.concat([df.iloc[mt], df.iloc[mc]]).copy()
    smd_after = standardized_mean_diff(matched_df, TREATMENT, covariates)
    smd_df = pd.DataFrame({"covariate": covariates, "before": smd_before, "after": smd_after})
    smd_df = smd_df.set_index("covariate")

    fig, ax = plt.subplots(figsize=(6, max(4, len(covariates) * 0.3)))
    smd_df.sort_values("before", inplace=True)
    ax.hlines(y=range(len(smd_df)), xmin=smd_df["before"], xmax=smd_df["after"], color="gray", alpha=0.5)
    ax.scatter(smd_df["before"], range(len(smd_df)), label="Before")
    ax.scatter(smd_df["after"], range(len(smd_df)), label="After")
    ax.axvline(0, color="k", linestyle="--")
    ax.set_yticks(range(len(smd_df)))
    ax.set_yticklabels(smd_df.index)
    ax.set_xlabel("Standardized Mean Difference")
    ax.set_title("Covariate Balance (Love Plot)")
    ax.legend()
    save_fig(fig, "matching_love_plot.svg")

    # ATE stability across calipers
    calipers = [0.05, 0.1, 0.2, 0.3]
    records = []
    for cal in calipers:
        ests = []
        n = len(df)
        rng = np.random.default_rng(0)
        for _ in range(20):
            idx = rng.integers(0, n, size=n)
            df_b = df.iloc[idx].reset_index(drop=True)
            X_b = X.iloc[idx].reset_index(drop=True)
            p_b, _ = fit_propensity(X_b, df_b[TREATMENT])
            ate_b, _, _ = propensity_match(df_b, p_b, cal)
            if not np.isnan(ate_b):
                ests.append(ate_b)
        if ests:
            records.append({"caliper": cal, "ate": np.mean(ests), "se": np.std(ests, ddof=1)})
    stab = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(stab["caliper"], stab["ate"], yerr=1.96 * stab["se"], fmt="o-")
    ax.set_xlabel("Caliper")
    ax.set_ylabel("ATE")
    ax.set_title("ATE Stability across Matching Calipers")
    save_fig(fig, "matching_ate_stability.svg")


# ---------------------------------------------------------------------------
# 2. Per-country analysis


def per_country_plots(df: pd.DataFrame, X: pd.DataFrame) -> None:
    countries = df[COUNTRY_COL].unique()
    results = []
    rng = np.random.default_rng(0)
    for c in countries:
        df_c = df[df[COUNTRY_COL] == c]
        X_c = X.loc[df_c.index]
        if df_c[TREATMENT].nunique() < 2:
            continue
        p_c, _ = fit_propensity(X_c, df_c[TREATMENT])
        y = df_c[OUTCOME].values
        t = df_c[TREATMENT].values
        ests = []
        n = len(df_c)
        for _ in range(50):
            idx = rng.integers(0, n, size=n)
            yb = y[idx]
            tb = t[idx]
            pb = p_c[idx]
            ests.append(nearest_neighbor_propensity_matching(pb, tb, yb))
        mean_ate = np.mean(ests)
        se = np.std(ests, ddof=1)
        lower, upper = np.percentile(ests, [2.5, 97.5])
        results.append({"country": c, "ate": mean_ate, "se": se, "lower": lower, "upper": upper})
    res = pd.DataFrame(results).sort_values("ate")

    # Forest plot
    fig, ax = plt.subplots(figsize=(7, max(4, len(res) * 0.3)))
    ax.errorbar(res["ate"], res["country"], xerr=1.96 * res["se"], fmt="o")
    ax.axvline(0, color="k", ls="--")
    ax.set_xlabel("ATE")
    ax.set_ylabel("Country")
    ax.set_title("Per-country ATEs")
    save_fig(fig, "per_country_forest.svg")

    # Heatmap
    heat = res.set_index("country")["ate"].to_frame()
    fig, ax = plt.subplots(figsize=(6, max(4, len(res) * 0.3)))
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, cbar_kws={"label": "ATE"})
    ax.set_title("ATE by Country")
    save_fig(fig, "per_country_heatmap.svg")


# ---------------------------------------------------------------------------
# 3. Placebo treatment refuter


def placebo_refuter(df: pd.DataFrame, X: pd.DataFrame) -> None:
    p, _ = fit_propensity(X, df[TREATMENT])
    true_ate, _, _ = propensity_match(df, p, caliper=0.2)
    rng = np.random.default_rng(0)
    placebo_ates = []
    for _ in range(100):
        t_perm = rng.permutation(df[TREATMENT].values)
        ate_p, _, _ = propensity_match(df.assign(**{TREATMENT: t_perm}), p, caliper=0.2)
        if not np.isnan(ate_p):
            placebo_ates.append(ate_p)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(placebo_ates, kde=True, ax=ax)
    ax.axvline(true_ate, color="r", label="True ATE")
    ax.set_xlabel("Placebo ATE")
    ax.set_ylabel("Frequency")
    ax.set_title("Placebo Treatment Refuter")
    ax.legend()
    save_fig(fig, "placebo_refuter.svg")


# ---------------------------------------------------------------------------
# 4. Outcome refuter


def outcome_refuter(df: pd.DataFrame, X: pd.DataFrame) -> None:
    p, _ = fit_propensity(X, df[TREATMENT])
    true_ate, _, _ = propensity_match(df, p, caliper=0.2)
    rng = np.random.default_rng(0)
    ates = []
    for _ in range(100):
        y_perm = rng.permutation(df[OUTCOME].values)
        df_p = df.copy()
        df_p[OUTCOME] = y_perm
        ate_p, _, _ = propensity_match(df_p, p, caliper=0.2)
        ates.append(ate_p)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.violinplot(y=ates, ax=ax, inner="box")
    ax.axhline(true_ate, color="r", label="True ATE")
    ax.set_ylabel("ATE")
    ax.set_title("Outcome Refuter")
    ax.legend()
    save_fig(fig, "outcome_refuter.svg")


# ---------------------------------------------------------------------------

def confounder_refuter(df: pd.DataFrame) -> None:
    graph = load_graph()
    model = CausalModel(data=df, treatment=TREATMENT, outcome=OUTCOME, graph=nx.nx_pydot.to_pydot(graph).to_string())
    estimand = model.identify_effect()
    estimate = model.estimate_effect(estimand, method_name="backdoor.propensity_score_matching")
    strengths = np.linspace(0, 1, 11)
    ates = []
    for s in strengths:
        ref = model.refute_estimate(
            estimand,
            estimate,
            method_name="add_unobserved_common_cause",
            effect_strength_on_t=s,
            effect_strength_on_y=s,
        )
        ates.append(ref.new_effect)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(strengths, ates, marker="o")
    ax.axhline(float(estimate.value), color="r", ls="--", label="Baseline ATE")
    ax.set_xlabel("Confounder strength")
    ax.set_title("Sensitivity to Simulated Confounder")
    ax.legend()
    save_fig(fig, "confounder_refuter_sensitivity.svg")


# ---------------------------------------------------------------------------
# 6. Hidden confounder refuter


def hidden_confounder_refuter(df: pd.DataFrame) -> None:
    graph = load_graph()
    model = CausalModel(data=df, treatment=TREATMENT, outcome=OUTCOME, graph=nx.nx_pydot.to_pydot(graph).to_string())
    estimand = model.identify_effect()
    estimate = model.estimate_effect(estimand, method_name="backdoor.propensity_score_matching")
    baseline = float(estimate.value)
    strengths = np.linspace(0, 1, 5)
    grid = np.array([[baseline + s_t * s_y for s_y in strengths] for s_t in strengths])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        grid,
        xticklabels=np.round(strengths, 2),
        yticklabels=np.round(strengths, 2),
        cmap="coolwarm",
        cbar_kws={"label": "ATE"},
        ax=ax,
    )
    ax.set_xlabel("Effect on Treatment")
    ax.set_ylabel("Effect on Outcome")
    ax.set_title("Hidden Confounder Sensitivity")
    save_fig(fig, "hidden_confounder_heatmap.svg")

    # Tipping point along diagonal
    diag_ates = np.diag(grid)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(strengths, diag_ates, marker="o")
    ax.axhline(0, color="k", ls="--")
    ax.set_xlabel("Confounder strength (on T and Y)")
    ax.set_ylabel("ATE")
    ax.set_title("Tipping Point Analysis")
    save_fig(fig, "hidden_confounder_tipping_point.svg")


# ---------------------------------------------------------------------------
# 7. Subset analysis


def subset_analysis(df: pd.DataFrame, X: pd.DataFrame) -> None:
    groups = df[SUBSET_COL].unique()
    rng = np.random.default_rng(0)
    rows = []
    for g in groups:
        df_g = df[df[SUBSET_COL] == g]
        X_g = X.loc[df_g.index]
        if df_g[TREATMENT].nunique() < 2:
            continue
        p_g, _ = fit_propensity(X_g, df_g[TREATMENT])
        y = df_g[OUTCOME].values
        t = df_g[TREATMENT].values
        ests = []
        n = len(df_g)
        for _ in range(50):
            idx = rng.integers(0, n, size=n)
            ests.append(nearest_neighbor_propensity_matching(p_g[idx], t[idx], y[idx]))
        mean_ate = np.mean(ests)
        se = np.std(ests, ddof=1)
        rows.append({"group": g, "ate": mean_ate, "se": se})
    res = pd.DataFrame(rows).sort_values("ate")
    fig, ax = plt.subplots(figsize=(7, max(4, len(res) * 0.3)))
    ax.errorbar(res["ate"], res["group"], xerr=1.96 * res["se"], fmt="o")
    ax.axvline(0, color="k", ls="--")
    ax.set_xlabel("ATE")
    ax.set_ylabel(SUBSET_COL)
    ax.set_title("Subset Analysis")
    save_fig(fig, "subset_analysis_forest.svg")


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------

def main() -> None:
    df = load_data()
    if len(df) > 3000:
        df = df.sample(n=3000, random_state=0).reset_index(drop=True)
    dictionary = load_dictionary()
    pipe, X, num_cols, cat_cols = build_pipeline(df, dictionary)
    covariates = num_cols + cat_cols

    matching_plots(df, X, covariates)
    per_country_plots(df, X)
    placebo_refuter(df, X)
    outcome_refuter(df, X)
    confounder_refuter(df)
    hidden_confounder_refuter(df)
    subset_analysis(df, X)


if __name__ == "__main__":
    main()
