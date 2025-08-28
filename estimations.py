"""Run basic causal estimators using DoWhy.

This script loads the raw EQLS data, prepares it using the existing
`structure_data` helper and estimates the average treatment effect (ATE)
with several estimation strategies implemented in DoWhy:

* Propensity score matching
* Propensity score weighting
* Propensity score stratification
* Linear regression adjustment

The treatment is ``Y11_Q57`` (perception of financial situation) and the
outcome is ``Y11_MWIndex`` (mental well-being index).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict

import networkx as nx
import pandas as pd
from dowhy import CausalModel

from structure_data import choose_columns, preprocess_data

ROOT = Path(__file__).resolve().parent
RAW_DATA = ROOT / "raw_data/csv/eqls_2007and2011.csv"
DICT_PATH = ROOT / "data/dictionary.json"
GRAPH_PATH = ROOT / "graphs/full_causal.gpickle"
TREATMENT = "Y11_Q57"
OUTCOME = "Y11_MWIndex"


def load_data() -> pd.DataFrame:
    """Load and preprocess the raw EQLS data."""
    df = choose_columns()
    df = preprocess_data(
        df,
        na_threshold=0.5,
        impute_strategy="drop",
        treatment_dichotomize_value="median",
        treatment_column=TREATMENT,
    )
    return df


def load_graph() -> nx.DiGraph:
    """Load the causal graph describing relationships among variables."""
    with open(GRAPH_PATH, "rb") as f:
        return pickle.load(f)


def estimate_effects(df: pd.DataFrame, graph: nx.DiGraph) -> Dict[str, float]:
    """Estimate ATE using several DoWhy estimators."""
    model = CausalModel(
        data=df,
        treatment=TREATMENT,
        outcome=OUTCOME,
        graph=nx.nx_pydot.to_pydot(graph).to_string(),
    )
    estimand = model.identify_effect()
    methods = [
        "backdoor.propensity_score_matching",
        "backdoor.propensity_score_weighting",
        "backdoor.propensity_score_stratification",
        "backdoor.linear_regression",
    ]
    results: Dict[str, float] = {}
    for m in methods:
        try:
            est = model.estimate_effect(estimand, method_name=m)
            results[m] = float(est.value)
        except Exception:
            results[m] = float("nan")
    return results


def main() -> None:
    df = load_data()
    graph = load_graph()
    results = estimate_effects(df, graph)
    print("Estimation results (ATE):")
    for name, val in results.items():
        print(f"  {name:<40} {val:.4f}")


if __name__ == "__main__":
    main()
