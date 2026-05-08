"""
Experiment 003 - Bail test: critical epsilon for cross-partisan exposure.

Real-world anchor
-----------------
Bail et al. (2018, PNAS): exposing Twitter users to opposing views
increased political polarization. Counterintuitive result that any model
of opinion dynamics should be able to either reproduce or explain.

Hypothesis
----------
Under bounded-confidence dynamics, the diverse-exposure policy has a
critical confidence threshold (epsilon*) below which it does not reduce
polarization (and may increase it), and above which it does.

Estimand
--------
ATE of q = 0.20 on final polarization, as a function of epsilon, on a
scale-free network with polarized initial opinions.

Identification
--------------
Internal: paired counterfactual (same seed, same network, same init).
External: stylized; calibration to real data is a separate task.

Threats
-------
- Bounded confidence is one of many opinion-update rules.
- Real polarization includes affect and identity, not just variance.
- Network topology is stylized; real social media follow graphs differ.
- T = 300 may not be steady state.

Status
------
Confirmatory test of conditional-effect hypothesis. If the sign-flip is
not observed, the bounded-confidence model is incompatible with the
empirical result and should not be used to predict similar policies.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from experiment import run_paired_experiment
from experiments._utils import (
    make_experiment_dir,
    save_config,
    save_runs,
    save_summary,
)
from inference import paired_ate

EXPERIMENT_ID = "exp_003_bail_test"

CONFIG = {
    "experiment_id": EXPERIMENT_ID,
    "n_runs": 500,
    "n_agents": 200,
    "q_treatment": 0.20,
    "network_kind": "scale_free",
    "epsilons": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    "steps": 500,
    "initial_condition": "polarized",
    "anchor_paper": "Bail et al. (2018), PNAS",
}


def main() -> None:
    out_dir = make_experiment_dir(EXPERIMENT_ID)
    save_config(out_dir, CONFIG)

    all_runs = []
    summary_rows = []

    for epsilon in CONFIG["epsilons"]:
        df = run_paired_experiment(
            n_runs=CONFIG["n_runs"],
            n_agents=CONFIG["n_agents"],
            q_treatment=CONFIG["q_treatment"],
            network_kind=CONFIG["network_kind"],
            epsilon=epsilon,
            steps=CONFIG["steps"],
            initial_condition=CONFIG["initial_condition"],
        )
        df["epsilon"] = epsilon
        all_runs.append(df)

        ate = paired_ate(df, effect_column="effect_polarization")
        summary_rows.append({"epsilon": epsilon, **ate})

        print(f"epsilon={epsilon:.2f}  ATE={ate['ate']:+.6f}  "
              f"CI=[{ate['ci_low']:+.6f}, {ate['ci_high']:+.6f}]")

    save_runs(out_dir, pd.concat(all_runs, ignore_index=True))

    summary_df = pd.DataFrame(summary_rows)
    save_summary(out_dir, summary_df)

    plot_critical_epsilon(summary_df, out_dir / "plots" / "critical_epsilon.png")
    print(f"\nResults saved to {out_dir}")


def plot_critical_epsilon(summary: pd.DataFrame, out_path) -> None:
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        summary["epsilon"],
        summary["ate"],
        yerr=1.96 * summary["se"],
        marker="o",
        capsize=4,
        linewidth=2,
    )
    plt.axhline(0, color="black", linewidth=1, linestyle="--")
    plt.title("Cross-partisan exposure: ATE vs confidence threshold")
    plt.xlabel("Confidence threshold epsilon")
    plt.ylabel("ATE on final polarization")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()