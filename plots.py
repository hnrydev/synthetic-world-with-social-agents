from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_effect_distribution(
    df: pd.DataFrame,
    effect_column: str = "effect_polarization",
) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(df[effect_column], bins=40, edgecolor="black")
    plt.axvline(df[effect_column].mean(), color="red", linewidth=2)
    plt.title("Distribution of paired treatment effects")
    plt.xlabel(effect_column)
    plt.ylabel("Run count")
    plt.tight_layout()
    plt.show()


def plot_dose_response(summary: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))

    for network_kind, group in summary.groupby("network_kind"):
        ordered = group.sort_values("q_treatment")
        plt.errorbar(
            ordered["q_treatment"],
            ordered["ate"],
            yerr=1.96 * ordered["se"],
            marker="o",
            capsize=3,
            label=network_kind,
        )

    plt.axhline(0, color="black", linewidth=1)
    plt.title("Dose-response: diverse exposure policy")
    plt.xlabel("Diverse exposure probability q")
    plt.ylabel("ATE on final polarization")
    plt.legend()
    plt.tight_layout()
    plt.show()