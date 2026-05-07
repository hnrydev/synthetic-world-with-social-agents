from __future__ import annotations

from experiment import run_paired_experiment, run_sensitivity_sweep
from inference import paired_ate, summarize_by_group
from plots import plot_dose_response, plot_effect_distribution


def main() -> None:
    df = run_paired_experiment(
        n_runs=500,
        n_agents=200,
        q_treatment=0.15,
        network_kind="small_world",
        epsilon=0.25,
        steps=300,
    )

    print("Main paired experiment")
    print(paired_ate(df, effect_column="effect_polarization"))
    plot_effect_distribution(df, effect_column="effect_polarization")

    sweep = run_sensitivity_sweep()
    summary = summarize_by_group(
        sweep,
        group_cols=["network_kind", "epsilon", "q_treatment"],
        effect_column="effect_polarization",
    )

    print("\nSensitivity summary")
    print(summary)
    plot_dose_response(summary)


if __name__ == "__main__":
    main()