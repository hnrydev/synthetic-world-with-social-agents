from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from model import ModelConfig, OpinionModel, fragmentation, polarization
from network import build_social_network


def make_initial_opinions(
    n_agents: int,
    seed: int,
    initial_condition: str = "polarized",
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    if initial_condition == "uniform":
        return rng.uniform(0.0, 1.0, size=n_agents)

    if initial_condition == "polarized":
        left = rng.normal(0.25, 0.08, size=n_agents // 2)
        right = rng.normal(0.75, 0.08, size=n_agents - n_agents // 2)
        return np.clip(np.concatenate([left, right]), 0.0, 1.0)

    raise ValueError(f"Unknown initial condition: {initial_condition}")


def run_one_condition(
    graph: nx.Graph,
    initial_opinions: np.ndarray,
    q_diverse_exposure: float,
    seed: int,
    epsilon: float = 0.25,
    steps: int = 300,
) -> dict[str, float]:
    config = ModelConfig(
        n_agents=len(initial_opinions),
        epsilon=epsilon,
        q_diverse_exposure=q_diverse_exposure,
        steps=steps,
    )

    model = OpinionModel(
        graph=graph.copy(),
        initial_opinions=initial_opinions,
        config=config,
        seed=seed,
    )

    final_opinions = model.run()

    return {
        "final_polarization": polarization(final_opinions),
        "final_fragmentation": fragmentation(final_opinions),
        "final_mean_opinion": float(np.mean(final_opinions)),
    }


def run_paired_experiment(
    n_runs: int = 500,
    n_agents: int = 200,
    q_treatment: float = 0.15,
    network_kind: str = "small_world",
    epsilon: float = 0.25,
    steps: int = 300,
    initial_condition: str = "polarized",
) -> pd.DataFrame:
    rows = []

    for seed in range(n_runs):
        graph = build_social_network(n_agents, kind=network_kind, seed=seed)
        initial_opinions = make_initial_opinions(
            n_agents=n_agents,
            seed=seed,
            initial_condition=initial_condition,
        )

        control = run_one_condition(
            graph=graph,
            initial_opinions=initial_opinions,
            q_diverse_exposure=0.0,
            seed=seed + 10_000,
            epsilon=epsilon,
            steps=steps,
        )

        treated = run_one_condition(
            graph=graph,
            initial_opinions=initial_opinions,
            q_diverse_exposure=q_treatment,
            seed=seed + 10_000,
            epsilon=epsilon,
            steps=steps,
        )

        rows.append(
            {
                "seed": seed,
                "network_kind": network_kind,
                "epsilon": epsilon,
                "q_treatment": q_treatment,
                "control_polarization": control["final_polarization"],
                "treated_polarization": treated["final_polarization"],
                "effect_polarization": treated["final_polarization"]
                - control["final_polarization"],
                "control_fragmentation": control["final_fragmentation"],
                "treated_fragmentation": treated["final_fragmentation"],
                "effect_fragmentation": treated["final_fragmentation"]
                - control["final_fragmentation"],
            }
        )

    return pd.DataFrame(rows)


def run_sensitivity_sweep() -> pd.DataFrame:
    frames = []

    for network_kind in ["small_world", "scale_free", "random"]:
        for epsilon in [0.15, 0.25, 0.35]:
            for q in [0.05, 0.10, 0.20, 0.35]:
                df = run_paired_experiment(
                    n_runs=200,
                    q_treatment=q,
                    network_kind=network_kind,
                    epsilon=epsilon,
                )
                frames.append(df)

    return pd.concat(frames, ignore_index=True)