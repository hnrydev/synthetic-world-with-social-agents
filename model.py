from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from agents import Agent


@dataclass
class ModelConfig:
    n_agents: int = 200
    epsilon: float = 0.25
    mu: float = 0.35
    q_diverse_exposure: float = 0.0
    steps: int = 300


class OpinionModel:
    def __init__(
        self,
        graph: nx.Graph,
        initial_opinions: np.ndarray,
        config: ModelConfig,
        seed: int,
        stubborn_ids: list[int] | None = None,
    ) -> None:
        self.graph = graph
        self.config = config
        self.rng = np.random.default_rng(seed)

        stubborn_set = set(stubborn_ids or [])

        self.agents: list[Agent] = [
            Agent(
                id=i,
                opinion=float(initial_opinions[i]),
                stubborn=(i in stubborn_set),
                kind="partisan" if i in stubborn_set else "citizen",
            )
            for i in range(config.n_agents)
        ]

        self.history: list[dict[str, float]] = []

    def step(self) -> None:
        i = int(self.rng.integers(0, self.config.n_agents))
        partner_id = self._choose_interaction_partner(i)

        if partner_id is None:
            self._record()
            return

        self.agents[i].interact_with(
            self.agents[partner_id],
            epsilon=self.config.epsilon,
            mu=self.config.mu,
        )

        self._record()

    def run(self) -> np.ndarray:
        self._record()

        for _ in range(self.config.steps):
            self.step()

        return self.opinions_array()

    def opinions_array(self) -> np.ndarray:
        return np.array([a.opinion for a in self.agents], dtype=float)

    def _choose_interaction_partner(self, i: int) -> int | None:
        use_diverse_exposure = self.rng.random() < self.config.q_diverse_exposure

        if use_diverse_exposure:
            candidates = [j for j in range(self.config.n_agents) if j != i]
        else:
            candidates = list(self.graph.neighbors(i))

        if not candidates:
            return None

        return int(self.rng.choice(candidates))

    def _record(self) -> None:
        opinions = self.opinions_array()
        self.history.append(
            {
                "polarization": polarization(opinions),
                "fragmentation": fragmentation(opinions),
                "mean_opinion": float(np.mean(opinions)),
            }
        )


def polarization(opinions: np.ndarray) -> float:
    return float(np.var(opinions))


def fragmentation(opinions: np.ndarray, threshold: float = 0.08) -> int:
    sorted_opinions = np.sort(opinions)
    clusters = 1

    for left, right in zip(sorted_opinions[:-1], sorted_opinions[1:]):
        if right - left > threshold:
            clusters += 1

    return clusters