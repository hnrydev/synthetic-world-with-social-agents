from __future__ import annotations

import networkx as nx


def build_social_network(
    n_agents: int,
    kind: str = "small_world",
    seed: int | None = None,
) -> nx.Graph:
    if kind == "small_world":
        return nx.watts_strogatz_graph(n=n_agents, k=8, p=0.08, seed=seed)

    if kind == "scale_free":
        return nx.barabasi_albert_graph(n=n_agents, m=4, seed=seed)

    if kind == "random":
        return nx.erdos_renyi_graph(n=n_agents, p=0.04, seed=seed)

    raise ValueError(f"Invalid network type: {kind}")