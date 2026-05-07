from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Agent:
    """A single social actor with a continuous opinion in [0, 1].

    Attributes
    ----------
    id : int
        Stable identifier matching the agent's node id in the network graph.
    opinion : float
        Current opinion, clipped to [0, 1].
    stubborn : bool
        If True, the agent never updates. Useful for media nodes, partisans,
        or any "anchor" intervention.
    kind : str
        Free-form tag for subgroup analysis ("citizen", "media", "partisan").
    """

    id: int
    opinion: float
    stubborn: bool = False
    kind: str = "citizen"

    def interact_with(
        self,
        other: "Agent",
        epsilon: float,
        mu: float,
    ) -> None:
        """Bounded-confidence update.

        If the two agents are within `epsilon` of each other, both move
        toward the mean by step size `mu`. Stubborn agents do not move,
        but they can still pull non-stubborn neighbors.
        """
        if abs(self.opinion - other.opinion) > epsilon:
            return

        midpoint_pull_self = mu * (other.opinion - self.opinion)
        midpoint_pull_other = mu * (self.opinion - other.opinion)

        if not self.stubborn:
            self.opinion = _clip01(self.opinion + midpoint_pull_self)

        if not other.stubborn:
            other.opinion = _clip01(other.opinion + midpoint_pull_other)


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x