import numpy as np

from experiment import make_initial_opinions, run_paired_experiment
from model import fragmentation, polarization
from network import build_social_network


def test_network_has_expected_number_of_agents():
    graph = build_social_network(200, kind="small_world", seed=1)
    assert graph.number_of_nodes() == 200


def test_initial_opinions_are_valid_probabilities():
    opinions = make_initial_opinions(200, seed=1)
    assert np.all(opinions >= 0.0)
    assert np.all(opinions <= 1.0)


def test_polarization_is_nonnegative():
    opinions = np.array([0.1, 0.2, 0.8, 0.9])
    assert polarization(opinions) >= 0.0


def test_fragmentation_detects_multiple_clusters():
    opinions = np.array([0.1, 0.11, 0.8, 0.82])
    assert fragmentation(opinions, threshold=0.08) == 2


def test_paired_experiment_outputs_effect_columns():
    df = run_paired_experiment(n_runs=3, n_agents=50, steps=20)
    assert "effect_polarization" in df.columns
    assert "effect_fragmentation" in df.columns
    assert len(df) == 3