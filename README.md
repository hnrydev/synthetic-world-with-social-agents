# Synthetic World with Social Agents

A small agent-based model (ABM) for testing social and political policies under
**paired counterfactual experiments**. The simulator pairs a control run with a
treated run that share seed, network, and initial opinions, isolating the policy
as the only source of variation.

## What it does

- Builds a social network of N agents (small-world, scale-free, or random).
- Each agent holds a continuous opinion in [0, 1].
- Opinions update by **bounded-confidence dynamics** (Hegselmann-Krause /
  Deffuant family): if two interacting agents are within `epsilon` of each
  other, both move toward the mean.
- A **policy lever** (`q_diverse_exposure`) replaces a fraction of network
  interactions with random non-neighbor exposure (a stylized media or
  recommender intervention).
- Runs paired control vs treated trials across many seeds and estimates the
  average treatment effect (ATE) on macro outcomes (polarization,
  fragmentation).

## Why paired runs

Because we control the data-generating process, running with and without the
policy on the same seed gives a clean within-pair counterfactual. The ATE is
identified by design **inside the model**. External validity is a separate
question and depends on how well the network, opinion distribution, and
parameters are calibrated to a real setting.

## Project layout


abc/
  network.py        # network builders (Watts-Strogatz, BA, ER)
  model.py          # OpinionModel + outcome metrics
  experiment.py    # paired-run experiment + sensitivity sweeps
  inference.py     # ATE, SEs, group summaries
  plots.py         # effect distribution and dose-response
  run.py           # CLI entry point
  tests/
    test_models.py


## Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install numpy pandas networkx matplotlib pytest