from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"


def make_experiment_dir(experiment_id: str) -> Path:
    out = RESULTS_ROOT / experiment_id
    (out / "plots").mkdir(parents=True, exist_ok=True)
    return out


def save_config(out_dir: Path, config: dict) -> None:
    config_with_meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **config,
    }
    (out_dir / "config.json").write_text(
        json.dumps(config_with_meta, indent=2, default=str)
    )


def save_runs(out_dir: Path, df: pd.DataFrame) -> None:
    df.to_csv(out_dir / "runs.csv", index=False)


def save_summary(out_dir: Path, summary: dict | pd.DataFrame) -> None:
    if isinstance(summary, dict):
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
    else:
        summary.to_csv(out_dir / "summary.csv", index=False)