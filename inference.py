from __future__ import annotations

import math

import pandas as pd


def paired_ate(
    df: pd.DataFrame,
    effect_column: str = "effect_polarization",
) -> dict[str, float]:
    effects = df[effect_column]
    n = len(effects)
    ate = float(effects.mean())
    se = float(effects.std(ddof=1) / math.sqrt(n))
    ci_low = ate - 1.96 * se
    ci_high = ate + 1.96 * se

    return {
        "n": n,
        "ate": ate,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def summarize_by_group(
    df: pd.DataFrame,
    group_cols: list[str],
    effect_column: str = "effect_polarization",
) -> pd.DataFrame:
    rows = []

    for keys, group in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)

        stats = paired_ate(group, effect_column=effect_column)
        row = dict(zip(group_cols, keys))
        row.update(stats)
        rows.append(row)

    return pd.DataFrame(rows)