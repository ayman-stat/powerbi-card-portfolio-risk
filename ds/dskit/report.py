"""Small reporting helpers: persist metrics for READMEs and model cards, print tables as markdown."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return round(float(o), 6)
    if isinstance(o, np.integer):
        return int(o)
    return o


def update_metrics(path: str, section: str, values: dict) -> dict:
    """Merge one notebook's headline numbers into reports/metrics.json (each notebook owns its own section)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    data[section] = _clean(values)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return data


def md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Markdown table (for pasting into the model card)."""
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: f"{v:{floatfmt}}" if pd.notna(v) else "")
    cols = [d.index.name or ""] + list(map(str, d.columns))
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in d.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row.values]) + " |")
    return "\n".join(lines)
