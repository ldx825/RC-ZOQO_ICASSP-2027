from __future__ import annotations

import csv
import json
from pathlib import Path
import random
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "outputs" / "csv"
FIGURE_DIR = ROOT / "outputs" / "figures"
LOG_DIR = ROOT / "outputs" / "logs"


def ensure_output_dirs() -> None:
    for path in (CSV_DIR, FIGURE_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_config(name: str) -> dict:
    with (ROOT / "configs" / name).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, stem: str) -> None:
    ensure_output_dirs()
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row[column]) for column in columns) + " |" for row in rows]
    return "\n".join([header, divider, *body]) + "\n"

