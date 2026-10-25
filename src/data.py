"""Loading Twin-2K-500 from a pinned local snapshot.

All paths are relative to the repo root. Run `python -m scripts.download_data`
first; it fetches the exact Hugging Face revision recorded in DATASET_REVISION.
"""
from __future__ import annotations

import glob
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "LLM-Digital-Twin/Twin-2K-500"
DATASET_REVISION = "f883165a3026fde855dfd448e0cd16443ab257b6"
RAW_DIR = REPO_ROOT / "data" / "raw" / "twin2k500"
CATALOG_DIR = RAW_DIR / "question_catalog_and_human_response_csv"
SIM_DIR = RAW_DIR / "LLM_simulation_results"


@lru_cache(maxsize=None)
def load_catalog() -> list[dict]:
    with open(CATALOG_DIR / "question_catalog.json", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def load_responses(which: str, labels: bool = False) -> pd.DataFrame:
    """Human responses in catalog-column space, indexed by pid.

    which: "wave1_3" (persona columns + earlier answers to the wave-4 items)
           or "wave4" (wave-4 answers only).
    """
    if which not in ("wave1_3", "wave4"):
        raise ValueError(which)
    suffix = "_label" if labels else ""
    df = pd.read_csv(CATALOG_DIR / f"{which}_response{suffix}.csv", low_memory=False)
    return df.set_index("pid").sort_index()


@lru_cache(maxsize=None)
def load_wave_split() -> pd.DataFrame:
    files = sorted(glob.glob(str(RAW_DIR / "wave_split" / "chunks" / "*.parquet")))
    return pd.concat([pd.read_parquet(f) for f in files]).set_index("pid").sort_index()


@lru_cache(maxsize=None)
def load_full_persona() -> pd.DataFrame:
    files = sorted(glob.glob(str(RAW_DIR / "full_persona" / "chunks" / "*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files])
    df["pid"] = df["pid"].astype(int)  # stored as strings here, ints in wave_split
    return df.set_index("pid").sort_index()


def iter_questions(blocks_json: str):
    """Yield (block_name, question_dict) from a wave_split / full_persona JSON string."""
    blocks = json.loads(blocks_json)
    if isinstance(blocks, dict):
        blocks = [blocks]
    for block in blocks:
        for q in block.get("Questions", []):
            yield block.get("BlockName", ""), q


def load_formatted_mapping() -> pd.DataFrame:
    """Map from the paper's Qualtrics-style column names to catalog CSV columns."""
    with open(SIM_DIR / "wave4_formatted_to_catalog_mapping.json", encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))


def list_simulation_specs() -> list[str]:
    root = SIM_DIR / "llm_simulations_all_specifications"
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def load_simulation(spec: str, which: str = "llm_imputed") -> pd.DataFrame:
    """Load a precomputed simulation CSV and rename its columns to catalog columns.

    spec: a folder name under llm_simulations_all_specifications, or "default"
          for GPT4.1-mini-simulation-llm-vs-human.
    which: "llm_imputed", "wave1_3" or "wave4".
    """
    if spec == "default":
        d = SIM_DIR / "GPT4.1-mini-simulation-llm-vs-human"
    else:
        d = SIM_DIR / "llm_simulations_all_specifications" / spec / "csv_comparison" / "csv_formatted"
    # Row 1 holds question text, not data.
    df = pd.read_csv(d / f"responses_{which}_formatted.csv", skiprows=[1], low_memory=False)
    mapping = load_formatted_mapping()
    rename = dict(zip(mapping.formatted_column, mapping.catalog_csv_column))
    df = df.rename(columns={"TWIN_ID": "pid", **rename})
    keep = ["pid"] + [c for c in mapping.catalog_csv_column if c in df.columns]
    df = df[keep].apply(pd.to_numeric, errors="coerce")
    df["pid"] = df["pid"].astype(int)
    return df.set_index("pid").sort_index()
