"""
role_normalisation.py

Reads cleaned CSVs produced by normalisation.py and uses the Claude API to
map inconsistent free-text role/title strings to canonical job role names
and predict the industry sector — both in a single API call per batch.

New columns written per source column:
    llm_<col>_normalized             – canonical role name (or "Any" / null)
    llm_<col>_sector                 – industry sector (e.g. "Manufacturing", "IT")
    llm_<col>_collar                 – work type: "White Collar", "Blue Collar", "Grey Collar", or null
    llm_<col>_blue_collar_equivalent – related blue collar role in the same domain
    llm_<col>_grey_collar_equivalent – related grey collar role in the same domain
    llm_<col>_white_collar_equivalent– related white collar role in the same domain

Columns processed per CSV type:
    profile_clean.csv         -> role
    job_posting_clean.csv     -> role, title
    job_application_clean.csv -> job_role, job_title

The key optimisation: all unique values are collected across every dataset
first, normalised in one pass (deduplication avoids redundant API calls),
then the lookup is applied to each file.

Run after normalisation.py:
    python preprocessing/role_normalisation.py
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
import pandas as pd

# ── API key ────────────────────────────────────────────────────────────────────

def _load_api_key() -> str:
    raw = os.environ.get("CLAUDE_API_KEY", "")
    if not raw:
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("CLAUDE_API_KEY="):
                    raw = line.split("=", 1)[1]
                    break
    match = re.search(r"sk-ant-[A-Za-z0-9\-_]+", raw)
    if match:
        return match.group(0)
    raise ValueError(
        "CLAUDE_API_KEY not found or malformed. Set it in the .env file or as "
        "an environment variable."
    )

CLAUDE_API_KEY = _load_api_key()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "processed_data"

# ── Column config ──────────────────────────────────────────────────────────────
# Maps CSV filename → list of source columns to normalize.
SHEET_ROLE_COLS: dict[str, list[str]] = {
    "profile_clean.csv":         ["role"],
    "job_posting_clean.csv":     ["role", "title", "job_title"],
    "job_application_clean.csv": ["job_role", "job_title"],
}

_BATCH_SIZE  = 50
_CONCURRENCY = 3

# Values that are truly empty/invalid — skip the API for these only.
_EMPTY_VALUES = {"", "none", "n/a", "na", "null"}


# ── LLM batch normalisation ────────────────────────────────────────────────────

def _normalize_batch(
    batch_num: int,
    raw_values: list[str],
) -> tuple[int, list[dict]]:
    """
    Ask Claude to map each raw role string to a canonical role name and sector.
    Returns (batch_num, list of {"role": ..., "sector": ...} dicts).
    """
    numbered = "\n".join(
        f"{i + 1}. {v}" for i, v in enumerate(raw_values)
    )

    prompt = (
        "You are an expert in Indian labour market job roles.\n"
        "For each numbered raw role/title string, return:\n"
        "  - role: a clean, canonical job role name\n"
        "  - sector: the industry sector it belongs to\n"
        "  - collar: the work type classification\n"
        "  - blue_collar_equivalent: a related blue collar role in the same domain\n"
        "  - grey_collar_equivalent: a related grey collar role in the same domain\n"
        "  - white_collar_equivalent: a related white collar role in the same domain\n"
        "\n"
        "Rules for role:\n"
        "  - Use standard, concise English job titles (e.g. 'CNC Operator', 'Electrician', 'Accountant')\n"
        "  - If multiple roles are listed, pick the primary one\n"
        "  - If the input means the person is open to any role (e.g. 'Any', 'Anything', 'Any job', 'Open'), use \"Any\"\n"
        "  - If the input is 'ITI Other', 'ITI - Other', or any variation of ITI with 'other/misc/unspecified', use \"ITI Technician\"\n"
        "  - If the input is truly nonsensical or uninterpretable, use null\n"
        "  - Preserve meaningful specialisations (e.g. 'Diesel Mechanic' not just 'Mechanic')\n"
        "  - Use Title Case\n"
        "\n"
        "Rules for sector:\n"
        "  - Use one of these standard sectors:\n"
        "    Manufacturing, IT & Technology, Finance & Banking, Healthcare,\n"
        "    Education & Training, Retail & Sales, Construction & Real Estate,\n"
        "    Agriculture, Logistics & Transport, Hospitality & Tourism,\n"
        "    Government & Public Sector, Media & Entertainment, Telecommunications,\n"
        "    Energy & Utilities, Automotive, Textiles & Apparel, Any, Other\n"
        "  - If role is 'Any', sector should also be 'Any'\n"
        "  - If role is null, sector should be null\n"
        "\n"
        "Rules for collar:\n"
        "  - Use exactly one of: \"White Collar\", \"Blue Collar\", \"Grey Collar\", \"Any\", or null\n"
        "  - White Collar: knowledge/office/professional work (e.g. Accountant, Software Engineer, Manager)\n"
        "  - Blue Collar: manual/trade/physical labour (e.g. Welder, Electrician, CNC Operator, Factory Worker)\n"
        "  - Grey Collar: skilled technical roles bridging both (e.g. Technician, Nurse, Supervisor, Driver)\n"
        "  - If role is 'Any', collar should be 'Any'\n"
        "  - If role is null, collar should be null\n"
        "\n"
        "Rules for collar equivalents:\n"
        "  - Suggest a realistic related role in the SAME domain/sector at each collar level\n"
        "  - The equivalent for the role's own collar type should be the role itself (or very close)\n"
        "  - Examples for 'CNC Operator' (Blue Collar, Manufacturing):\n"
        "      blue_collar_equivalent:  'Machine Operator'\n"
        "      grey_collar_equivalent:  'CNC Programmer'\n"
        "      white_collar_equivalent: 'Manufacturing Engineer'\n"
        "  - Examples for 'Accountant' (White Collar, Finance):\n"
        "      blue_collar_equivalent:  'Data Entry Operator'\n"
        "      grey_collar_equivalent:  'Accounts Executive'\n"
        "      white_collar_equivalent: 'Chartered Accountant'\n"
        "  - If role is 'Any' or null, all three equivalents should also be 'Any' or null\n"
        "  - Use Title Case\n"
        "\n"
        "Reply with ONLY a JSON array, one object per entry, in the same order.\n"
        "Example: [{\"role\": \"CNC Operator\", \"sector\": \"Manufacturing\", \"collar\": \"Blue Collar\", "
        "\"blue_collar_equivalent\": \"Machine Operator\", \"grey_collar_equivalent\": \"CNC Programmer\", "
        "\"white_collar_equivalent\": \"Manufacturing Engineer\"}]\n"
        "\n"
        f"Roles:\n{numbered}"
    )

    try:
        import httpx
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": CLAUDE_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": _BATCH_SIZE * 150,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60.0,
        )
        response.raise_for_status()
        raw = response.json()["content"][0]["text"].strip()

        if raw.startswith("```"):
            raw = re.sub(r"^```[^\n]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)

        results: list = json.loads(raw)

        while len(results) < len(raw_values):
            results.append({"role": None, "sector": None, "collar": None, "blue_collar_equivalent": None, "grey_collar_equivalent": None, "white_collar_equivalent": None})

        return batch_num, results[: len(raw_values)]

    except Exception as exc:
        print(f"      [!] Batch {batch_num} failed: {type(exc).__name__}: {exc!r}", flush=True)
        return batch_num, [None] * len(raw_values)


_NULL_ENTRY: dict = {
    "role": None, "sector": None, "collar": None,
    "blue_collar_equivalent": None, "grey_collar_equivalent": None,
    "white_collar_equivalent": None,
}


def _build_lookup(unique_values: list[str]) -> dict[str, dict]:
    """
    Normalize all unique role strings in parallel batches.
    Returns a dict mapping raw value -> normalised fields.
    """
    lookup: dict[str, dict] = {}

    new_values = [v for v in unique_values if v.lower().strip() not in _EMPTY_VALUES]
    empty = {v for v in unique_values if v.lower().strip() in _EMPTY_VALUES}

    for v in empty:
        lookup[v] = _NULL_ENTRY

    print(f"    Unique values  : {len(unique_values)} total  "
          f"| {len(empty)} empty->null  "
          f"| {len(new_values)} new (API)")

    if not new_values:
        return lookup

    batches = [
        new_values[i: i + _BATCH_SIZE]
        for i in range(0, len(new_values), _BATCH_SIZE)
    ]
    print(f"    Batches        : {len(batches)} "
          f"(size {_BATCH_SIZE}, concurrency {_CONCURRENCY})")

    results_by_batch = []
    with ThreadPoolExecutor(max_workers=_CONCURRENCY) as executor:
        futures = {
            executor.submit(_normalize_batch, i, batch): i
            for i, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            results_by_batch.append(future.result())

    results_by_batch = sorted(results_by_batch, key=lambda x: x[0])

    done = 0
    for batch_num, predictions in results_by_batch:
        batch = batches[batch_num]
        for raw, pred in zip(batch, predictions):
            if pred is None:
                continue
            if not isinstance(pred, dict):
                pred = _NULL_ENTRY
            lookup[raw] = {
                k: pred.get(k) if isinstance(pred.get(k), str) else None
                for k in ("role", "sector", "collar",
                          "blue_collar_equivalent", "grey_collar_equivalent",
                          "white_collar_equivalent")
            }
        done += len(batch)
        print(f"      ... {done}/{len(new_values)} normalized")

    return lookup


# ── Per-column enrichment ──────────────────────────────────────────────────────

def _collect_unique_values(
    dataset_dirs: list[Path],
    csv_name: str,
    source_cols: list[str],
) -> set[str]:
    """Gather all non-empty unique values for the given columns across all dataset dirs."""
    values: set[str] = set()
    for d in dataset_dirs:
        path = d / csv_name
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        for col in source_cols:
            if col not in df.columns:
                continue
            values.update(
                v.strip()
                for v in df[col].dropna().astype(str)
                if v.strip()
            )
    return values


_OUTPUT_FIELDS = (
    ("normalized",             "role"),
    ("sector",                 "sector"),
    ("collar",                 "collar"),
    ("blue_collar_equivalent", "blue_collar_equivalent"),
    ("grey_collar_equivalent", "grey_collar_equivalent"),
    ("white_collar_equivalent","white_collar_equivalent"),
)

def _apply_lookup(
    df: pd.DataFrame,
    source_cols: list[str],
    lookup: dict[str, dict],
) -> pd.DataFrame:
    """Add all llm_<col>_* output columns using the prebuilt lookup."""
    df = df.copy()
    for col in source_cols:
        if col not in df.columns:
            for suffix, _ in _OUTPUT_FIELDS:
                df[f"llm_{col}_{suffix}"] = None
            continue
        raw_series = df[col].fillna("").astype(str).str.strip()
        for suffix, key in _OUTPUT_FIELDS:
            df[f"llm_{col}_{suffix}"] = raw_series.map(
                lambda v, k=key: lookup.get(v, {}).get(k)
            )
    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    dataset_dirs = sorted(d for d in PROCESSED_DIR.iterdir() if d.is_dir())

    if not dataset_dirs:
        print(f"No dataset folders found under {PROCESSED_DIR}. "
              "Run normalisation.py first.")
        return

    for csv_name, source_cols in SHEET_ROLE_COLS.items():
        print(f"\n=== {csv_name} ({', '.join(source_cols)}) ===")

        unique_values = _collect_unique_values(dataset_dirs, csv_name, source_cols)
        if not unique_values:
            print("  No values found -- skipping.")
            continue

        lookup = _build_lookup(sorted(unique_values))

        for dataset_dir in dataset_dirs:
            path = dataset_dir / csv_name
            if not path.exists():
                continue
            df = pd.read_csv(path, low_memory=False)
            df = _apply_lookup(df, source_cols, lookup)
            if csv_name == "profile_clean.csv" and "llm_role_normalized" in df.columns:
                df["normalised_role"] = df["llm_role_normalized"]
            df.to_csv(path, index=False)
            print(f"  Saved -> {path}")

    print("\nRole normalisation complete.")


if __name__ == "__main__":
    main()
