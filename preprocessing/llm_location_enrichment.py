"""
llm_location_enrichment.py

Reads every cleaned CSV produced by normalisation.py, calls the Claude API
to predict the Indian state and district from existing location fields, and
writes two new columns back into each file:

    llm_location_state    – predicted Indian state  (e.g. "Karnataka")
    llm_location_district – predicted district       (e.g. "Bengaluru Urban")

API key
-------
Read from the CLAUDE_API_KEY environment variable or the .env file in the
project root.  Never hard-code the key in this file.

Run after normalisation.py:
    python preprocessing/llm_location_enrichment.py
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

# ── Per-sheet config ───────────────────────────────────────────────────────────
SHEET_LOCATION_COLS: dict[str, list[str]] = {
    "profile_clean.csv": [
        "location_city",
        "location_state",
        "location_address",
        "location_country",
        "current_location",
        "hometown",
    ],
    "organization_clean.csv": [
        "address",
    ],
    "job_posting_clean.csv": [
        "provider_location_city",
        "provider_location_state",
        "provider_location_address",
        "provider_location_country",
    ],
    "job_application_clean.csv": [
        "app_location_city",
        "app_location_state",
        "app_location_address",
        "seeker_current_location",
    ],
}

_BATCH_SIZE  = 50
_CONCURRENCY = 3


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_location_text(row: pd.Series, cols: list[str]) -> str:
    parts = []
    for col in cols:
        val = row.get(col)
        if val is None:
            continue
        if isinstance(val, float) and pd.isna(val):
            continue
        s = str(val).strip()
        if s:
            parts.append(s)
    return ", ".join(parts)


def _predict_batch(
    batch_num: int,
    texts: list[str],
) -> tuple[int, list[dict]]:
    import httpx
    numbered = "\n".join(
        f"{i + 1}. {t if t else '(no location data)'}"
        for i, t in enumerate(texts)
    )
    prompt = (
        "You are an expert in Indian geography.\n"
        "For each numbered location description, identify:\n"
        "  - state: the full Indian state name (e.g. 'Karnataka'), or null\n"
        "  - district: the district within that state (e.g. 'Bengaluru Urban'), or null\n"
        "Reply with ONLY a JSON array — one object per entry, in the same order.\n"
        "Example: [{\"state\": \"Karnataka\", \"district\": \"Bengaluru Urban\"}, ...]\n\n"
        f"Locations:\n{numbered}"
    )
    print(f"Batch {batch_num}: sending {len(texts)} rows to API...", flush=True)
    try:
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": CLAUDE_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": _BATCH_SIZE * 30,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60.0,
        )
        response.raise_for_status()
        print(f"Batch {batch_num}: API response received, parsing...", flush=True)
        raw = response.json()["content"][0]["text"].strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[^\n]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        results: list[dict] = json.loads(raw)
        while len(results) < len(texts):
            results.append({"state": None, "district": None})
        return batch_num, results[: len(texts)]
    except Exception as exc:
        print(f"      [!] Batch {batch_num} failed: {type(exc).__name__}: {exc!r}", flush=True)
        return batch_num, [None] * len(texts)


# ── Core enrichment ────────────────────────────────────────────────────────────

def enrich_dataframe(df: pd.DataFrame, location_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["llm_location_state"]    = None
    df["llm_location_district"] = None

    existing_cols = [c for c in location_cols if c in df.columns]
    if not existing_cols:
        print("      No matching location columns found — skipping.")
        return df

    loc_texts = df.apply(lambda r: _build_location_text(r, existing_cols), axis=1)
    needs_llm = [
        (idx, txt)
        for idx, txt in loc_texts.items()
        if txt
    ]

    if not needs_llm:
        return df

    batches = [needs_llm[i: i + _BATCH_SIZE] for i in range(0, len(needs_llm), _BATCH_SIZE)]
    print(
        f"      Rows to enrich via API: {len(needs_llm)} "
        f"({len(batches)} batches, batch size {_BATCH_SIZE}, concurrency {_CONCURRENCY})"
    )

    # Test first batch — fail fast if API is unreachable
    first_batch_num, first_results = _predict_batch(0, [txt for _, txt in batches[0]])
    if all(r is None for r in first_results):
        raise RuntimeError(
            "LLM API unreachable — first batch failed. "
            "Check network connectivity and CLAUDE_API_KEY. Aborting pipeline."
        )

    results_by_batch = [(first_batch_num, first_results)]
    with ThreadPoolExecutor(max_workers=_CONCURRENCY) as executor:
        futures = {
            executor.submit(_predict_batch, i, [txt for _, txt in batch]): i
            for i, batch in enumerate(batches)
            if i > 0
        }
        for future in as_completed(futures):
            results_by_batch.append(future.result())

    results_by_batch = sorted(results_by_batch, key=lambda x: x[0])

    completed = 0
    for (_, predictions), batch in zip(results_by_batch, batches):
        for (idx, _), pred in zip(batch, predictions):
            if not isinstance(pred, dict):
                pred = {"state": None, "district": None}
            df.at[idx, "llm_location_state"]    = pred.get("state")
            df.at[idx, "llm_location_district"] = pred.get("district")
        completed += len(batch)
        print(f"      ... {completed}/{len(needs_llm)} rows done")

    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    dataset_dirs = sorted(d for d in PROCESSED_DIR.iterdir() if d.is_dir())

    if not dataset_dirs:
        print(f"No dataset folders found under {PROCESSED_DIR}. Run normalisation.py first.")
        return

    for dataset_dir in dataset_dirs:
        print(f"\n-- {dataset_dir.name} --")

        for csv_name, loc_cols in SHEET_LOCATION_COLS.items():
            csv_path = dataset_dir / csv_name
            if not csv_path.exists():
                print(f"  {csv_name}: not found, skipping.")
                continue

            print(f"  {csv_name}")
            df = pd.read_csv(csv_path, low_memory=False)
            df = enrich_dataframe(df, loc_cols)
            df.to_csv(csv_path, index=False)
            print(f"      Enriched with LLM predictions and saved -> {csv_path}")

    print("\nAll sheets enriched.")


if __name__ == "__main__":
    main()
