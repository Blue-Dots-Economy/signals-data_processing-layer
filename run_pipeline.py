"""
run_pipeline.py

Single entry point for the full Blue Dots data pipeline.

Steps:
    1. normalisation.py            — clean and flatten all raw CSVs
    2. llm_location_enrichment.py  — predict location_state and location_district
    3. role_normalisation.py       — map free-text role/title to canonical job roles
    4. flag_test.py                — flag test/spam records across all processed CSVs
    5. matchmaking.py              — match seekers to open jobs; write processed_data/matches/
    6. populate_dashboard.py       — upload UP and KA data to dashboard Google Sheets

Usage:
    python run_pipeline.py
    python run_pipeline.py --resume          # resume dashboard upload from last checkpoint
    python run_pipeline.py --state KA        # only upload KA data to the dashboard
    python run_pipeline.py --state UP        # only upload UP data to the dashboard
"""
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from preprocessing import normalisation
from preprocessing import llm_location_enrichment
from preprocessing import role_normalisation
from preprocessing import flag_test
from preprocessing import matchmaking
import populate_dashboard

PROCESSED_DIR  = Path(__file__).parent / "processed_data"
STEP_LOG_SHEET = "1zDf-c4Z_5As1hsR0rtwLashpUdCcnYwOlpg7HtEYj3g"

# One tab per processed CSV — written once after all steps complete
INTERMEDIATE_TABLES = [
    ("UP Job Postings",   "up_provider", "job_posting_clean.csv"),
    ("UP Applications",   "up_provider", "job_application_clean.csv"),
    ("UP Provider Users", "up_provider", "user_clean.csv"),
    ("UP Provider Orgs",  "up_provider", "organization_clean.csv"),
    ("UP Seekers",        "up_seeker",   "profile_clean.csv"),
    ("UP Seeker Users",   "up_seeker",   "user_clean.csv"),
    ("UP Members",        "up_seeker",   "member_clean.csv"),
    ("UP Seeker Orgs",    "up_seeker",   "organization_clean.csv"),
    ("KA Job Postings",   "ka_provider", "job_posting_clean.csv"),
    ("KA Applications",   "ka_provider", "job_application_clean.csv"),
    ("KA Provider Users", "ka_provider", "user_clean.csv"),
    ("KA Provider Orgs",  "ka_provider", "organization_clean.csv"),
    ("KA Seekers",        "ka_seeker",   "profile_clean.csv"),
    ("KA Seeker Users",   "ka_seeker",   "user_clean.csv"),
    ("KA Members",        "ka_seeker",   "member_clean.csv"),
    ("KA Seeker Orgs",    "ka_seeker",   "organization_clean.csv"),
    ("UP Matches",        "matches",     "up_matches.csv"),
    ("KA Matches",        "matches",     "ka_matches.csv"),
]

LOG_HEADERS = [
    "run_timestamp", "duration_s",
    "up_job_postings", "up_applications", "up_provider_users", "up_provider_orgs",
    "up_seekers", "up_seeker_users", "up_members", "up_seeker_orgs",
    "ka_job_postings", "ka_applications", "ka_provider_users", "ka_provider_orgs",
    "ka_seekers", "ka_seeker_users", "ka_members", "ka_seeker_orgs",
    "up_matches", "ka_matches",
    "errors",
]

_gs_client = None


def _get_sheet():
    global _gs_client
    if _gs_client is None:
        _gs_client = populate_dashboard._get_gspread_client()
    return _gs_client.open_by_key(STEP_LOG_SHEET)


def _get_or_create_ws(sh, title: str, rows: int = 1, cols: int = 1):
    try:
        return sh.worksheet(title)
    except Exception:
        import gspread
        return sh.add_worksheet(title=title, rows=rows, cols=cols)


def _upload_intermediate_data(run_start: float) -> None:
    """Write all processed CSVs to the intermediate sheet (one tab per table)
    and append a summary row to the Logs tab."""
    try:
        sh = _get_sheet()
    except Exception as e:
        print(f"  [Intermediate] Warning: could not connect to sheet ({e})")
        return

    counts: dict[str, int] = {}
    errors: list[str] = []

    for tab_name, subdir, filename in INTERMEDIATE_TABLES:
        path = PROCESSED_DIR / subdir / filename
        col_key = tab_name.lower().replace(" ", "_")

        if not path.exists():
            print(f"  [Intermediate] {tab_name}: file not found, skipping.")
            errors.append(tab_name)
            counts[col_key] = 0
            continue

        try:
            df = pd.read_csv(path, low_memory=False)
            df = df.fillna("").astype(str)
            data = [list(df.columns)] + df.values.tolist()
            row_count = len(df)

            ws = _get_or_create_ws(sh, tab_name, rows=row_count + 2, cols=len(df.columns))
            ws.clear()
            ws.update(data, value_input_option="RAW")
            counts[col_key] = row_count
            print(f"  [Intermediate] '{tab_name}': {row_count:,} rows uploaded.")
        except Exception as e:
            print(f"  [Intermediate] '{tab_name}': error — {e}")
            errors.append(tab_name)
            counts[col_key] = 0

        time.sleep(1)

    # Append one row to Logs tab
    try:
        duration = round(time.time() - run_start)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        log_ws = _get_or_create_ws(sh, "Logs", rows=1000, cols=len(LOG_HEADERS))
        existing = log_ws.get("1:1")
        if not existing or not existing[0]:
            log_ws.append_row(LOG_HEADERS)

        log_row = [
            timestamp,
            duration,
            counts.get("up_job_postings", 0),
            counts.get("up_applications", 0),
            counts.get("up_provider_users", 0),
            counts.get("up_provider_orgs", 0),
            counts.get("up_seekers", 0),
            counts.get("up_seeker_users", 0),
            counts.get("up_members", 0),
            counts.get("up_seeker_orgs", 0),
            counts.get("ka_job_postings", 0),
            counts.get("ka_applications", 0),
            counts.get("ka_provider_users", 0),
            counts.get("ka_provider_orgs", 0),
            counts.get("ka_seekers", 0),
            counts.get("ka_seeker_users", 0),
            counts.get("ka_members", 0),
            counts.get("ka_seeker_orgs", 0),
            counts.get("up_matches", 0),
            counts.get("ka_matches", 0),
            " | ".join(errors) if errors else "",
        ]
        log_ws.append_row(log_row, value_input_option="RAW")
        print(f"  [Intermediate] Logs tab updated ({duration}s elapsed).")
    except Exception as e:
        print(f"  [Intermediate] Warning: could not write Logs tab ({e})")


def main(resume: bool = False, only_state: str | None = None, test: bool = False):
    run_start = time.time()

    print("=" * 50)
    print("Step 1: Normalisation")
    print("=" * 50)
    normalisation.main()

    print()
    print("=" * 50)
    print("Step 2: LLM Location Enrichment")
    print("=" * 50)
    llm_location_enrichment.main()

    print()
    print("=" * 50)
    print("Step 3: Role Normalisation")
    print("=" * 50)
    role_normalisation.main()

    print()
    print("=" * 50)
    print("Step 4: Test / Spam Flagging")
    print("=" * 50)
    flag_test.main()

    print()
    print("=" * 50)
    print("Step 5: Matchmaking")
    print("=" * 50)
    matchmaking.main(only_state=only_state)

    print()
    print("=" * 50)
    print("Uploading intermediate data to sheet...")
    print("=" * 50)
    _upload_intermediate_data(run_start)

    print()
    print("=" * 50)
    print("Step 6: Populate Dashboard")
    print("=" * 50)
    populate_dashboard.main(resume=resume, only_state=only_state, test=test)

    print()
    print("Pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--state", choices=["UP", "KA"], default=None)
    parser.add_argument("--test", action="store_true", help="Upload to test sheets instead of production sheets.")
    args = parser.parse_args()
    main(resume=args.resume, only_state=args.state, test=args.test)
