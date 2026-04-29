"""
google_sheets_connector.py

Uploads all processed CSVs under processed_data/ to a single Google Spreadsheet.
Each CSV becomes one tab named  <folder>_<file>  (e.g. ka_provider_job_posting).

Requirements:
    pip install gspread

Auth:
    1. Go to Google Cloud Console → APIs & Services → Credentials
    2. Create a Service Account and download the JSON key
    3. Share your target Google Spreadsheet with the service account email
    4. Set GSHEET_SERVICE_ACCOUNT_FILE env var to the JSON key path, OR
       pass --creds <path> on the command line

Usage:
    python google_sheets_connector.py --spreadsheet "Blue Dots Pipeline"
    python google_sheets_connector.py --spreadsheet "Blue Dots Pipeline" --creds service_account.json
    python google_sheets_connector.py --spreadsheet-id 1BxiM...  # use existing sheet by ID
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

try:
    import gspread
    from gspread.exceptions import APIError
except ImportError:
    sys.exit(
        "gspread is not installed. Run:  pip install gspread"
    )

PROCESSED_DATA_DIR = Path(__file__).parent / "processed_data"

# Google Sheets API allows at most 500 rows per batchUpdate request.
BATCH_SIZE = 500

# Seconds to wait between API calls to avoid hitting rate limits (60 writes/min).
WRITE_DELAY = 1.2


def find_csvs(root: Path) -> list[tuple[str, Path]]:
    """Return (tab_name, csv_path) pairs for every CSV under root."""
    results = []
    for csv_path in sorted(root.rglob("*.csv")):
        folder = csv_path.parent.name          # e.g. ka_provider
        stem = csv_path.stem.replace("_clean", "")  # e.g. job_posting
        tab_name = f"{folder}_{stem}"          # e.g. ka_provider_job_posting
        results.append((tab_name, csv_path))
    return results


def read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    """Return (headers, rows) from a CSV file."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def upload_tab(worksheet, headers: list[str], rows: list[list[str]]) -> None:
    """Clear a worksheet and upload headers + rows in batches."""
    worksheet.clear()
    time.sleep(WRITE_DELAY)

    all_rows = [headers] + rows
    for i in range(0, len(all_rows), BATCH_SIZE):
        chunk = all_rows[i : i + BATCH_SIZE]
        worksheet.append_rows(chunk, value_input_option="RAW")
        if i + BATCH_SIZE < len(all_rows):
            time.sleep(WRITE_DELAY)

    print(f"    Uploaded {len(rows):,} rows × {len(headers)} cols")


def get_or_create_worksheet(spreadsheet, tab_name: str):
    """Return the worksheet with tab_name, creating it if it doesn't exist."""
    try:
        return spreadsheet.worksheet(tab_name)
    except gspread.exceptions.WorksheetNotFound:
        return spreadsheet.add_worksheet(title=tab_name, rows=1, cols=1)


def main():
    parser = argparse.ArgumentParser(description="Upload processed CSVs to Google Sheets")
    parser.add_argument("--creds", metavar="FILE",
                        default=os.environ.get("GSHEET_SERVICE_ACCOUNT_FILE",
                                               "service_account.json"),
                        help="Path to service account JSON key")
    parser.add_argument("--ka-id", metavar="ID",
                        default="1e2Y2Qq2JfXDYk-WJuW7ZmpdN3g9ga3U_lCrbwKv7DTs",
                        help="Spreadsheet ID for Karnataka (ka_*) data")
    parser.add_argument("--up-id", metavar="ID",
                        default="1sccnSgXj1pzgsuyg7HdsKAaQ2z6Iob0cDSW27i0-GY0",
                        help="Spreadsheet ID for UP (up_*) data")
    args = parser.parse_args()

    creds_path = Path(args.creds)
    if not creds_path.exists():
        sys.exit(f"Credentials file not found: {creds_path}")

    # --- Auth ---
    gc = gspread.service_account(filename=str(creds_path))

    # --- Open both spreadsheets ---
    print(f"Opening Karnataka spreadsheet ({args.ka_id})...")
    ka_sheet = gc.open_by_key(args.ka_id)
    print(f"Opening UP spreadsheet ({args.up_id})...")
    up_sheet = gc.open_by_key(args.up_id)

    # --- Find CSVs ---
    csv_list = find_csvs(PROCESSED_DATA_DIR)
    if not csv_list:
        sys.exit(f"No CSV files found under {PROCESSED_DATA_DIR}")

    print(f"\nFound {len(csv_list)} CSV files to upload:\n")

    # --- Upload each CSV to the correct spreadsheet ---
    for tab_name, csv_path in csv_list:
        folder = csv_path.parent.name          # ka_provider / ka_seeker / up_*
        spreadsheet = ka_sheet if folder.startswith("ka_") else up_sheet
        relative = csv_path.relative_to(PROCESSED_DATA_DIR.parent)
        print(f"  [{tab_name}]  <- {relative}")
        headers, rows = read_csv(csv_path)
        if not headers:
            print("    Skipped (empty file)")
            continue
        try:
            ws = get_or_create_worksheet(spreadsheet, tab_name)
            upload_tab(ws, headers, rows)
        except APIError as e:
            print(f"    ERROR: {e}")
            print("    Waiting 60 s before retrying...")
            time.sleep(60)
            ws = get_or_create_worksheet(spreadsheet, tab_name)
            upload_tab(ws, headers, rows)
        time.sleep(WRITE_DELAY)

    print(f"\nDone.")
    print(f"  Karnataka: https://docs.google.com/spreadsheets/d/{ka_sheet.id}")
    print(f"  UP:        https://docs.google.com/spreadsheets/d/{up_sheet.id}")


if __name__ == "__main__":
    main()
