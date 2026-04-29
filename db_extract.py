"""
db_extract.py

Fetches raw tables directly from the KA and UP PostgreSQL databases
and writes them as CSVs into raw_data/ in the same structure the
pipeline expects.

Replaces the manual CSV upload step.

Tables fetched:
  Seeker DB  (port 5433): profile, user, member, organization
  Provider DB (port 5432): job_posting, job_application, member, organization, user

Credentials are loaded from:
  1. Secret Manager (DB_CREDENTIALS) — when running in Cloud Run
  2. db_credentials.json in the project root — when running locally
"""

import json
from pathlib import Path

import pandas as pd
import psycopg2

BASE_DIR = Path(__file__).parent
RAW_DIR  = BASE_DIR / "raw_data"

# ── Credentials ────────────────────────────────────────────────────────────────

def _load_credentials() -> dict:
    local = BASE_DIR / "db_credentials.json"
    if local.exists():
        return json.loads(local.read_text(encoding="utf-8"))
    from google.cloud import secretmanager
    sm   = secretmanager.SecretManagerServiceClient()
    name = "projects/blue-dots-project/secrets/DB_CREDENTIALS/versions/latest"
    data = sm.access_secret_version(request={"name": name}).payload.data
    return json.loads(data.decode("utf-8"))


# ── DB fetch ───────────────────────────────────────────────────────────────────

def _fetch_table(host: str, port: int, dbname: str, user: str,
                 password: str, table: str) -> pd.DataFrame:
    print(f"    Fetching {table} from {host}:{port} ...")
    conn = psycopg2.connect(
        host=host, port=port, dbname=dbname,
        user=user, password=password,
        connect_timeout=30,
    )
    try:
        df = pd.read_sql(f'SELECT * FROM "{table}"', conn)
    finally:
        conn.close()
    print(f"    {table}: {len(df)} rows")
    return df


def _save(df: pd.DataFrame, out_dir: Path, filename: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # PostgreSQL JSONB columns arrive as Python dicts/lists — serialize to valid JSON strings
    # so that downstream CSV readers can parse them correctly.
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
            )
    df.to_csv(out_dir / filename, index=False)


# ── Per-dataset extraction ─────────────────────────────────────────────────────

def extract_seeker(host: str, creds: dict, out_dir: Path) -> None:
    c = creds["seeker"]
    for table in ["profile", "user", "member", "organization"]:
        df = _fetch_table(host, c["port"], c["dbname"], c["user"], c["password"], table)
        _save(df, out_dir, f"{table}.csv")


def extract_provider(host: str, creds: dict, out_dir: Path) -> None:
    c = creds["provider"]
    for table in ["job_posting", "job_application", "member", "organization", "user"]:
        df = _fetch_table(host, c["port"], c["dbname"], c["user"], c["password"], table)
        _save(df, out_dir, f"{table}.csv")


# ── Entry point ────────────────────────────────────────────────────────────────

def main(env: str = "prod") -> None:
    creds = _load_credentials()

    host_key = f"host_{env}"

    print("\nExtracting KA seeker ...")
    extract_seeker(creds["ka"][host_key], creds["ka"], RAW_DIR / "ka_seeker")

    print("\nExtracting KA provider ...")
    extract_provider(creds["ka"][host_key], creds["ka"], RAW_DIR / "ka_provider")

    print("\nExtracting UP seeker ...")
    extract_seeker(creds["up"]["host_prod"], creds["up"], RAW_DIR / "up_seeker")

    print("\nExtracting UP provider ...")
    extract_provider(creds["up"]["host_prod"], creds["up"], RAW_DIR / "up_provider")

    print("\nDB extraction complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["prod", "uat"], default="prod")
    args = parser.parse_args()
    main(env=args.env)
