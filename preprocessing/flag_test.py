"""
flag_test.py
Reads known test/spam entries from test.txt and adds a test_flag column
(1 = test/spam, 0 = genuine) to every processed_data CSV.

Matching logic
--------------
Phone numbers  : exact match after digit-only normalisation; also flags
                 obviously invalid numbers (too short/long, all-same digit,
                 non-numeric content).
Email addresses: exact case-insensitive match; also flags known internal
                 domains (dhiway.com, ekstepplus.org) and plus-addressing
                 patterns like +test / +demo.
Org names      : exact case-insensitive match against the known test org
                 list from test.txt; also keyword-based heuristics
                 (test, demo, xyz, trial, ekstep, dhiway, dw …).
"""

import re
import pandas as pd
from pathlib import Path

BASE_DIR      = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "processed_data"
TEST_FILE     = BASE_DIR / "test.txt"

# ── Parse test.txt ─────────────────────────────────────────────────────────────

def parse_test_file(path: Path) -> tuple[set, set, set]:
    """Return (phone_digits_set, email_set, org_name_set) from test.txt."""
    phones: set[str] = set()
    emails: set[str] = set()
    org_names: set[str] = set()

    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            val = raw.strip()
            if not val:
                continue

            if "@" in val:
                emails.add(val.lower())

            elif re.search(r"\d", val):
                digits = re.sub(r"\D", "", val)
                if digits:
                    phones.add(digits)

            else:
                org_names.add(val)

    return phones, emails, org_names


# ── Phone helpers ──────────────────────────────────────────────────────────────

def _digits(val) -> str:
    """Strip everything except digits; return '' if nothing remains."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return re.sub(r"\D", "", str(val).strip())


def is_test_phone(val, test_phones: set[str]) -> bool:
    d = _digits(val)
    if not d:
        return False

    # Known test list (full digits match)
    if d in test_phones:
        return True

    # Also match without leading 91 or 091 country-code prefix
    for prefix in ("91", "091", "0"):
        if d.startswith(prefix) and len(d) > len(prefix):
            if d[len(prefix):] in test_phones or d[len(prefix):].lstrip("0") in test_phones:
                return True

    # Heuristic: all-same digit (e.g. 0000000000, 9999999999)
    if len(set(d)) == 1:
        return True

    # Heuristic: obviously wrong length for Indian mobile (10 or 12 digits)
    if len(d) < 8 or len(d) > 13:
        return True

    return False


# ── Email helpers ──────────────────────────────────────────────────────────────

_TEST_DOMAINS  = {"dhiway.com", "ekstepplus.org"}
_PLUS_TEST_RE  = re.compile(r"\+(test|demo|trial|sample)\d*@", re.IGNORECASE)


def is_test_email(val, test_emails: set[str]) -> bool:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    e = str(val).strip().lower()
    if not e or "@" not in e:
        return False

    if e in test_emails:
        return True

    domain = e.split("@")[-1]
    if domain in _TEST_DOMAINS:
        return True

    if _PLUS_TEST_RE.search(e):
        return True

    return False


# ── Org-name helpers ───────────────────────────────────────────────────────────

_TEST_KW_RE = re.compile(
    r"\b(test|demo|trial|sample|dummy|fake|xyz|dw|ekstep|dhiway)\b",
    re.IGNORECASE,
)


def is_test_org(val, test_org_names: set[str]) -> bool:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    n = str(val).strip()
    if not n:
        return False

    # Exact match (case-insensitive) against known list
    if n.lower() in {o.lower() for o in test_org_names}:
        return True

    # Keyword heuristic
    if _TEST_KW_RE.search(n):
        return True

    return False


# ── Per-sheet flagging ─────────────────────────────────────────────────────────

def flag_user(df: pd.DataFrame, phones: set, emails: set) -> pd.DataFrame:
    df = df.copy()
    df["test_flag"] = df.apply(
        lambda r: int(
            is_test_phone(r.get("phone_number"), phones)
            or is_test_email(r.get("email"), emails)
        ),
        axis=1,
    )
    return df


def flag_profile(df: pd.DataFrame, phones: set) -> pd.DataFrame:
    df = df.copy()
    df["test_flag"] = df.apply(
        lambda r: int(is_test_phone(r.get("phone"), phones)),
        axis=1,
    )
    return df


def flag_organization(
    df: pd.DataFrame,
    phones: set,
    emails: set,
    org_names: set,
) -> pd.DataFrame:
    df = df.copy()
    df["test_flag"] = df.apply(
        lambda r: int(
            is_test_phone(r.get("contact_phone"), phones)
            or is_test_email(r.get("contact_email"), emails)
            or is_test_org(r.get("name"), org_names)
        ),
        axis=1,
    )
    return df


def flag_member(df: pd.DataFrame, flagged_org_ids: set[str]) -> pd.DataFrame:
    """Flag members whose organisation is already marked as test/spam."""
    df = df.copy()
    df["test_flag"] = df["organization_id"].apply(
        lambda oid: 1 if str(oid) in flagged_org_ids else 0
    )
    return df


def flag_job_application(
    df: pd.DataFrame, phones: set, emails: set
) -> pd.DataFrame:
    df = df.copy()
    df["test_flag"] = df.apply(
        lambda r: int(
            is_test_phone(r.get("seeker_phone"), phones)
            or is_test_email(r.get("seeker_email"), emails)
        ),
        axis=1,
    )
    return df


def flag_job_posting(df: pd.DataFrame, org_names: set) -> pd.DataFrame:
    df = df.copy()
    df["test_flag"] = df.apply(
        lambda r: int(is_test_org(r.get("organization_name"), org_names)),
        axis=1,
    )
    return df


# ── Sheet dispatch table ───────────────────────────────────────────────────────

# Maps csv filename → flag-type key
SHEET_DISPATCH: dict[str, str] = {
    "user_clean.csv":            "user",
    "profile_clean.csv":         "profile",
    "organization_clean.csv":    "organization",
    "member_clean.csv":          "member",
    "job_application_clean.csv": "job_application",
    "job_posting_clean.csv":     "job_posting",
}


def apply_flags(
    df: pd.DataFrame,
    flag_type: str,
    phones: set,
    emails: set,
    org_names: set,
    flagged_org_ids: set[str] | None = None,
) -> pd.DataFrame:
    if flag_type == "user":
        return flag_user(df, phones, emails)
    if flag_type == "profile":
        return flag_profile(df, phones)
    if flag_type == "organization":
        return flag_organization(df, phones, emails, org_names)
    if flag_type == "member":
        return flag_member(df, flagged_org_ids or set())
    if flag_type == "job_application":
        return flag_job_application(df, phones, emails)
    if flag_type == "job_posting":
        return flag_job_posting(df, org_names)
    raise ValueError(f"Unknown flag type: {flag_type}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    phones, emails, org_names = parse_test_file(TEST_FILE)
    print(
        f"Loaded from test.txt: {len(phones)} phone entries, "
        f"{len(emails)} email entries, {len(org_names)} org-name entries"
    )

    total_rows = total_flagged = 0

    # Non-member sheets first so we can propagate flagged org IDs to members
    NON_MEMBER = {k: v for k, v in SHEET_DISPATCH.items() if v != "member"}
    MEMBER     = {k: v for k, v in SHEET_DISPATCH.items() if v == "member"}

    for subdir in sorted(PROCESSED_DIR.iterdir()):
        if not subdir.is_dir():
            continue
        print(f"\n{subdir.name}/")

        # Pass 1: flag everything except members
        for csv_name, flag_type in NON_MEMBER.items():
            path = subdir / csv_name
            if not path.exists():
                continue
            df = pd.read_csv(path, low_memory=False)
            if "test_flag" in df.columns:
                df = df.drop(columns=["test_flag"])
            df = apply_flags(df, flag_type, phones, emails, org_names)
            df.to_csv(path, index=False)
            flagged = int(df["test_flag"].sum())
            total_rows    += len(df)
            total_flagged += flagged
            print(f"  {csv_name:<32} {len(df):>6} rows  |  {flagged:>5} flagged")

        # Collect flagged org IDs from this dataset's organisation_clean
        org_path = subdir / "organization_clean.csv"
        flagged_org_ids: set[str] = set()
        if org_path.exists():
            org_df = pd.read_csv(org_path, low_memory=False)
            if "test_flag" in org_df.columns and "id" in org_df.columns:
                flagged_org_ids = set(
                    org_df.loc[org_df["test_flag"] == 1, "id"].astype(str).tolist()
                )

        # Pass 2: flag members using propagated org IDs
        for csv_name, flag_type in MEMBER.items():
            path = subdir / csv_name
            if not path.exists():
                continue
            df = pd.read_csv(path, low_memory=False)
            if "test_flag" in df.columns:
                df = df.drop(columns=["test_flag"])
            df = apply_flags(df, flag_type, phones, emails, org_names, flagged_org_ids)
            df.to_csv(path, index=False)
            flagged = int(df["test_flag"].sum())
            total_rows    += len(df)
            total_flagged += flagged
            print(f"  {csv_name:<32} {len(df):>6} rows  |  {flagged:>5} flagged")

    print(f"\nDone. {total_flagged} / {total_rows} rows flagged across all sheets.")


if __name__ == "__main__":
    main()
