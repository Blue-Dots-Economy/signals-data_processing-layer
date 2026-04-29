"""
populate_dashboard.py

Populates dashboard Google Sheets with UP and KA data across three tabs each:
  OPEN_ROLES      <- <state>_provider job_posting_clean  (status = 'open')
  job_application <- <state>_provider job_application_clean
  seeker_profile  <- <state>_seeker   profile_clean

Design principles
-----------------
1. Dynamic headers: column order is read from row 1 of each destination tab,
   so reordering columns in the sheet never silently misaligns data.
2. Checkpointing: after each successful tab upload the state is saved to
   .pipeline_state.json.  Run with --resume to skip already-completed tabs.
3. Column mapping: data is built as a dict keyed by column name, then
   ordered according to the live sheet headers — unknown destination columns
   are left blank rather than causing an error.

Usage:
    python populate_dashboard.py           # full run (overwrites all tabs)
    python populate_dashboard.py --resume  # skip tabs already marked done
    python populate_dashboard.py --state KA   # only KA dataset
    python populate_dashboard.py --state UP   # only UP dataset
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime as _dt
from pathlib import Path

import gspread
import pandas as pd
from gspread.exceptions import APIError

BASE_DIR      = Path(__file__).parent
PROCESSED_DIR = BASE_DIR / "processed_data"
CREDS_FILE    = BASE_DIR / "service_account.json"
# UP sheets
SHEET_ID_UP_PROVIDER = "1gbuBqb8-T48mHe2xteAZK1qDhOaPVJzYCWQQT1MOXEQ"  # OPEN_ROLES + job_application
SHEET_ID_UP_SEEKER   = "1J2WDeSOCaIVz2KvWMmI9dVE4iTh_Dqbt8Aqb6423a2U"  # seeker_profile
# KA sheets
SHEET_ID_KA_PROVIDER = "1ivktbvMI7EasBta7CNmRA6LOc6N4y4AUXqyyFXizoZU"  # OPEN_ROLES + job_application
SHEET_ID_KA_SEEKER   = "1N4fzt9ONsSyE7cCRglwGBxge-GWybN7cCq1PLMCrrbk"  # seeker_profile
SHEET_ID_SUMMARY     = "1ko6geEQmGPR3LcsH7ECR5SpeGVmAbuQC4VX78M4LxJg"   # Pipeline History

# Test sheets (used when --test flag is passed; production sheets are never touched)
TEST_SHEET_ID_KA_PROVIDER = "1b2TCepXJc2t871CTTqjBJMpBJWIk27m-zxwTcMV5sys"
TEST_SHEET_ID_KA_SEEKER   = "1p8OTqnY5X1ZR-ykAMu7yLTxZfG2xCIvKKgX-h3-fJVo"
TEST_SHEET_ID_UP_PROVIDER = "1hUdKQDQYquWs_CuDwQvDdrh12oiexh7PqzhLML84t_k"
TEST_SHEET_ID_UP_SEEKER   = "1odIv77B5J22uPzJICW-chSKwKHIZCswcWog4X8q9El4"
STATE_FILE    = BASE_DIR / ".pipeline_state.json"

# backwards-compat aliases used in TABS (kept for any external references)
SHEET_ID        = SHEET_ID_UP_PROVIDER
SHEET_ID_SEEKER = SHEET_ID_UP_SEEKER

BATCH_SIZE  = 500
WRITE_DELAY = 1.2


# ── Helpers ────────────────────────────────────────────────────────────────────

def safe(val) -> str:
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val)


# Google Sheets epoch starts 1899-12-30
_SHEETS_EPOCH = _dt(1899, 12, 30)

def to_sheets_date(val) -> float | str:
    """Convert a date/datetime string to a Google Sheets numeric serial.
    Returns the serial float so MAXIFS/date formulas work correctly.
    Falls back to the raw string if parsing fails.
    """
    s = safe(val)
    if not s:
        return ""
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            delta = _dt.strptime(s.strip(), fmt) - _SHEETS_EPOCH
            return delta.days + delta.seconds / 86400
        except ValueError:
            continue
    return s


def to_num(val):
    """Return val as int or float; empty string if blank/unparseable."""
    s = safe(val)
    if not s:
        return ""
    try:
        f = float(s)
        return int(f) if f == int(f) else f
    except (ValueError, TypeError):
        return s


def _parse_date(val) -> "_dt | None":
    s = safe(val).strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return _dt.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _fmt_date_text(dt) -> str:
    """Format datetime as '9-Dec-2025' (no leading zero) for display."""
    if dt is None:
        return ""
    return f"{dt.day}-{dt.strftime('%b')}-{dt.strftime('%Y')}"


def fmt_date(val) -> str:
    """Format a date/datetime value as M/D/YYYY (no leading zeros)."""
    if isinstance(val, _dt):
        dt = val
    else:
        dt = _parse_date(val)
    if dt is None:
        return ""
    return f"{dt.month}/{dt.day}/{dt.year}"


def clean_phone(val) -> str:
    s = safe(val)
    if not s:
        return ""
    digits = re.sub(r"\D", "", s)
    if len(digits) == 10:
        return f"+91{digits}"
    if len(digits) == 12 and digits.startswith("91"):
        return f"+{digits}"
    if len(digits) == 13 and digits.startswith("091"):
        return f"+91{digits[3:]}"
    return f"+{digits}" if digits else ""


def fmt_salary(row) -> str:
    mn, mx = row.get("min_monthly_in_hand"), row.get("max_monthly_in_hand")
    has_min = mn is not None and safe(mn) not in ("", "nan")
    has_max = mx is not None and safe(mx) not in ("", "nan")
    try:
        if has_min and has_max:
            return f"{int(float(mn))} - {int(float(mx))}"
        if has_min:
            return str(int(float(mn)))
        if has_max:
            return str(int(float(mx)))
    except (ValueError, TypeError):
        pass
    return ""


def parse_gps(metadata_val):
    s = safe(metadata_val)
    if not s:
        return "", ""
    try:
        m = json.loads(s)
        who = m.get("whoIAm", {})
        if isinstance(who, dict):
            loc = who.get("locationData", {})
            if isinstance(loc, dict):
                gps = loc.get("gps", {})
                if isinstance(gps, dict):
                    return to_num(gps.get("lat")), to_num(gps.get("lng"))
    except Exception:
        pass
    return "", ""


def row_from_dict(data: dict, headers: list[str]) -> list:
    """Build a sheet row in header order; blank for any header not in data."""
    return [data.get(h, "") for h in headers]


# ── Sheet upload ───────────────────────────────────────────────────────────────

def get_formula_columns(ws) -> list[tuple[int, str]]:
    """
    Read row 2 with formula rendering and return [(col_idx, formula), ...]
    for every cell that contains a formula (starts with '=').
    Call this BEFORE clearing the sheet.
    """
    try:
        row2 = ws.get("2:2", value_render_option="FORMULA")
    except Exception:
        return []
    if not row2:
        return []
    return [
        (i, cell)
        for i, cell in enumerate(row2[0])
        if isinstance(cell, str) and cell.startswith("=")
    ]


def restore_formulas(
    sh,
    ws,
    formula_cols: list[tuple[int, str]],
    num_data_rows: int,
) -> None:
    """
    Write formula templates back to row 2, then copy them down to every
    data row using the Sheets API copyPaste request (preserves relative refs).
    """
    if not formula_cols or num_data_rows < 1:
        return

    # Write each formula template to its row-2 cell
    for col_idx, formula in formula_cols:
        cell = gspread.utils.rowcol_to_a1(2, col_idx + 1)
        try:
            ws.update([[formula]], cell, value_input_option="USER_ENTERED")
        except APIError as e:
            print(f"    Warning: could not write formula to {cell}: {e}")
        time.sleep(0.3)

    # Copy formulas down to all data rows in one batch request
    requests = [
        {
            "copyPaste": {
                "source": {
                    "sheetId": ws.id,
                    "startRowIndex": 1,          # row 2 (0-indexed)
                    "endRowIndex": 2,
                    "startColumnIndex": col_idx,
                    "endColumnIndex": col_idx + 1,
                },
                "destination": {
                    "sheetId": ws.id,
                    "startRowIndex": 1,
                    "endRowIndex": num_data_rows + 1,
                    "startColumnIndex": col_idx,
                    "endColumnIndex": col_idx + 1,
                },
                "pasteType": "PASTE_FORMULA",
                "pasteOrientation": "NORMAL",
            }
        }
        for col_idx, _ in formula_cols
    ]
    sh.batch_update({"requests": requests})
    print(f"    Restored {len(formula_cols)} formula column(s) across {num_data_rows} rows")


def upload_tab(ws, headers: list, rows: list) -> None:
    ws.clear()
    time.sleep(WRITE_DELAY)
    all_rows = [headers] + rows
    for i in range(0, len(all_rows), BATCH_SIZE):
        chunk = all_rows[i: i + BATCH_SIZE]
        try:
            ws.append_rows(chunk, value_input_option="USER_ENTERED")
        except APIError as e:
            print(f"    API error: {e} — waiting 60 s then retrying…")
            time.sleep(60)
            ws.append_rows(chunk, value_input_option="USER_ENTERED")
        if i + BATCH_SIZE < len(all_rows):
            time.sleep(WRITE_DELAY)
    print(f"    Uploaded {len(rows):,} rows × {len(headers)} cols")




def get_or_create_ws(sh, title: str):
    try:
        return sh.worksheet(title)
    except gspread.exceptions.WorksheetNotFound:
        return sh.add_worksheet(title=title, rows=1, cols=1)


# ── Checkpointing ──────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Load CSVs ──────────────────────────────────────────────────────────────────

def _load_match_counts(matches_csv: Path) -> dict:
    """
    Load match CSV and return count dicts keyed by job or seeker id.
    Returns a dict with four keys:
      job_right_fit    {provider_job_id: int}
      job_partial_fit  {provider_job_id: int}
      sk_right_fit     {seeker_id: int}
      sk_partial_fit   {seeker_id: int}
    Returns empty dicts for all keys if the file doesn't exist yet.
    """
    empty = {"job_right_fit": {}, "job_partial_fit": {}, "sk_right_fit": {}, "sk_partial_fit": {}}
    if not matches_csv.exists():
        print(f"  Note: no match file found at {matches_csv} -- fit counts will be blank.")
        return empty
    try:
        df = pd.read_csv(matches_csv, usecols=["provider_job_id", "seeker_id", "match_category"], low_memory=False)
    except (pd.errors.EmptyDataError, ValueError):
        print(f"  Note: match file at {matches_csv} is empty -- fit counts will be blank.")
        return empty
    rf  = df[df["match_category"] == "Right Fit"]
    pf  = df[df["match_category"] == "Partial Fit"]
    return {
        "job_right_fit":   rf.groupby("provider_job_id").size().to_dict(),
        "job_partial_fit": pf.groupby("provider_job_id").size().to_dict(),
        "sk_right_fit":    rf.groupby("seeker_id").size().to_dict(),
        "sk_partial_fit":  pf.groupby("seeker_id").size().to_dict(),
    }


def _load_state_data(provider_dir: Path, seeker_dir: Path, state_label: str) -> dict:
    matches_csv = BASE_DIR / "processed_data" / "matches" / f"{state_label.lower()}_matches.csv"
    counts = _load_match_counts(matches_csv)
    return dict(
        job_posting     = pd.read_csv(provider_dir / "job_posting_clean.csv",     low_memory=False),
        job_application = pd.read_csv(provider_dir / "job_application_clean.csv", low_memory=False),
        user_provider   = pd.read_csv(provider_dir / "user_clean.csv",            low_memory=False),
        org_provider    = pd.read_csv(provider_dir / "organization_clean.csv",    low_memory=False),
        profile         = pd.read_csv(seeker_dir   / "profile_clean.csv",         low_memory=False),
        user_seeker     = pd.read_csv(seeker_dir   / "user_clean.csv",            low_memory=False),
        member_seeker   = pd.read_csv(seeker_dir   / "member_clean.csv",          low_memory=False),
        org_seeker      = pd.read_csv(seeker_dir   / "organization_clean.csv",    low_memory=False),
        right_fit_counts   = counts["job_right_fit"],
        partial_fit_counts = counts["job_partial_fit"],
        sk_right_fit_counts  = counts["sk_right_fit"],
        sk_partial_fit_counts = counts["sk_partial_fit"],
    )


def load_data():
    """Load UP data (backwards-compatible alias)."""
    return _load_state_data(PROCESSED_DIR / "up_provider", PROCESSED_DIR / "up_seeker", "UP")


def load_ka_data():
    """Load KA data."""
    return _load_state_data(PROCESSED_DIR / "ka_provider", PROCESSED_DIR / "ka_seeker", "KA")


# ── OPEN_ROLES ─────────────────────────────────────────────────────────────────

def build_open_roles(data: dict, headers: list[str]) -> list[list]:
    TODAY         = _dt.today().date()
    job_posting   = data["job_posting"]
    user_provider = data["user_provider"]
    job_app_df    = data["job_application"]
    right_fit_counts   = data.get("right_fit_counts", {})
    partial_fit_counts = data.get("partial_fit_counts", {})

    open_jobs = job_posting[
        job_posting["status"].str.strip().str.lower() == "open"
    ].copy()

    user_lkp = (
        user_provider.drop_duplicates("id")
        .set_index("id")[["name", "phone_number", "email"]]
        .to_dict("index")
    )

    # Application lookup keyed by job_id
    app_by_job: dict = {}
    for _, a in job_app_df.iterrows():
        jid = safe(a.get("job_id"))
        app_by_job.setdefault(jid, []).append({
            "applied_at_dt":      _parse_date(a.get("applied_at")),
            "application_status": safe(a.get("application_status")).lower(),
            "status":             safe(a.get("status")).lower(),
        })

    # First pass: build combo keys for duplicate detection (cols C–I)
    combo_keys = []
    for _, r in open_jobs.iterrows():
        cb  = safe(r.get("created_by"))
        u   = user_lkp.get(cb, {})
        key = "||".join([
            safe(r.get("organization_name")),
            safe(r.get("role")),
            safe(r.get("positions")),
            fmt_salary(r),
            safe(r.get("job_description")),
            safe(u.get("name")),
            safe(u.get("phone_number")),
        ])
        combo_keys.append(key)
    key_counts: dict = {}
    for k in combo_keys:
        key_counts[k] = key_counts.get(k, 0) + 1

    rows = []
    for idx, (_, r) in enumerate(open_jobs.iterrows()):
        cb  = safe(r.get("created_by"))
        u   = user_lkp.get(cb, {})
        jid = safe(r.get("id"))

        # Base values
        org_name      = safe(r.get("organization_name"))
        role_val      = safe(r.get("role"))
        openings_val  = to_num(r.get("positions"))
        salary_str    = fmt_salary(r)
        job_desc      = safe(r.get("job_description"))
        contact_name  = safe(u.get("name"))
        contact_num   = safe(u.get("phone_number"))
        contact_email = safe(u.get("email"))

        # Application counts
        apps          = app_by_job.get(jid, [])
        n_apps        = len(apps)
        n_shortlisted = sum(1 for a in apps if a["application_status"] == "shortlisted")
        n_rejected    = sum(1 for a in apps if a["application_status"] == "rejected")

        # Job Age
        job_post_dt = _parse_date(r.get("created_at"))
        job_age     = (TODAY - job_post_dt.date()).days if job_post_dt else ""

        # Last Application Date (max applied_at → dd-mmm-yyyy text)
        applied_dts  = [a["applied_at_dt"] for a in apps if a["applied_at_dt"]]
        last_app_dt  = max(applied_dts) if applied_dts else None
        last_app_str = fmt_date(last_app_dt)

        # Application Pending From (days since oldest open-status application)
        open_dts = [a["applied_at_dt"] for a in apps if a["applied_at_dt"] and a["status"] == "open"]
        app_pending = (TODAY - min(open_dts).date()).days if open_dts else ""

        # Shortlisted Age (days since oldest closed-status application)
        closed_dts = [a["applied_at_dt"] for a in apps if a["applied_at_dt"] and a["status"] == "closed"]
        shortlisted_age = (TODAY - min(closed_dts).date()).days if closed_dts else ""

        # Rejected Age (days since oldest archived-status application)
        archived_dts = [a["applied_at_dt"] for a in apps if a["applied_at_dt"] and a["status"] == "archived"]
        rejected_age = (TODAY - min(archived_dts).date()).days if archived_dts else ""

        # Duplicate (cols C–I appear more than once across all open jobs)
        duplicate = "Yes" if key_counts[combo_keys[idx]] > 1 else "No"

        # Profile Completion (cols C–J non-empty / 8)
        profile_completion = round(sum(1 for v in [org_name, role_val, safe(r.get('positions')), salary_str, job_desc, contact_name, contact_num, contact_email] if v) / 8, 4)

        # Job Status (mirrors IFS formula from reference sheet)
        n_resolved = n_shortlisted + n_rejected
        try:
            openings_int = int(float(safe(r.get("positions")))) if safe(r.get("positions")) else 0
        except (ValueError, TypeError):
            openings_int = 0
        # MIN(Shortlisted Age, Rejected Age) — ignore blanks
        age_candidates = [a for a in [shortlisted_age, rejected_age] if a != ""]
        min_resolved_age = min(age_candidates) if age_candidates else None

        if isinstance(job_age, (int, float)) and job_age <= 7:
            job_status = "New"
        elif n_apps > 0 and n_resolved >= openings_int and openings_int > 0:
            job_status = "Satisfied"
        elif n_apps > 0 and n_resolved > 0 and min_resolved_age is not None and min_resolved_age <= 30:
            job_status = "Active"
        elif (
            (n_apps > 0 and n_resolved > 0 and min_resolved_age is not None and 31 <= min_resolved_age <= 90 and n_resolved < openings_int)
            or (isinstance(job_age, (int, float)) and 7 < job_age <= 30 and n_apps == 0)
        ):
            job_status = "At Risk"
        else:
            job_status = "Inactive"

        # Recommended Action for Job Provider
        rec_provider_parts = []
        if n_apps > 0 and n_shortlisted == 0 and n_rejected == 0:
            rec_provider_parts.append("Nudge job providers")
        if isinstance(job_age, int) and job_age >= 30 and n_shortlisted == 0 and n_rejected == 0:
            rec_provider_parts.append("Check job freshness")
        rec_provider = "; ".join(rec_provider_parts)

        # Recommended Action for Job Seeker (Right Fit Seekers Count is blank → always empty)
        rec_seeker = ""

        # Request_type
        request_type = "Jobs" if jid else ""

        # Follow up for (mirrors sheet formula: C=org_name, D=role, E=openings, F=salary,
        #   G=job_desc, H=contact_name, J=contact_email, K=org_id, L=org_name,
        #   O=shortlisted, P=rejected, Q=current_openings)
        follow_parts = []
        if not org_name or not role_val or not safe(r.get("positions")):
            follow_parts.append("Update Location")
        if not salary_str:
            follow_parts.append("Update Role")
        if not job_desc:
            follow_parts.append("Update Openings")
        if not contact_name:
            follow_parts.append("Update Salary")
        if not contact_email or not safe(r.get("organization_id")) or not org_name:
            follow_parts.append("Update Contact Details")
        if isinstance(openings_val, int) and n_shortlisted > (n_rejected + openings_val):
            follow_parts.append("Shortlist / Reject Applications")
        follow_up = ", ".join(follow_parts)

        d = {
            "id":                              safe(r.get("id")),
            "location":                        safe(r.get("location")),
            "organization_name":               org_name,
            "Role":                            role_val,
            "Openings":                        openings_val,
            "Salary":                          salary_str,
            "Job Description":                 job_desc,
            "Contact Name of created_by id":   contact_name,
            "Contact Number of created_by id": contact_num,
            "Contact email ID":                contact_email,
            "organisation id":                 safe(r.get("organization_id")),
            "organisation name":               org_name,
            "Job Post Date":                   fmt_date(r.get("created_at")),
            "Applications":                    n_apps,
            "Shortlisted":                     n_shortlisted,
            "Rejected":                        n_rejected,
            "Current Openings (":              openings_val,
            "Duplicate":                       duplicate,
            "Job Sector":                      safe(r.get("llm_role_sector")),
            "Job Age":                         job_age,
            "Right Fit Seekers Count":         right_fit_counts.get(jid, 0),
            "Last Application Date":           last_app_str,
            "Application Pending From":        app_pending,
            "Shortlisted Age":                 shortlisted_age,
            "Rejected Age":                    rejected_age,
            "Recommended Action for Job Provider": rec_provider,
            "Recommended Action for Job Seeker":   rec_seeker,
            "Request_type":                    request_type,
            "Profile Completion":              profile_completion,
            "Follow up for":                   follow_up,
            "Status":                          job_status,
            "normalised_role":                 safe(r.get("llm_role_normalized")),
            "location_state":                  safe(r.get("llm_location_state")),
            "location_district":               safe(r.get("llm_location_district")),
            "test":                            to_num(r.get("test_flag")),
            "Partial Fit Seeker Count":        partial_fit_counts.get(jid, 0),
            "title":                           safe(r.get("title")),
            "normalised_title":                safe(r.get("llm_title_normalized")),
            "normalised_title_sector":         safe(r.get("llm_title_sector")),
        }
        rows.append(row_from_dict(d, headers))
    return rows


# ── job_application ────────────────────────────────────────────────────────────

def build_job_application(data: dict, headers: list[str]) -> list[list]:
    job_application = data["job_application"]
    job_posting     = data["job_posting"]
    org_provider    = data["org_provider"]
    user_seeker     = data["user_seeker"]
    user_provider   = data["user_provider"]
    profile         = data["profile"]
    member_seeker   = data["member_seeker"]
    org_seeker      = data["org_seeker"]

    job_lkp = (
        job_posting.drop_duplicates("id")
        .set_index("id")[["organization_name", "organization_id", "created_by",
                          "llm_location_state", "llm_location_district"]]
        .to_dict("index")
    )
    org_prov_lkp = (
        org_provider.drop_duplicates("id")
        .set_index("id")[["contact_person", "contact_phone", "contact_email"]]
        .to_dict("index")
    )
    user_prov_lkp = (
        user_provider.drop_duplicates("id")
        .set_index("id")[["phone_number", "email"]]
        .to_dict("index")
    )
    user_sk_lkp = (
        user_seeker.drop_duplicates("id")
        .set_index("id")[["phone_number", "email"]]
        .to_dict("index")
    )
    profile_lkp = (
        profile.drop_duplicates("user_id")
        .set_index("user_id")[["llm_location_district", "llm_location_state", "iti_institute"]]
        .to_dict("index")
    )
    # job_application.user_id == profile.id; bridge to profile.user_id (Firebase ID)
    # which is the key used in member_seeker
    prof_id_to_uid = (
        profile.drop_duplicates("id")
        .dropna(subset=["id", "user_id"])
        .set_index("id")["user_id"]
        .to_dict()
    )
    member_org_lkp = (
        member_seeker.dropna(subset=["user_id"])
        .groupby("user_id")["organization_id"].first()
        .to_dict()
    )
    org_sk_lkp = (
        org_seeker.drop_duplicates("id")
        .set_index("id")["name"]
        .to_dict()
    )

    rows = []
    for _, r in job_application.iterrows():
        uid = safe(r.get("user_id"))
        jid = safe(r.get("job_id"))

        job      = job_lkp.get(jid, {})
        org_id   = safe(job.get("organization_id"))
        org_prov = org_prov_lkp.get(org_id, {})
        u_prov   = user_prov_lkp.get(safe(job.get("created_by")), {})
        u_sk     = user_sk_lkp.get(uid, {})
        prof     = profile_lkp.get(uid, {})

        # Resolve seeker org: job_app.user_id -> profile.id -> profile.user_id -> member -> org
        prof_uid        = prof_id_to_uid.get(uid, "")
        seeker_org_id   = safe(member_org_lkp.get(prof_uid, ""))
        seeker_org_name = safe(org_sk_lkp.get(seeker_org_id, ""))

        raw_phone = safe(r.get("seeker_phone"))

        d = {
            "id":                    safe(r.get("id")),
            "job_id":                safe(r.get("job_id")),
            "transaction_id":        safe(r.get("transaction_id")),
            "status":                safe(r.get("status")),
            "application_status":    safe(r.get("application_status")),
            "user_name":             safe(r.get("user_name")),
            "user_id":               uid,
            "location":              safe(r.get("location")),
            "contact":               safe(r.get("contact")),
            "metadata":              safe(r.get("metadata")),
            "applied_at":            fmt_date(r.get("applied_at")),
            "updated_at":            fmt_date(r.get("updated_at")),
            "Contact Number":        raw_phone,
            "Formatted Phone Number": clean_phone(raw_phone),
            "Company Name":          safe(job.get("organization_name")),
            "Location":              safe(r.get("location")),
            "Test Flag":             to_num(r.get("test_flag")),
            "user_phone":            safe(u_prov.get("phone_number")),
            "user_email":            safe(u_prov.get("email")),
            "user_district":         safe(job.get("llm_location_district")),
            "user_state":            safe(job.get("llm_location_state")),
            "job role":              safe(r.get("job_role")),
            "institution name":      safe(prof.get("iti_institute")),
            "institution id":        "",
            "contact name":          safe(org_prov.get("contact_person")),
            "contact number":        safe(org_prov.get("contact_phone")),
            "contact email":         safe(org_prov.get("contact_email")),
            "seeker_user_id":        seeker_org_id,
            "Organisation":          seeker_org_name,
            "request_type":          "",
            "title":                 safe(r.get("job_title")),
            "normalised_title":      safe(r.get("llm_job_title_normalized")),
            "normalised_title_sector": safe(r.get("llm_job_title_sector")),
        }
        rows.append(row_from_dict(d, headers))
    return rows


# ── seeker_profile ─────────────────────────────────────────────────────────────

def build_seeker_profile(data: dict, headers: list[str]) -> list[list]:
    TODAY         = _dt.today().date()
    profile       = data["profile"]
    user_seeker   = data["user_seeker"]
    member_seeker = data["member_seeker"]
    org_seeker    = data["org_seeker"]
    job_app_df    = data["job_application"]
    sk_right   = data.get("sk_right_fit_counts",   {})
    sk_partial = data.get("sk_partial_fit_counts", {})

    user_email_lkp = (
        user_seeker.drop_duplicates("id")
        .set_index("id")["email"]
        .to_dict()
    )
    member_org_lkp = (
        member_seeker.dropna(subset=["user_id"])
        .groupby("user_id")["organization_id"].first()
        .to_dict()
    )
    org_name_lkp = (
        org_seeker.drop_duplicates("id")
        .set_index("id")["name"]
        .to_dict()
    )

    # Application lookup keyed by normalised phone (strip trailing .0)
    # job_application.user_id does not overlap with seeker profile.user_id;
    # phone is the reliable shared key between the two datasets.
    def _norm_phone(val) -> str:
        return re.sub(r"\.0$", "", safe(val)).strip()

    app_by_phone: dict = {}
    for _, a in job_app_df.iterrows():
        phone = _norm_phone(a.get("seeker_phone"))
        if phone:
            app_by_phone.setdefault(phone, []).append({
                "applied_at_dt":      _parse_date(a.get("applied_at")),
                "application_status": safe(a.get("application_status")).lower(),
            })

    rows = []
    for _, r in profile.iterrows():
        uid      = safe(r.get("user_id"))
        lat, lng = parse_gps(r.get("metadata"))
        org_id   = safe(member_org_lkp.get(uid, ""))
        org_name = safe(org_name_lkp.get(org_id, ""))

        # Base field values
        name_val   = safe(r.get("name"))
        loc_blob   = safe(r.get("current_location"))
        loc_dist   = safe(r.get("llm_location_district"))
        loc_state  = safe(r.get("llm_location_state"))
        email_val  = safe(user_email_lkp.get(uid, ""))
        phone_val  = safe(r.get("phone"))
        trade_val  = safe(r.get("iti_specialization"))
        inst_val   = safe(r.get("iti_institute"))
        role_val   = safe(r.get("role"))
        age_val    = to_num(r.get("age"))
        salary_val = to_num(r.get("monthly_in_hand_preferred"))
        qual_val   = safe(r.get("highest_qualification"))

        # Application counts — link via normalised phone number
        phone_key     = _norm_phone(r.get("phone"))
        apps          = app_by_phone.get(phone_key, [])
        n_apps        = len(apps)
        n_shortlisted = sum(1 for a in apps if a["application_status"] == "shortlisted")
        n_rejected    = sum(1 for a in apps if a["application_status"] == "rejected")

        # Created_On and Profile Age
        created_dt  = _parse_date(r.get("created_at"))
        created_on  = f"{created_dt.month}/{created_dt.day}/{created_dt.year}" if created_dt else ""
        profile_age = (TODAY - created_dt.date()).days if created_dt else 0

        # Last Applied Age (days since most recent application; blank if never applied)
        applied_dts      = [a["applied_at_dt"] for a in apps if a["applied_at_dt"]]
        last_applied_dt  = max(applied_dts) if applied_dts else None
        last_applied_date = fmt_date(last_applied_dt)
        last_applied_age = (TODAY - last_applied_dt.date()).days if last_applied_dt else ""

        # Status
        if profile_age <= 7:
            status = "New"
        elif last_applied_age != "" and last_applied_age <= 30:
            status = "Active"
        elif profile_age > 7 and last_applied_age != "" and 31 <= last_applied_age <= 90:
            status = "At Risk"
        elif profile_age > 7 and (last_applied_age == "" or last_applied_age > 90):
            status = "Inactive"
        else:
            status = ""

        # Profile Completion (cols C,D,G:N,P:R → 13 fields)
        pc_fields = [
            name_val, loc_blob,                                    # C, D
            loc_dist, loc_state, "", email_val, phone_val,         # G–K
            trade_val, inst_val, role_val,                         # L–N
            str(age_val) if age_val != "" else "",                 # P
            str(salary_val) if salary_val != "" else "",           # Q
            qual_val,                                              # R
        ]
        profile_completion = round(sum(1 for v in pc_fields if v) / 13, 4)

        # Follow up for (mirrors sheet formula column refs)
        follow_parts = []
        if not uid:                                          # B = user_id
            follow_parts.append("Update Name")
        if not name_val or lat == "" or not "":              # C=name, F=lat, I=pincode
            follow_parts.append("Update Location")
        if not phone_val and not trade_val:                  # K=phone AND L=trade both empty
            follow_parts.append("Update Contact Details")
        if age_val == "":                                    # P = Age
            follow_parts.append("Update Role")
        follow_up = ", ".join(follow_parts)

        d = {
            "id":                 safe(r.get("id")),
            "user_id":            uid,
            "name":               name_val,
            "location":           loc_blob,
            "longitude":          lng,
            "latitude":           lat,
            "location_district ": loc_dist,
            "location_state":     loc_state,
            "pincode":            "",
            "email id":           email_val,
            "Phone Number":       phone_val,
            "trade":              trade_val,
            "institution name":   inst_val,
            "Role":               role_val,
            "test":               to_num(r.get("test_flag")),
            "Age":                age_val,
            "Expected Salary":    salary_val,
            "Qualification":      qual_val,
            "Org id":             org_id,
            "Org Name":           org_name,
            "Recommended Action": "Kaam Ki Baat" if safe(r.get("id")) else "",
            "Applications":       n_apps,
            "Shortlisted":        n_shortlisted,
            "Rejected":           n_rejected,
            "Profile Completion": profile_completion,
            "Follow up for":      follow_up,
            "Created_On":         created_on,
            "Profile Age":        profile_age,
            "Last Applied Age":   last_applied_age,
            "Last Application Date": last_applied_date,
            "Status":             status,
            "Partial Fit":        sk_partial.get(uid, 0),
            "Direct Fit":         sk_right.get(uid, 0),
        }
        rows.append(row_from_dict(d, headers))
    return rows


# ── Main ───────────────────────────────────────────────────────────────────────

DATASETS = {
    "UP": {
        "load": load_data,
        "tabs": {
            "OPEN_ROLES":      (build_open_roles,      SHEET_ID_UP_PROVIDER),
            "job_application": (build_job_application,  SHEET_ID_UP_PROVIDER),
            "seeker_profile":  (build_seeker_profile,   SHEET_ID_UP_SEEKER),
        },
    },
    "KA": {
        "load": load_ka_data,
        "tabs": {
            "OPEN_ROLES":      (build_open_roles,      SHEET_ID_KA_PROVIDER),
            "job_application": (build_job_application,  SHEET_ID_KA_PROVIDER),
            "seeker_profile":  (build_seeker_profile,   SHEET_ID_KA_SEEKER),
        },
    },
}

# backwards-compat: TABS still points to UP for any external callers
TABS = DATASETS["UP"]["tabs"]


def _get_gspread_client() -> gspread.Client:
    """Load Google credentials from Secret Manager (cloud) or local file (dev)."""
    if CREDS_FILE.exists():
        return gspread.service_account(filename=str(CREDS_FILE))
    # Running in Cloud Run — load from Secret Manager
    import json
    from google.oauth2.service_account import Credentials
    from google.cloud import secretmanager
    sm = secretmanager.SecretManagerServiceClient()
    name = "projects/blue-dots-project/secrets/GOOGLE_SERVICE_ACCOUNT/versions/latest"
    secret = sm.access_secret_version(request={"name": name})
    info = json.loads(secret.payload.data.decode("utf-8"))
    scopes = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    return gspread.authorize(creds)


HISTORY_HEADERS = [
    "run_timestamp",
    "ka_seekers", "ka_job_postings", "ka_job_applications", "ka_matches",
    "up_seekers", "up_job_postings", "up_job_applications", "up_matches",
]


def _count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return len(pd.read_csv(path, usecols=[0], low_memory=False))
    except Exception:
        return 0


def _append_run_summary(gc: gspread.Client) -> None:
    from datetime import datetime, timezone
    sh  = gc.open_by_key(SHEET_ID_SUMMARY)
    try:
        ws = sh.worksheet("Pipeline History")
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet("Pipeline History", rows=1000, cols=len(HISTORY_HEADERS))
        ws.append_row(HISTORY_HEADERS)

    p = PROCESSED_DIR
    row = [
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        _count(p / "ka_seeker"   / "profile_clean.csv"),
        _count(p / "ka_provider" / "job_posting_clean.csv"),
        _count(p / "ka_provider" / "job_application_clean.csv"),
        _count(p / "matches"     / "ka_matches.csv"),
        _count(p / "up_seeker"   / "profile_clean.csv"),
        _count(p / "up_provider" / "job_posting_clean.csv"),
        _count(p / "up_provider" / "job_application_clean.csv"),
        _count(p / "matches"     / "up_matches.csv"),
    ]
    ws.append_row(row)
    print(f"  Run summary appended: {row}")


def main(resume: bool = False, only_state: str | None = None, test: bool = False) -> None:
    gc = _get_gspread_client()
    state = load_state() if resume else {}

    if test:
        print("*** TEST MODE: uploading to test sheets, not production ***")
        active_datasets = {
            "UP": {
                "load": load_data,
                "tabs": {
                    "OPEN_ROLES":      (build_open_roles,     TEST_SHEET_ID_UP_PROVIDER),
                    "job_application": (build_job_application, TEST_SHEET_ID_UP_PROVIDER),
                    "seeker_profile":  (build_seeker_profile,  TEST_SHEET_ID_UP_SEEKER),
                },
            },
            "KA": {
                "load": load_ka_data,
                "tabs": {
                    "OPEN_ROLES":      (build_open_roles,     TEST_SHEET_ID_KA_PROVIDER),
                    "job_application": (build_job_application, TEST_SHEET_ID_KA_PROVIDER),
                    "seeker_profile":  (build_seeker_profile,  TEST_SHEET_ID_KA_SEEKER),
                },
            },
        }
    else:
        active_datasets = DATASETS

    datasets_to_run = (
        {only_state: active_datasets[only_state]} if only_state else active_datasets
    )

    for ds_name, ds in datasets_to_run.items():
        print(f"\n{'='*50}")
        print(f"Loading {ds_name} data...")
        data = ds["load"]()

        # Open every unique sheet ID needed for this dataset
        sheet_ids = {sid for _, sid in ds["tabs"].values()}
        sheets = {sid: gc.open_by_key(sid) for sid in sheet_ids}

        for tab_name, (build_fn, sheet_id) in ds["tabs"].items():
            state_key = f"{ds_name}_{tab_name}"
            if resume and state.get(state_key, {}).get("done"):
                print(f"\n{ds_name}/{tab_name}: already done (--resume), skipping.")
                continue

            print(f"\nBuilding {ds_name}/{tab_name}...")
            sh = sheets[sheet_id]
            ws = get_or_create_ws(sh, tab_name)

            # Read headers dynamically from the sheet's first row
            raw = ws.get("1:1")
            headers = raw[0] if raw else []
            while headers and headers[-1] == "":
                headers.pop()
            if not headers:
                print(f"  WARNING: no headers found in {tab_name}, skipping.")
                continue

            rows = build_fn(data, headers)
            print(f"  {len(rows):,} rows")
            upload_tab(ws, headers, rows)

            state[state_key] = {"done": True, "rows": len(rows)}
            save_state(state)

    if test:
        print("\nSkipping Pipeline History update (test mode).")
    else:
        print("\nUpdating Pipeline History...")
        _append_run_summary(gc)

    print("\nAll done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tabs already marked as done in .pipeline_state.json",
    )
    parser.add_argument(
        "--state",
        choices=["UP", "KA"],
        default=None,
        help="Only run for one state (UP or KA). Omit to run both.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Upload to test sheets instead of production sheets.",
    )
    args = parser.parse_args()
    main(resume=args.resume, only_state=args.state, test=args.test)
