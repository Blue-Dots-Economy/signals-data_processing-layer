"""
matchmaking.py

Matches seeker profiles against open job postings for UP and KA datasets.

Matching criteria (weights):
  - Role match     (40%): seeker iti_specialization vs provider llm_title_normalized (fuzzy)
  - Location match (30%): seeker vs provider state / district
  - Salary match   (20%): seeker monthly_in_hand_preferred vs provider min/max salary
  - Education match(10%): seeker highest_qualification vs provider min_education_level

Fit classification:
  - Right Fit:   role match (>=50) AND location match (>=50)
  - Partial Fit: role match (>=50) only

Only pairs with overall confidence >= 40% are written to output.
Test-flagged records and non-open jobs are excluded.

Outputs: processed_data/matches/up_matches.csv  &  processed_data/matches/ka_matches.csv

Usage:
    python matchmaking.py             # both states
    python matchmaking.py --state KA  # KA only
    python matchmaking.py --state UP  # UP only
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

BASE_DIR      = Path(__file__).parent.parent   # repo root (preprocessing/../)
PROCESSED_DIR = BASE_DIR / "processed_data"
MATCHES_DIR   = PROCESSED_DIR / "matches"

CONFIDENCE_THRESHOLD = 0.40   # minimum weighted confidence to emit a row
ROLE_MIN_SCORE       = 0.30   # pre-filter: skip pairs below this before full scoring
ROLE_MATCH_THRESHOLD = 0.50   # minimum role score to count as a "role match"
LOC_MATCH_THRESHOLD  = 0.50   # minimum location score to count as a "location match"

WEIGHTS = {"role": 0.40, "location": 0.30, "salary": 0.20, "education": 0.10}

EDU_HIERARCHY = {
    "8th pass": 1, "8th": 1,
    "10th pass": 2, "10th": 2, "ssc": 2, "matriculation": 2,
    "12th pass": 3, "12th": 3, "hsc": 3, "intermediate": 3,
    "iti": 4, "diploma": 4, "iti/diploma": 4, "polytechnic": 4,
    "graduate": 5, "graduation": 5, "bachelor": 5,
    "b.tech": 5, "b.sc": 5, "b.com": 5, "b.a": 5, "be": 5,
    "postgraduate": 6, "post graduate": 6, "master": 6, "m.tech": 6, "mba": 6,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def safe(val) -> str:
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val).strip()


def to_float(val):
    s = safe(val)
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


# ── Role scoring ───────────────────────────────────────────────────────────────

def _single_role_score(a: str, b: str) -> float:
    """Fuzzy token-set ratio between two role strings, normalised to 0–1."""
    if not a or not b:
        return 0.0
    if a.lower() == b.lower():
        return 1.0
    return fuzz.token_set_ratio(a, b) / 100.0


def role_score_for_pair(sk_iti_str: str, pr_role: str) -> tuple[float, str]:
    """
    Seeker may have pipe-separated trades (e.g. 'Fitter|Electrician').
    Return (max_score, best_matching_trade).
    """
    trades = [t.strip() for t in sk_iti_str.split("|") if t.strip()]
    if not trades:
        return 0.0, ""
    scores = [(_single_role_score(t, pr_role), t) for t in trades]
    return max(scores, key=lambda x: x[0])


def build_role_matrix(sk_iti_values: list[str], pr_roles: list[str]) -> pd.DataFrame:
    """
    Precompute role scores for all unique (sk_iti_str, pr_role) pairs.
    Returns DataFrame: sk_iti_str | pr_role | role_score | best_trade
    """
    records = []
    for sk_iti, pr_role in itertools.product(sk_iti_values, pr_roles):
        score, best_trade = role_score_for_pair(sk_iti, pr_role)
        if score >= ROLE_MIN_SCORE:
            records.append((sk_iti, pr_role, round(score, 4), best_trade))
    return pd.DataFrame(records, columns=["sk_iti_str", "pr_role", "role_score", "best_trade"])


# ── Location scoring ───────────────────────────────────────────────────────────

def _loc_score(sk_state: str, sk_dist: str, pr_state: str, pr_dist: str):
    """
    Returns float 0–1, or np.nan if provider has no location (neutral).
    District match = 1.0 | State match only = 0.6 | No match = 0.0
    """
    p_state = pr_state.lower()
    s_state = sk_state.lower()
    if not p_state:
        return np.nan   # provider didn't specify — neutral
    if not s_state:
        return np.nan   # seeker has no location — neutral
    if p_state not in s_state and s_state not in p_state:
        return 0.0
    p_dist = pr_dist.lower()
    s_dist = sk_dist.lower()
    if not p_dist or not s_dist:
        return 0.6      # state matches, no district info
    if p_dist in s_dist or s_dist in p_dist:
        return 1.0
    return 0.6


def vec_loc_score(df: pd.DataFrame) -> pd.Series:
    return df.apply(
        lambda r: _loc_score(
            safe(r["sk_location_state"]), safe(r["sk_location_district"]),
            safe(r["pr_location_state"]),  safe(r["pr_location_district"]),
        ),
        axis=1,
    )


# ── Salary scoring ─────────────────────────────────────────────────────────────

def _sal_score(sk_sal, pr_min, pr_max):
    """
    Returns float 0–1, or np.nan if either side is missing (neutral).
    Within range = 1.0 | Within 20% of range = 0.5 | Outside = 0.0
    """
    if sk_sal is None or (pr_min is None and pr_max is None):
        return np.nan
    lo = pr_min if pr_min is not None else pr_max * 0.8
    hi = pr_max if pr_max is not None else pr_min * 1.2
    if lo > hi:
        lo, hi = hi, lo
    if lo <= sk_sal <= hi:
        return 1.0
    buf = max((hi - lo) * 0.2, 2000)
    if (lo - buf) <= sk_sal <= (hi + buf):
        return 0.5
    return 0.0


def vec_sal_score(df: pd.DataFrame) -> pd.Series:
    return df.apply(
        lambda r: _sal_score(
            to_float(r["monthly_in_hand_preferred"]),
            to_float(r["min_monthly_in_hand"]),
            to_float(r["max_monthly_in_hand"]),
        ),
        axis=1,
    )


# ── Education scoring ──────────────────────────────────────────────────────────

def _edu_level(qual_str: str):
    """Return highest education level int from a pipe-separated qualification string."""
    if not qual_str:
        return None
    levels = []
    for part in qual_str.lower().split("|"):
        part = part.strip()
        if part in EDU_HIERARCHY:
            levels.append(EDU_HIERARCHY[part])
            continue
        for key, val in EDU_HIERARCHY.items():
            if key in part:
                levels.append(val)
                break
    return max(levels) if levels else None


def _edu_score(sk_qual: str, pr_min_edu: str):
    """
    Returns float 0–1, or np.nan if either side is missing (neutral).
    Meets/exceeds = 1.0 | One level below = 0.5 | Further below = 0.0
    """
    if not pr_min_edu:
        return np.nan
    s_lv = _edu_level(sk_qual)
    p_lv = _edu_level(pr_min_edu)
    if s_lv is None or p_lv is None:
        return np.nan
    diff = s_lv - p_lv
    if diff >= 0:
        return 1.0
    if diff == -1:
        return 0.5
    return 0.0


def vec_edu_score(df: pd.DataFrame) -> pd.Series:
    return df.apply(
        lambda r: _edu_score(safe(r["highest_qualification"]), safe(r["min_education_level"])),
        axis=1,
    )


# ── Confidence ─────────────────────────────────────────────────────────────────

def vec_confidence(df: pd.DataFrame) -> pd.Series:
    """Weighted average of available (non-NaN) score columns."""
    cols = {"role_score": "role", "location_score": "location",
            "salary_score": "salary", "education_score": "education"}
    result = []
    for _, r in df.iterrows():
        total_w, wsum = 0.0, 0.0
        for col, key in cols.items():
            v = r[col]
            if not (v != v):  # not NaN
                w = WEIGHTS[key]
                wsum  += v * w
                total_w += w
        result.append(wsum / total_w if total_w > 0 else 0.0)
    return pd.Series(result, index=df.index)


def confidence_label(score: float) -> str:
    if score >= 0.80: return "High"
    if score >= 0.60: return "Medium"
    if score >= 0.40: return "Low"
    return "Very Low"


# ── Main matching ──────────────────────────────────────────────────────────────

def match_dataset(provider_dir: Path, seeker_dir: Path, state: str) -> pd.DataFrame:
    print(f"\n{'='*50}")
    print(f"[{state}] Loading data...")

    job_posting   = pd.read_csv(provider_dir / "job_posting_clean.csv",  low_memory=False)
    user_provider = pd.read_csv(provider_dir / "user_clean.csv",          low_memory=False)
    org_provider  = pd.read_csv(provider_dir / "organization_clean.csv",  low_memory=False)
    profile       = pd.read_csv(seeker_dir   / "profile_clean.csv",       low_memory=False)
    user_seeker   = pd.read_csv(seeker_dir   / "user_clean.csv",          low_memory=False)
    member_seeker = pd.read_csv(seeker_dir   / "member_clean.csv",        low_memory=False)
    org_seeker    = pd.read_csv(seeker_dir   / "organization_clean.csv",  low_memory=False)

    # ── Filter ────────────────────────────────────────────────────────────────
    open_jobs = job_posting[
        (job_posting["status"].str.strip().str.lower() == "open") &
        (job_posting["test_flag"].fillna(0).astype(float) == 0)
    ].copy()

    seekers = profile[
        profile["test_flag"].fillna(0).astype(str).str.strip().isin(["0", "0.0"])
    ].copy()

    print(f"[{state}] {len(open_jobs)} open jobs | {len(seekers)} seekers")

    # ── Lookup tables ─────────────────────────────────────────────────────────
    user_sk_lkp = (
        user_seeker.drop_duplicates("id")
        .set_index("id")[["phone_number", "email"]]
        .to_dict("index")
    )
    member_org_lkp = (
        member_seeker.dropna(subset=["user_id"])
        .groupby("user_id")["organization_id"].first()
        .to_dict()
    )
    org_sk_lkp = org_seeker.drop_duplicates("id").set_index("id")["name"].to_dict()
    user_pr_lkp = (
        user_provider.drop_duplicates("id")
        .set_index("id")[["name", "phone_number", "email"]]
        .to_dict("index")
    )

    # ── Seeker role field (iti_specialization → role) ────────────────────────
    seekers["sk_iti_str"] = seekers["iti_specialization"].apply(safe)
    seekers.loc[seekers["sk_iti_str"] == "", "sk_iti_str"] = seekers.loc[
        seekers["sk_iti_str"] == "", "role"
    ].apply(safe)

    # Provider role: llm_title_normalized -> llm_job_title_normalized -> job_title -> title -> role (fallback chain)
    def _pr_role(r):
        return (safe(r.get("llm_title_normalized"))
                or safe(r.get("llm_job_title_normalized"))
                or safe(r.get("job_title"))
                or safe(r.get("title"))
                or safe(r.get("role")))
    open_jobs["pr_role"] = open_jobs.apply(_pr_role, axis=1)

    # ── Precompute role matrix ────────────────────────────────────────────────
    sk_iti_values = [v for v in seekers["sk_iti_str"].unique() if v]
    pr_roles      = [v for v in open_jobs["pr_role"].unique()   if v]

    print(f"[{state}] Precomputing {len(sk_iti_values)} × {len(pr_roles)} role pairs "
          f"({len(sk_iti_values)*len(pr_roles):,} total)...")

    role_matrix = build_role_matrix(sk_iti_values, pr_roles)
    print(f"[{state}] {len(role_matrix):,} role pairs pass the {ROLE_MIN_SCORE:.0%} threshold")

    if role_matrix.empty:
        print(f"[{state}] No role matches found. Skipping.")
        return pd.DataFrame()

    # ── Slim down before the big join ─────────────────────────────────────────
    sk_cols = [
        "id", "user_id", "name", "phone", "sk_iti_str",
        "iti_specialization", "role",
        "llm_location_state", "llm_location_district",
        "monthly_in_hand_preferred", "highest_qualification",
        "test_flag",
    ]
    pr_cols = [
        "id", "organization_name", "organization_id", "created_by",
        "pr_role", "role", "title", "job_title", "llm_title_normalized", "llm_job_title_normalized",
        "llm_location_state", "llm_location_district",
        "min_monthly_in_hand", "max_monthly_in_hand",
        "min_education_level", "positions",
    ]
    sk_slim = seekers[[c for c in sk_cols if c in seekers.columns]].copy()
    pr_slim = open_jobs[[c for c in pr_cols if c in open_jobs.columns]].copy()

    # ── Join: seekers → role_matrix → jobs ───────────────────────────────────
    merged = (
        sk_slim
        .merge(role_matrix, on="sk_iti_str", how="inner")
        .merge(pr_slim, on="pr_role", how="inner", suffixes=("_sk", "_pr"))
    )
    print(f"[{state}] {len(merged):,} candidate pairs after role pre-filter")

    if merged.empty:
        return pd.DataFrame()

    # Rename location cols for clarity
    merged.rename(columns={
        "llm_location_state_sk":    "sk_location_state",
        "llm_location_district_sk": "sk_location_district",
        "llm_location_state_pr":    "pr_location_state",
        "llm_location_district_pr": "pr_location_district",
    }, inplace=True)

    # ── Score all four criteria ───────────────────────────────────────────────
    print(f"[{state}] Scoring location, salary, education...")
    merged["location_score"]  = vec_loc_score(merged)
    merged["salary_score"]    = vec_sal_score(merged)
    merged["education_score"] = vec_edu_score(merged)

    # ── Confidence & threshold filter ─────────────────────────────────────────
    merged["confidence"] = vec_confidence(merged)
    merged = merged[merged["confidence"] >= CONFIDENCE_THRESHOLD].copy()
    print(f"[{state}] {len(merged):,} pairs above {CONFIDENCE_THRESHOLD:.0%} confidence")

    if merged.empty:
        return pd.DataFrame()

    # ── Fit category ──────────────────────────────────────────────────────────
    role_ok = merged["role_score"] >= ROLE_MATCH_THRESHOLD
    loc_ok  = merged["location_score"].fillna(-1) >= LOC_MATCH_THRESHOLD

    merged["match_category"] = np.where(
        role_ok & loc_ok, "Right Fit",
        np.where(role_ok, "Partial Fit", None)
    )
    merged = merged[merged["match_category"].notna()].copy()
    print(f"[{state}] {len(merged):,} final matches  "
          f"({(merged['match_category']=='Right Fit').sum():,} Right Fit  |  "
          f"{(merged['match_category']=='Partial Fit').sum():,} Partial Fit)")

    # ── Enrich with lookup data ───────────────────────────────────────────────
    def enrich(r):
        uid    = safe(r.get("user_id"))
        cb     = safe(r.get("created_by"))
        org_id = safe(member_org_lkp.get(uid, ""))
        u_sk   = user_sk_lkp.get(uid, {})
        u_pr   = user_pr_lkp.get(cb, {})
        return pd.Series({
            "seeker_email":           safe(u_sk.get("email")),
            "seeker_org_id":          org_id,
            "seeker_org_name":        safe(org_sk_lkp.get(org_id, "")),
            "provider_contact_name":  safe(u_pr.get("name")),
            "provider_contact_phone": safe(u_pr.get("phone_number")),
            "provider_contact_email": safe(u_pr.get("email")),
        })

    extra = merged.apply(enrich, axis=1)
    merged = pd.concat([merged, extra], axis=1)

    # ── Build final output DataFrame ──────────────────────────────────────────
    def pct(v):
        return round(v * 100, 1) if (v == v) else "N/A"   # NaN → N/A

    out = pd.DataFrame({
        # ── IDs & classification ──────────────────────────────────────────
        "seeker_id":               merged["user_id"].apply(safe),
        "provider_job_id":         merged["id_pr"].apply(safe),
        "match_category":          merged["match_category"],
        "confidence_score_%":      (merged["confidence"] * 100).round(1),
        "confidence_label":        merged["confidence"].apply(confidence_label),

        # ── Match indicators ──────────────────────────────────────────────
        "role_match_%":            merged["role_score"].apply(lambda v: round(v*100,1)),
        "location_match_%":        merged["location_score"].apply(pct),
        "salary_match_%":          merged["salary_score"].apply(pct),
        "education_match_%":       merged["education_score"].apply(pct),

        # ── Job role (which role this pairing corresponds to) ─────────────
        "job_role":                merged["role_pr"].apply(safe),
        "job_title":               merged["title"].apply(safe) if "title" in merged.columns else "",
        "job_normalised_title":    merged["pr_role"].apply(safe),

        # ── Seeker profile ────────────────────────────────────────────────
        "seeker_name":             merged["name"].apply(safe),
        "seeker_phone":            merged["phone"].apply(safe),
        "seeker_email":            merged["seeker_email"],
        "seeker_matched_trade":    merged["best_trade"],
        "seeker_iti_full":         merged["iti_specialization"].apply(safe) if "iti_specialization" in merged.columns else "",
        "seeker_role":             merged["role_sk"].apply(safe),
        "seeker_location_district": merged["sk_location_district"].apply(safe),
        "seeker_location_state":   merged["sk_location_state"].apply(safe),
        "seeker_expected_salary":  merged["monthly_in_hand_preferred"].apply(safe),
        "seeker_qualification":    merged["highest_qualification"].apply(safe),
        "seeker_org_id":           merged["seeker_org_id"],
        "seeker_org_name":         merged["seeker_org_name"],

        # ── Provider / job details ────────────────────────────────────────
        "provider_org_name":       merged["organization_name"].apply(safe),
        "provider_org_id":         merged["organization_id"].apply(safe),
        "job_location_district":   merged["pr_location_district"].apply(safe),
        "job_location_state":      merged["pr_location_state"].apply(safe),
        "job_salary_min":          merged["min_monthly_in_hand"].apply(safe) if "min_monthly_in_hand" in merged.columns else "",
        "job_salary_max":          merged["max_monthly_in_hand"].apply(safe) if "max_monthly_in_hand" in merged.columns else "",
        "job_min_education":       merged["min_education_level"].apply(safe) if "min_education_level" in merged.columns else "",
        "job_openings":            merged["positions"].apply(safe) if "positions" in merged.columns else "",
        "provider_contact_name":   merged["provider_contact_name"],
        "provider_contact_phone":  merged["provider_contact_phone"],
        "provider_contact_email":  merged["provider_contact_email"],
    })

    out.sort_values("confidence_score_%", ascending=False, inplace=True)
    return out


# ── Entry point ────────────────────────────────────────────────────────────────

def main(only_state: str | None = None):
    MATCHES_DIR.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("UP", PROCESSED_DIR / "up_provider", PROCESSED_DIR / "up_seeker"),
        ("KA", PROCESSED_DIR / "ka_provider", PROCESSED_DIR / "ka_seeker"),
    ]
    if only_state:
        datasets = [(s, p, k) for s, p, k in datasets if s == only_state]

    for state, provider_dir, seeker_dir in datasets:
        df = match_dataset(provider_dir, seeker_dir, state)
        out_path = MATCHES_DIR / f"{state.lower()}_matches.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[{state}] Written to {out_path}  ({len(df):,} rows)")

    print("\nMatchmaking complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", choices=["UP", "KA"], default=None,
                        help="Only run for one state. Omit to run both.")
    args = parser.parse_args()
    main(only_state=args.state)
