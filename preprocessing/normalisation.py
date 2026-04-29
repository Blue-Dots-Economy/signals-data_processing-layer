"""
normalisation.py
Extracts, flattens, standardises, and cleans all raw CSV data.
Keeps Karnataka (KA) and Uttar Pradesh (UP) data separate.
Output: processed_data/ folder with subfolders ka_seeker, up_seeker,
        ka_provider, up_provider.
Lists are joined with "|".  Phone numbers are standardised to +91XXXXXXXXXX.
"""

import json
import re
import pandas as pd
from pathlib import Path

# ── Claude API key ─────────────────────────────────────────────────────────────
# Set your Anthropic API key here before running.
CLAUDE_API_KEY: str | None = None   # e.g. "sk-ant-..."

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
RAW_DIR  = BASE_DIR / "raw_data"
OUT_DIR  = BASE_DIR / "processed_data"

# ── Generic helpers ────────────────────────────────────────────────────────────

def parse_json(val) -> dict:
    """Safely parse a JSON string; returns {} on failure."""
    if val is None:
        return {}
    try:
        if pd.isna(val):
            return {}
    except (TypeError, ValueError):
        pass
    if isinstance(val, dict):
        return val
    if val in ("", "{}", "[]"):
        return {}
    try:
        result = json.loads(val)
        return result if isinstance(result, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def ensure_dict(val) -> dict:
    """Ensure a value is a dict, parsing it from JSON if needed."""
    if isinstance(val, dict):
        return val
    if val is None or val == "":
        return {}
    return parse_json(val)


def clean_str(val) -> str | None:
    """Strip whitespace; return None for blank/NA values."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    return s if s else None


def clean_phone(val) -> str | None:
    """Normalise to +91XXXXXXXXXX for Indian numbers; preserve others."""
    if not val:
        return None
    s = str(val).strip()
    if not s:
        return None
    digits = re.sub(r"\D", "", s)
    if len(digits) == 10:
        return f"+91{digits}"
    if len(digits) == 12 and digits.startswith("91"):
        return f"+{digits}"
    if len(digits) == 13 and digits.startswith("091"):
        return f"+91{digits[3:]}"
    return f"+{digits}" if digits else None


def clean_bool(val) -> bool | None:
    """Normalise t/f strings and NaN to Python bool / None."""
    if isinstance(val, bool):
        return val
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, str):
        return val.strip().lower() in ("t", "true", "1", "yes")
    return bool(val)


def join_list(val, sep: str = "|") -> str | None:
    """Join list items with sep; pass-through strings unchanged."""
    if isinstance(val, list):
        joined = sep.join(str(v).strip() for v in val if v is not None and str(v).strip())
        return joined if joined else None
    return clean_str(val)


def flatten_location(loc: dict, prefix: str) -> dict:
    """Return prefixed flat columns from a location dict."""
    if not isinstance(loc, dict):
        return {}
    gps = loc.get("gps") or {}
    return {
        f"{prefix}_city":    clean_str(loc.get("city") or loc.get("tag")),
        f"{prefix}_state":   clean_str(loc.get("state")),
        f"{prefix}_address": clean_str(loc.get("address")),
        f"{prefix}_country": clean_str(loc.get("country")),
        f"{prefix}_lat":     gps.get("lat") if isinstance(gps, dict) else None,
        f"{prefix}_lng":     gps.get("lng") if isinstance(gps, dict) else None,
    }


# ── Seeker extractions ─────────────────────────────────────────────────────────

def extract_seeker_profile(df: pd.DataFrame) -> pd.DataFrame:
    # Pre-parse all metadata in one vectorised pass to avoid repeated json.loads
    parsed_meta = df["metadata"].apply(parse_json)
    new_cols = []
    for (_, row), meta in zip(df.iterrows(), parsed_meta):
        who  = ensure_dict(meta.get("whoIAm"))
        have = ensure_dict(meta.get("whatIHave"))
        want = ensure_dict(meta.get("whatIWant"))
        loc  = ensure_dict(who.get("locationData"))

        # Only columns that are NEW (not already present in the raw CSV)
        r = {
            # identity
            "name":         clean_str(meta.get("name") or who.get("name")),
            "role":         clean_str(meta.get("role")),
            "gender":       clean_str(meta.get("gender") or who.get("gender")),
            "age":          meta.get("age") or who.get("age") or have.get("age"),
            "industry":     clean_str(meta.get("industry")),
            # contact / location (whoIAm)
            "phone":                clean_phone(who.get("phone")),
            "date_of_birth":        clean_str(who.get("dateOfBirth")),
            "hometown":             clean_str(who.get("hometown")),
            "current_location":     clean_str(who.get("currentLocation") or who.get("location")),
            "desired_location":     clean_str(who.get("desiredLocation")),
            "father_name":          clean_str(who.get("fatherName")),
            "mother_name":          clean_str(who.get("motherName")),
            "aadhar_number":        clean_str(who.get("aadharNumber")),
            "location_city":        clean_str(loc.get("city")),
            "location_state":       clean_str(loc.get("state")),
            "location_address":     clean_str(loc.get("address")),
            "location_country":     clean_str(loc.get("country")),
            "is_age_verified":      clean_bool(who.get("isAgeVerified")),
            "is_name_verified":     clean_bool(who.get("isNameVerified")),
            "is_phone_verified":    clean_bool(who.get("isPhoneVerified")),
            "is_location_verified": clean_bool(who.get("isLocationVerified")),
            # whatIHave
            "roll_number":              clean_str(have.get("rollNumber")),
            "iti_institute":            clean_str(have.get("itiInstitute")),
            "iti_specialization":       join_list(have.get("itiSpecialization")),
            "training_duration_years":  have.get("trainingDuration"),
            "highest_qualification":    join_list(have.get("highestQualification")),
            "language_spoken":          join_list(have.get("languageSpoken")),
            "machines_operated":        join_list(have.get("machinesOperated")),
            "previous_company":         clean_str(have.get("previousCompany")),
            "previous_location":        clean_str(str(have.get("previousLocation")) if have.get("previousLocation") else None),
            "current_monthly_salary":   have.get("currentMonthlySalary"),
            "total_years_experience":   have.get("totalYearsOfExperience"),
            "fitter_assessment_score":  have.get("fitterAssessmentScore"),
            "intent_assessment_score":  have.get("intentAssessmentScore"),
            # whatIWant
            "preferred_mode_of_work":   join_list(want.get("preferredModeOfWork")),
            "monthly_in_hand_preferred": want.get("monthlyInHandPreferred"),
            "monthly_ot_expectation":   want.get("monthlyOTExpectation"),
            "monthly_pf_esic":          clean_str(str(want.get("monthlyPFESIC")) if want.get("monthlyPFESIC") is not None else None),
            "work_hours_per_day":       want.get("workHoursPerDay") or meta.get("workHoursPerDay"),
            # education — serialised list of dicts
            "education": join_list(
                [json.dumps(e, ensure_ascii=False) if isinstance(e, dict) else str(e)
                 for e in (meta.get("education") or [])]
            ),
            # top-level flags / metadata
            "status":               clean_str(meta.get("status")),
            "source":               clean_str(meta.get("source")),
            "agent_id":             clean_str(meta.get("agentId")),
            "is_aadhar_verified":   clean_bool(meta.get("isAadharVerified")),
            "is_gender_verified":   clean_bool(meta.get("isGenderVerified")),
            "is_hometown_verified": clean_bool(meta.get("isHometownVerified")),
            # whoIAm location coordinates
            "location_lat": loc.get("lat"),
            "location_lng": loc.get("lng"),
            # whatIHave — additional fields
            "work_experience": (
                json.dumps(have.get("workExperience"), ensure_ascii=False)
                if isinstance(have.get("workExperience"), (dict, list))
                else clean_str(have.get("workExperience"))
            ),
            "work_experience_years":              have.get("workExperienceYears"),
            "vehicle_ownership":                  clean_str(have.get("vehicleOwnership")),
            "communication_skills_score":         have.get("communicationSkillsScore"),
            "domain_knowledge":                   clean_str(have.get("domainKnowledge")),
            "software_skills":                    join_list(have.get("softwareSkills")),
            "languages_known":                    join_list(have.get("languagesKnown")),
            "highest_qualification_or_skill": (
                json.dumps(have.get("highestQualificationOrSkill"), ensure_ascii=False)
                if isinstance(have.get("highestQualificationOrSkill"), dict)
                else clean_str(have.get("highestQualificationOrSkill"))
            ),
            "highest_education":                          clean_str(have.get("highestEducation")),
            "highest_educational_institute_attended":     clean_str(have.get("highestEducationalInstituteAttended")),
            "name_of_last_role_held":                     clean_str(have.get("nameOfLastRoleHeld")),
            "juki_machine_experience":                    clean_str(have.get("jukiMachineExperience")),
            "juki_machine_straight_line": (
                json.dumps(have.get("jukiMachineStraightLine"), ensure_ascii=False)
                if isinstance(have.get("jukiMachineStraightLine"), dict)
                else clean_str(have.get("jukiMachineStraightLine"))
            ),
            "consent_physical_health":      clean_bool(have.get("consent for physical health")),
            "consent_safety_requirements":  clean_bool(have.get("consent for safety requirements")),
            "quality_proof_image":          clean_str(have.get("qualityProofImage")),
            "skill_proof_video":            clean_str(have.get("skillProofVideo")),
            "task_video":                   clean_str(have.get("taskVideo")),
            # whatIWant — additional fields
            "monthly_travel_allowance":      want.get("monthlyTravelAllowance"),
            "monthly_performance_variable":  want.get("monthlyPerformanceVariable"),
            "monthly_pf_health_insurance":   want.get("monthlyPFHealthInsurance"),
            "interested_job_roles":          join_list(want.get("nameOfJobRolesInterestedIn")),
            "nature_of_jobs_interested":     join_list(want.get("natureOfJobsInterestedIn")),
            "preferred_work_mode":           clean_str(want.get("preferredWorkMode")),
            "stay_preferences":              join_list(want.get("stayPreferences")),
            "max_distance_daily_travel":     want.get("maxDistanceofDailyTravel"),
            "max_cost_per_sharing_bed":      want.get("maxCostPerSharingBed"),
            "other_help_needed":             clean_str(want.get("otherHelpNeeded")),
            # whatIHave — missing fields
            "iti_institute_slug":                clean_str(have.get("itiInstituteSlug")),
            "assessment_scores": (
                json.dumps(have.get("assessmentScores"), ensure_ascii=False)
                if have.get("assessmentScores") is not None else None
            ),
            "document_verification_status": (
                json.dumps(have.get("documentVerificationStatus"), ensure_ascii=False)
                if have.get("documentVerificationStatus") is not None else None
            ),
            "college_name":                          clean_str(have.get("collegeName")),
            "highest_educational_qualification":     clean_str(have.get("highestEducationalQualification")),
            "highest_qualification_other":           clean_str(have.get("highestQualification_other")),
            # role-specific assessment scores (KA schema variants)
            "electrician_assessment_score":          have.get("electricianAssessmentScore"),
            "machine_operator_assessment_score":     have.get("machineOperatorAssessmentScore"),
            "mechanic_assessment_score":             have.get("mechanicAssessmentScore"),
            "hr_knowledge":                          have.get("hrKnowledge"),
            "hr_work_experience":                    have.get("hrWorkExperience"),
            "warehouse_experience":                  have.get("warehouseExperience"),
            "presentability_score":                  have.get("presentabilityScore"),
            # Gen 2 flat top-level fields
            "skills":               join_list(meta.get("skills")),
            "skill_certifications": join_list(meta.get("skillCertifications")),
            "certificates":         join_list(meta.get("certificates")),
            "gen2_experience":      clean_str(meta.get("experience")),
            "notes":                clean_str(meta.get("notes")),
        }
        new_cols.append(r)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(new_cols)], axis=1)


def extract_organisation(df: pd.DataFrame) -> pd.DataFrame:
    """Works for both seeker and provider organisation tables."""
    parsed_meta = df["metadata"].apply(parse_json)
    new_cols = []
    for (_, row), meta in zip(df.iterrows(), parsed_meta):
        # Only columns that are NEW (not already present in the raw CSV)
        r = {
            "address":        clean_str(meta.get("address")),
            "gst_number":     clean_str(meta.get("gstNumber")),
            "contact_person": clean_str(meta.get("contactPersonName")),
            "contact_email":  clean_str(meta.get("contactEmail")),
            "contact_phone":  clean_phone(str(meta.get("contactPhone")) if meta.get("contactPhone") else None),
            "website":        clean_str(meta.get("website")),
            "description":    clean_str(meta.get("description")),
        }
        new_cols.append(r)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(new_cols)], axis=1)


def clean_user(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    str_cols  = ["name", "email", "role", "ban_reason", "image"]
    bool_cols = ["email_verified", "phone_number_verified", "banned",
                 "terms_accepted", "privacy_accepted"]
    for col in str_cols:
        if col in out.columns:
            out[col] = out[col].apply(clean_str)
    for col in bool_cols:
        if col in out.columns:
            out[col] = out[col].apply(clean_bool)
    if "phone_number" in out.columns:
        out["phone_number"] = out["phone_number"].apply(
            lambda v: clean_phone(str(v)) if pd.notna(v) else None
        )
    return out


def clean_member(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["role"] = out["role"].apply(clean_str)
    return out


# ── Provider extractions ───────────────────────────────────────────────────────

def extract_job_posting(df: pd.DataFrame) -> pd.DataFrame:
    parsed_meta = df["metadata"].apply(parse_json)
    new_cols = []
    for (_, row), meta in zip(df.iterrows(), parsed_meta):
        needs        = ensure_dict(meta.get("jobNeeds"))
        basic        = ensure_dict(meta.get("basicInfo"))
        details      = ensure_dict(meta.get("jobDetails"))
        edu          = ensure_dict(needs.get("educationSubsection"))
        lang         = ensure_dict(needs.get("languagesSubsection"))
        ploc         = ensure_dict(basic.get("jobProviderLocation"))
        gps          = ensure_dict(ploc.get("gps"))
        loc_pref     = ensure_dict(needs.get("locationPreference"))
        loc_pref_gps = ensure_dict(loc_pref.get("gps"))

        # Compute nature_of_job and unified salary before building the row dict
        nature_of_job_val = clean_str(details.get("natureOfJob"))
        _noj = (nature_of_job_val or "").strip().lower()
        if any(t in _noj for t in ("internship", "apprenticeship", "trainee", "stipend")):
            unified_salary_min = details.get("stipendMin")
            unified_salary_max = details.get("stipendMax")
        elif any(t in _noj for t in ("task", "piece", "gig", "freelance", "daily wage", "daily-wage")):
            unified_salary_min = details.get("taskRateMin")
            unified_salary_max = details.get("taskRateMax")
        else:
            unified_salary_min = details.get("minMonthlyInHand") or details.get("minSalary") or details.get("salaryMin") or details.get("minCTC")
            unified_salary_max = details.get("maxMonthlyInHand") or details.get("maxSalary") or details.get("salaryMax") or details.get("maxCTC")

        # Only columns that are NEW (not already present in the raw CSV)
        r = {
            # role / industry
            "role":     clean_str(meta.get("role") or meta.get("jobRole") or meta.get("job_role") or details.get("title") or details.get("jobTitle") or row.get("title")),
            "industry": clean_str(meta.get("industry") or meta.get("sector") or meta.get("jobSector") or meta.get("industrySector")),
            # provider location (extracted from metadata.basicInfo)
            "provider_location_city":    clean_str(ploc.get("city") or ploc.get("tag")),
            "provider_location_state":   clean_str(ploc.get("state")),
            "provider_location_address": clean_str(ploc.get("address")),
            "provider_location_country": clean_str(ploc.get("country")),
            "provider_location_lat":     gps.get("lat") if isinstance(gps, dict) else None,
            "provider_location_lng":     gps.get("lng") if isinstance(gps, dict) else None,
            # job details (extracted from metadata.jobDetails)
            "job_type":                  join_list(details.get("jobType")),
            "positions":                 details.get("positions") or details.get("openings") or details.get("numberOfOpenings") or details.get("noOfOpenings") or details.get("vacancies"),
            "mode_of_work":              join_list(details.get("modeOfWork")),
            "food_provided":             clean_str(details.get("foodProvided")),
            "travel_provided":           clean_str(details.get("travelProvided")),
            "stay_provided":             clean_str(str(details.get("stayProvided")) if details.get("stayProvided") is not None else None),
            "job_description":           clean_str(details.get("jobDescription") or details.get("description") or details.get("jobDesc") or row.get("description")),
            "min_monthly_in_hand":       details.get("minMonthlyInHand") or details.get("minSalary") or details.get("salaryMin") or details.get("minCTC"),
            "max_monthly_in_hand":       details.get("maxMonthlyInHand") or details.get("maxSalary") or details.get("salaryMax") or details.get("maxCTC"),
            "working_hours_per_day":     details.get("workingHoursPerDay"),
            "hiring_urgency":            clean_str(details.get("hiringUrgency")),
            "hiring_manager_name":       clean_str(details.get("hiringManagerName")),
            "hiring_manager_designation": clean_str(details.get("hiringManagerDesignation")),
            # job needs
            "gender_preference":        clean_str(needs.get("genderPreference")),
            "age_lower_limit":          needs.get("ageAllowedLowerLimit"),
            "age_upper_limit":          needs.get("ageAllowedUpperLimit"),
            "min_education_level":      join_list(edu.get("minEducationLevel")),
            "iti_specialty_preference": join_list(edu.get("itiSpecialtyPreference")),
            "languages_required":       join_list(lang.get("languagesKnown")),
            # top-level  (status is already a native column on job_posting, skip to avoid status.1 duplicate)
            "job_title":    clean_str(details.get("title") or row.get("title")),
            "org_name":     clean_str(meta.get("org_name")),
            "org_slug":     clean_str(meta.get("org_slug")),
            "is_duplicate": clean_bool(meta.get("isDuplicate")),
            "agent_id":     clean_str(meta.get("agentId")),
            "agent_title":  clean_str(meta.get("agentTitle")),
            # basicInfo — provider details
            "job_provider_name":    clean_str(basic.get("jobProviderName")),
            "job_provider_type":    clean_str(basic.get("jobProviderType")),
            "hiring_manager_email": clean_str(basic.get("hiringManagerEmail")),
            "hiring_manager_phone": clean_phone(str(basic.get("hiringManagerPhoneNumber")) if basic.get("hiringManagerPhoneNumber") else None),
            # jobDetails — compensation & logistics
            "nature_of_job":                     nature_of_job_val,
            "monthly_attendance_bonus":          details.get("monthlyAttendanceBonus"),
            "monthly_attendance_bonus_criteria": clean_str(details.get("monthlyAttendanceBonusCriteria")),
            "monthly_travelling_allowance":      details.get("monthlyTravellingAllowance"),
            "monthly_average_ot":                details.get("monthlyAverageOT") or details.get("monthlyAverageOt"),
            "monthly_incentive_possible":        details.get("monthlyIncentivePossible"),
            "monthly_max_performance_variable":  details.get("monthlyMaxPerformanceBasedVariable"),
            "monthly_pf_esic_benefits":          details.get("monthlyPfEsicBenefits"),
            "ot_per_hour_rate":                  details.get("otPerHourRate"),
            "daily_travel_required":             clean_str(str(details.get("dailyTravelRequired")) if details.get("dailyTravelRequired") is not None else None),
            "valid_till":                        clean_str(details.get("validTill")),
            "stipend_min":                       details.get("stipendMin"),
            "stipend_max":                       details.get("stipendMax"),
            "salary_min":                        details.get("salaryMin"),
            "salary_max":                        details.get("salaryMax"),
            "task_rate_min":                     details.get("taskRateMin"),
            "task_rate_max":                     details.get("taskRateMax"),
            "callback_time":                     clean_str(details.get("callBackTime")),
            # jobNeeds — candidate requirements
            "work_experience_years_required": needs.get("workExperienceYears"),
            "candidate_experience_type":      clean_str(needs.get("candidateExperienceType")),
            "last_role_held":                 clean_str(needs.get("lastRoleHeld")),
            "vehicle_ownership_required":     clean_str(
                needs.get("vehicleOwnershipRequirement") or
                ensure_dict(needs.get("customerHandlingSubsection")).get("vehicleOwnershipRequirement")
            ),
            "highest_qualification_accepted": join_list(
                needs.get("highestQualificationAccepted") or
                ensure_dict(needs.get("highestQualificationSubsection")).get("highestQualification")
            ),
            # jobNeeds — location preference
            "location_preference_city":    clean_str(loc_pref.get("city") or loc_pref.get("tag")),
            "location_preference_state":   clean_str(loc_pref.get("state")),
            "location_preference_address": clean_str(loc_pref.get("address")),
            "location_preference_country": clean_str(loc_pref.get("country")),
            "location_preference_lat":     loc_pref_gps.get("lat"),
            "location_preference_lng":     loc_pref_gps.get("lng"),
            # basicInfo — additional fields
            "job_provider_logo":         clean_str(basic.get("jobProviderLogo")),
            "job_provider_registration": clean_str(basic.get("jobProviderRegistration")),
            # jobDetails — additional fields
            "designation":      clean_str(details.get("designation")),
            "devices_provided": join_list(details.get("devicesProvided")),
            "start_time":       clean_str(details.get("startTime")),
            "end_time":         clean_str(details.get("endTime")),
            # meta — original job tracking
            "original_job_id":  clean_str(meta.get("originalJobId")),
            # jobNeeds — subsections (serialised; KA-only unless noted)
            "communication_subsection": (
                json.dumps(needs.get("communicationSubsection"), ensure_ascii=False)
                if needs.get("communicationSubsection") is not None else None
            ),
            "domain_knowledge_subsection": (
                json.dumps(needs.get("domainKnowledgeSubsection"), ensure_ascii=False)
                if needs.get("domainKnowledgeSubsection") is not None else None
            ),
            "hr_knowledge_subsection": (
                json.dumps(needs.get("hrKnowledgeSubsection"), ensure_ascii=False)
                if needs.get("hrKnowledgeSubsection") is not None else None
            ),
            "hr_work_experience_subsection": (
                json.dumps(needs.get("hrWorkExperienceSubsection"), ensure_ascii=False)
                if needs.get("hrWorkExperienceSubsection") is not None else None
            ),
            "juki_machine_straight_line_subsection": (
                json.dumps(needs.get("jukiMachineStraightLineSubsection"), ensure_ascii=False)
                if needs.get("jukiMachineStraightLineSubsection") is not None else None
            ),
            "physical_fitness_subsection": (
                json.dumps(needs.get("physicalFitnessSubsection"), ensure_ascii=False)
                if needs.get("physicalFitnessSubsection") is not None else None
            ),
            "physical_health_requirements": (
                json.dumps(needs.get("physicalHealthRequirements"), ensure_ascii=False)
                if needs.get("physicalHealthRequirements") is not None else None
            ),
            "ppe_ready_safety_compliance": (
                json.dumps(needs.get("ppeReadySafetyCompliance"), ensure_ascii=False)
                if needs.get("ppeReadySafetyCompliance") is not None else None
            ),
            "educational_qualification_subsection": (
                json.dumps(needs.get("educationalQualificationSubsection"), ensure_ascii=False)
                if needs.get("educationalQualificationSubsection") is not None else None
            ),
            # jobNeeds.educationSubsection — additional field
            "skills_required": join_list(edu.get("skillsRequired")),
            # jobNeeds — UP-only fields
            "software_knowledge_preferred":       clean_str(
                ensure_dict(needs.get("highestQualificationSubsection")).get("softwareKnowledgePreferred")
            ),
            "communication_skill_score_required": needs.get("communicationSkillScore"),
            # unified salary — single canonical min/max based on nature_of_job
            "unified_salary_min": unified_salary_min,
            "unified_salary_max": unified_salary_max,
        }
        new_cols.append(r)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(new_cols)], axis=1)


def extract_job_application(df: pd.DataFrame) -> pd.DataFrame:
    parsed_meta = df["metadata"].apply(parse_json)
    new_cols = []
    for (_, row), meta in zip(df.iterrows(), parsed_meta):
        seeker_meta = ensure_dict(meta.get("metadata"))
        who         = ensure_dict(seeker_meta.get("whoIAm"))
        have        = ensure_dict(seeker_meta.get("whatIHave"))
        want        = ensure_dict(seeker_meta.get("whatIWant"))
        # job_snap lives at the top level of metadata, not inside metadata.metadata
        job_snap    = ensure_dict(meta.get("jobDetails"))
        inner_job   = ensure_dict(job_snap.get("jobDetails"))
        contact     = parse_json(row.get("contact"))
        app_loc     = parse_json(row.get("location"))
        app_gps     = ensure_dict(app_loc.get("gps"))

        # app_loc city/state can be dicts like {"code": "std:080", "name": "Bangalore"}
        city_raw  = app_loc.get("city",  "")
        state_raw = app_loc.get("state", "")

        # Only columns that are NEW (not already present in the raw CSV)
        r = {
            # seeker identity (extracted / cleaned from contact + metadata)
            "seeker_name":  clean_str(row.get("user_name") or meta.get("name")),
            "seeker_phone": clean_phone(contact.get("phone") or who.get("phone")),
            "seeker_email": clean_str(contact.get("email")),
            "seeker_age":   meta.get("age") or seeker_meta.get("age"),
            "seeker_gender": clean_str(meta.get("gender") or who.get("gender")),
            # application location (parsed from location JSON)
            "app_location_city":    clean_str(city_raw.get("name")  if isinstance(city_raw,  dict) else city_raw),
            "app_location_state":   clean_str(state_raw.get("name") if isinstance(state_raw, dict) else state_raw),
            "app_location_address": clean_str(app_loc.get("address")),
            "app_location_country": clean_str(app_loc.get("country")),
            # seeker profile snapshot (from metadata)
            "seeker_current_location":      clean_str(who.get("location") or who.get("currentLocation")),
            "seeker_iti_institute":         clean_str(have.get("itiInstitute")),
            "seeker_iti_specialization":    join_list(have.get("itiSpecialization")),
            "seeker_highest_qualification": join_list(have.get("highestQualification")),
            "seeker_language_spoken":       join_list(have.get("languageSpoken")),
            "seeker_roll_number":           clean_str(have.get("rollNumber")),
            "seeker_total_experience":      have.get("totalYearsOfExperience"),
            "seeker_expected_salary":       want.get("monthlyInHandPreferred"),
            "seeker_preferred_mode":        join_list(want.get("preferredModeOfWork")),
            # job snapshot (from metadata.jobDetails)
            "job_title":      clean_str(job_snap.get("title") or job_snap.get("jobTitle") or inner_job.get("title") or seeker_meta.get("interestedRole")),
            "job_role":       clean_str(job_snap.get("role")  or seeker_meta.get("interestedRole")),
            "job_industry":   clean_str(job_snap.get("industry") or seeker_meta.get("interestedIndustry")),
            "job_company":    clean_str(job_snap.get("company") or job_snap.get("jobProviderName")),
            "job_location":   clean_str(str(job_snap.get("location")) if job_snap.get("location") else None),
            "job_openings":   job_snap.get("openings") or job_snap.get("positions") or inner_job.get("positions"),
            "job_min_salary": job_snap.get("monthlyInHand") or inner_job.get("minMonthlyInHand"),
            "job_max_salary": inner_job.get("maxMonthlyInHand"),
            "match_score":    job_snap.get("matchScore"),
            # job snapshot — additional fields
            "job_provider_name":    clean_str(job_snap.get("jobProviderName") or ensure_dict(job_snap.get("basicInfo")).get("jobProviderName")),
            "job_hiring_manager":   clean_str(inner_job.get("hiringManagerName")),
            "job_nature":           clean_str(inner_job.get("natureOfJob")),
            "job_min_monthly_ot":   inner_job.get("monthlyAverageOT"),
            "job_valid_till":       clean_str(inner_job.get("validTill")),
            # application location — GPS
            "app_location_lat":  app_gps.get("lat"),
            "app_location_lng":  app_gps.get("lng"),
            # seeker snapshot — additional fields
            "seeker_interested_role":             clean_str(meta.get("interestedRole") or seeker_meta.get("interestedRole")),
            "seeker_interested_industry":         clean_str(meta.get("interestedIndustry") or seeker_meta.get("interestedIndustry")),
            "seeker_is_age_verified":             clean_bool(meta.get("isAgeVerified")),
            "seeker_is_aadhar_verified":          clean_bool(meta.get("isAadharVerified")),
            "seeker_is_name_verified":            clean_bool(meta.get("isNameVerified")),
            "seeker_is_hometown_verified":        clean_bool(meta.get("isHometownVerified")),
            "seeker_is_gender_verified":          clean_bool(meta.get("isGenderVerified")),
            "seeker_work_experience": (
                json.dumps(have.get("workExperience"), ensure_ascii=False)
                if isinstance(have.get("workExperience"), (dict, list))
                else clean_str(have.get("workExperience"))
            ),
            "seeker_work_experience_years":       have.get("workExperienceYears"),
            "seeker_vehicle_ownership":           clean_str(have.get("vehicleOwnership")),
            "seeker_communication_skills_score":  have.get("communicationSkillsScore"),
            "seeker_domain_knowledge":            clean_str(have.get("domainKnowledge")),
            "seeker_software_skills":             join_list(have.get("softwareSkills")),
            "seeker_monthly_travel_allowance":    want.get("monthlyTravelAllowance"),
            "seeker_monthly_performance_variable": want.get("monthlyPerformanceVariable"),
            "seeker_preferred_work_mode":         clean_str(want.get("preferredWorkMode")),
            "seeker_stay_preferences":            join_list(want.get("stayPreferences")),
            "seeker_max_cost_per_sharing_bed":    want.get("maxCostPerSharingBed"),
            # seeker snapshot — profile/education data from meta top-level
            "seeker_profile_id":   clean_str(meta.get("profileId")),
            "seeker_skills":       join_list(meta.get("skills")),
            "seeker_skill_certifications": join_list(meta.get("skillCertifications")),
            "seeker_certificates": join_list(meta.get("certificates")),
            "seeker_assessment_scores": (
                json.dumps(meta.get("assessmentScores"), ensure_ascii=False)
                if meta.get("assessmentScores") is not None else None
            ),
            "seeker_education": join_list(
                [json.dumps(e, ensure_ascii=False) if isinstance(e, dict) else str(e)
                 for e in (meta.get("education") or [])]
            ),
        }
        new_cols.append(r)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(new_cols)], axis=1)


# ── Pipeline runners ───────────────────────────────────────────────────────────

def find_data_dir(folder_name: str) -> Path:
    """Handle the nested folder pattern raw_data/X/X/."""
    outer = RAW_DIR / folder_name
    inner = outer / folder_name
    return inner if inner.exists() else outer


def process_seeker(data_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    profile = pd.read_csv(data_dir / "profile.csv",      low_memory=False)
    user    = pd.read_csv(data_dir / "user.csv",          low_memory=False)
    member  = pd.read_csv(data_dir / "member.csv",        low_memory=False)
    org     = pd.read_csv(data_dir / "organization.csv",  low_memory=False)

    extract_seeker_profile(profile).to_csv(out_dir / "profile_clean.csv",      index=False)
    clean_user(user)               .to_csv(out_dir / "user_clean.csv",          index=False)
    clean_member(member)           .to_csv(out_dir / "member_clean.csv",        index=False)
    extract_organisation(org)      .to_csv(out_dir / "organization_clean.csv",  index=False)

    print(f"  [{data_dir.name}] seeker done -> {out_dir}")


def process_provider(data_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    job_posting     = pd.read_csv(data_dir / "job_posting.csv",     low_memory=False)
    job_application = pd.read_csv(data_dir / "job_application.csv", low_memory=False)
    member  = pd.read_csv(data_dir / "member.csv",       low_memory=False)
    org     = pd.read_csv(data_dir / "organization.csv", low_memory=False)
    user    = pd.read_csv(data_dir / "user.csv",         low_memory=False)

    extract_job_posting(job_posting)         .to_csv(out_dir / "job_posting_clean.csv",     index=False)
    extract_job_application(job_application) .to_csv(out_dir / "job_application_clean.csv", index=False)
    clean_member(member)                     .to_csv(out_dir / "member_clean.csv",           index=False)
    extract_organisation(org)                .to_csv(out_dir / "organization_clean.csv",     index=False)
    clean_user(user)                         .to_csv(out_dir / "user_clean.csv",             index=False)

    print(f"  [{data_dir.name}] provider done -> {out_dir}")


# ── Entry point ────────────────────────────────────────────────────────────────

DATASETS = {
    "ka_seeker":   "seeker",
    "up_seeker":   "seeker",
    "ka_provider": "provider",
    "up_provider": "provider",
}


def main():
    for key, kind in DATASETS.items():
        print(f"\nProcessing {key} ...")
        data_dir = RAW_DIR / key
        out_dir  = OUT_DIR / key

        if kind == "seeker":
            process_seeker(data_dir, out_dir)
        else:
            process_provider(data_dir, out_dir)

    print("\nAll done.")


if __name__ == "__main__":
    main()
