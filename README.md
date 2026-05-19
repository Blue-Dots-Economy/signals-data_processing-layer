# Blue Dots Signals Data Processing Pipeline

An automated data pipeline that extracts, cleans, enriches, and matches job seekers to job postings for the Blue Dots platform (UP and KA states), then publishes results to a Google Sheets dashboard.

## Overview

The pipeline runs as a **Google Cloud Run Job** and executes the following steps in order:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `normalisation.py` | Flattens nested JSON metadata (`whoIAm / whatIHave / whatIWant`) into flat columns; standardises phone numbers to `+91XXXXXXXXXX`; classifies `nature_of_job` (internship / gig / regular) to pick the correct salary fields (`stipendMin/Max`, `taskRateMin/Max`, or `minMonthlyInHand/maxMonthlyInHand`); extracts job role from multiple schema variants (`role`, `jobRole`, `job_role`, `title`, `jobTitle`) |
| 2 | `llm_location_enrichment.py` | Predicts `location_state` and `location_district` from free-text location using Claude API |
| 3 | `role_normalisation.py` | Maps free-text job titles to canonical roles and sectors using Claude API |
| 4 | `flag_test.py` | Rule-based spam/test detection: flags by invalid phone patterns, known test phone numbers, internal email domains (`dhiway.com`, `ekstepplus.org`), plus-addressing (`+test`, `+demo`), org name keywords (`test`, `demo`, `xyz`, `ekstep`, `dhiway`); cascades org flags to linked members and job postings |
| 5 | `matchmaking.py` | Weighted rule-based seeker-to-job matching: role fit via fuzzy token scoring (40%), location match (30%), salary compatibility (20%), education level hierarchy (10%); outputs `High / Medium / Low` confidence labels with per-dimension scores |
| 6 | `populate_dashboard.py` | Computes **provider profile completion** (8 fields) and **seeker profile completion** (13 fields); derives **job status** (`New / Active / At Risk / Satisfied / Inactive`) from job age, application counts, and resolution rates; detects **duplicate postings** by matching 7 key fields; calculates **application age metrics** (job age, last application date, pending days, shortlisted/rejected age); generates **recommended actions** and **follow-up flags** for missing profile fields; uploads all data to Google Sheets |

## Tech Stack

- **Runtime**: Python 3.12, Docker, Google Cloud Run Jobs
- **Storage**: PostgreSQL (source), Google Cloud Storage (intermediate), Google Sheets (output)
- **AI**: Anthropic Claude API (role normalisation)
- **Key libraries**: `pandas`, `gspread`, `rapidfuzz`, `psycopg2`, `google-cloud-*`

## Project Structure

```
├── run_pipeline.py              # Main entry point
├── entrypoint.py                # Cloud Run Job entrypoint (handles GCS sync)
├── db_extract.py                # Extracts raw data from PostgreSQL
├── populate_dashboard.py        # Uploads results to Google Sheets
├── google_sheets_connector.py   # Google Sheets utilities
└── preprocessing/
    ├── normalisation.py
    ├── llm_location_enrichment.py
    ├── role_normalisation.py
    ├── flag_test.py
    └── matchmaking.py
```

## Running Locally

```bash
# Full pipeline
python run_pipeline.py

# Resume dashboard upload from last checkpoint
python run_pipeline.py --resume

# Upload only a specific state
python run_pipeline.py --state KA
python run_pipeline.py --state UP
```

## Environment

Credentials are loaded from **Google Secret Manager** in Cloud Run, or from a local `.env` / `db_credentials.json` for development. Never commit these files.

## Deploying

```bash
# Build and push image
docker build -t asia-south1-docker.pkg.dev/blue-dots-project/blue-dots-repo/blue-dots-pipeline:latest .
docker push asia-south1-docker.pkg.dev/blue-dots-project/blue-dots-repo/blue-dots-pipeline:latest

# Update and run the Cloud Run Job
gcloud run jobs update blue-dots-pipeline \
  --image asia-south1-docker.pkg.dev/blue-dots-project/blue-dots-repo/blue-dots-pipeline:latest \
  --region asia-south1 --project blue-dots-project
gcloud run jobs execute blue-dots-pipeline --region asia-south1 --project blue-dots-project --wait
```
