# Blue Dots Signals Data Processing Pipeline

An automated data pipeline that extracts, cleans, enriches, and matches job seekers to job postings for the Blue Dots platform (UP and KA states), then publishes results to a Google Sheets dashboard.

## Overview

The pipeline runs as a **Google Cloud Run Job** and executes the following steps in order:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `normalisation.py` | Clean and flatten raw CSVs from PostgreSQL |
| 2 | `llm_location_enrichment.py` | Predict state/district from free-text location |
| 3 | `role_normalisation.py` | Map free-text job titles to canonical roles using Claude API |
| 4 | `flag_test.py` | Flag test/spam records |
| 5 | `matchmaking.py` | Match seekers to open job postings |
| 6 | `populate_dashboard.py` | Upload processed data to Google Sheets dashboard |

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
