FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    anthropic pandas gspread google-auth \
    google-cloud-storage google-cloud-secret-manager rapidfuzz psycopg2-binary

WORKDIR /app

# Copy pipeline code (not data — data lives in GCS)
COPY preprocessing/ ./preprocessing/
COPY populate_dashboard.py google_sheets_connector.py run_pipeline.py test.txt db_extract.py ./

# Copy the entrypoint
COPY entrypoint.py /entrypoint.py

ENTRYPOINT ["python", "/entrypoint.py"]
