"""
entrypoint.py

Cloud Run Job entrypoint:
  1. Extract fresh data from PostgreSQL databases
  2. Run the full pipeline
  3. Upload processed_data/ back to GCS
"""

import os
import sys
from pathlib import Path
from google.cloud import storage

BUCKET_NAME = "blue-dots-pipeline"
APP_DIR     = Path("/app")

DOWNLOAD = []
UPLOAD = [
    ("processed_data/", APP_DIR / "processed_data"),
]


def sync_down(client: storage.Client) -> None:
    bucket = client.bucket(BUCKET_NAME)
    for prefix, local_dir in DOWNLOAD:
        print(f"  Downloading gs://{BUCKET_NAME}/{prefix} -> {local_dir}")
        blobs = list(client.list_blobs(BUCKET_NAME, prefix=prefix))
        for blob in blobs:
            rel = blob.name[len(prefix):]
            if not rel:
                continue
            dest = local_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(dest))
        print(f"  Downloaded {len(blobs)} files")


def sync_up(client: storage.Client) -> None:
    bucket = client.bucket(BUCKET_NAME)
    for prefix, local_dir in UPLOAD:
        if not local_dir.exists():
            print(f"  Skipping {local_dir} (does not exist)")
            continue
        files = [f for f in local_dir.rglob("*") if f.is_file()]
        print(f"  Uploading {len(files)} files -> gs://{BUCKET_NAME}/{prefix}")
        for f in files:
            blob_name = prefix + str(f.relative_to(local_dir)).replace("\\", "/")
            bucket.blob(blob_name).upload_from_filename(str(f))
    print("  Upload complete")


def main() -> None:
    client = storage.Client()

    sys.path.insert(0, str(APP_DIR))

    print()
    print("=" * 50)
    print("Connectivity test...")
    print("=" * 50)
    try:
        import httpx
        r = httpx.get("https://api.anthropic.com", timeout=10)
        print(f"  GET api.anthropic.com: HTTP {r.status_code}")
    except Exception as e:
        print(f"  GET api.anthropic.com FAILED: {type(e).__name__}: {e}")
    try:
        import httpx
        r = httpx.post("https://api.anthropic.com/v1/messages", timeout=10,
                       headers={"x-api-key": "test", "anthropic-version": "2023-06-01",
                                "content-type": "application/json"},
                       content=b'{"model":"claude-haiku-4-5-20251001","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}')
        print(f"  POST api.anthropic.com/v1/messages: HTTP {r.status_code}")
    except Exception as e:
        print(f"  POST api.anthropic.com/v1/messages FAILED: {type(e).__name__}: {e}")
    try:
        import httpx as _httpx
        _resp = _httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": __import__("re").search(r"sk-ant-[A-Za-z0-9\-_]+", os.environ.get("CLAUDE_API_KEY", "")).group(0),
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
            },
            timeout=30.0,
        )
        _resp.raise_for_status()
        print(f"  Anthropic API call: OK — {_resp.json()['content'][0]['text']!r}")
    except Exception as e:
        print(f"  Anthropic API call FAILED: {type(e).__name__}: {e}")
        print("  Aborting pipeline — fix Anthropic API connectivity before retrying.")
        sys.stdout.flush()
        sys.exit(1)
    sys.stdout.flush()

    print()
    print("=" * 50)
    print("Extracting data from databases...")
    print("=" * 50)
    sys.stdout.flush()
    import db_extract
    db_extract.main()
    sys.stdout.flush()

    print()
    print("=" * 50)
    print("Running pipeline...")
    print("=" * 50)
    sys.stdout.flush()
    import run_pipeline
    test_mode = os.environ.get("TEST_MODE", "").lower() in ("1", "true", "yes")
    if test_mode:
        print("  *** TEST MODE: results will be written to test sheets ***")
    run_pipeline.main(test=test_mode)
    sys.stdout.flush()

    print()
    print("=" * 50)
    print("Syncing results back to GCS...")
    print("=" * 50)
    sync_up(client)


if __name__ == "__main__":
    main()
