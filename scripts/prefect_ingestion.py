from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from prefect import flow, task

ROOT = Path(__file__).resolve().parents[1]


@task(retries=2, retry_delay_seconds=5)
def run_step(script_name: str):
    script = ROOT / "scripts" / script_name
    result = subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True, text=True)
    return result.returncode


@flow(name="fininsight-automated-ingestion", log_prints=True)
def fininsight_ingestion():
    """Automated ingestion/indexing flow for the financial knowledge base."""
    run_step("phase2_data_foundation.py")
    run_step("phase3_ingestion.py")
    run_step("phase4_chunking.py")
    run_step("phase5_embeddings.py")
    run_step("phase6_bm25.py")
    run_step("phase7_hybrid_retrieval.py")



if __name__ == "__main__":
    fininsight_ingestion()
