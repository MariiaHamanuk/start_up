"""
Startup post-mortem dataset analysis.
Pulls all ingested document metadata from Qdrant and computes statistics with Pandas + NumPy.

Usage:
    python scripts/analyze.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient

from app.config import settings


def fetch_metadata() -> pd.DataFrame:
    client = QdrantClient(url=settings.qdrant_url)
    records, _ = client.scroll(
        collection_name=settings.collection_name,
        with_payload=True,
        with_vectors=False,
        limit=10_000,
    )
    rows = []
    for record in records:
        payload = record.payload or {}
        meta = payload.get("metadata", {})
        rows.append({
            "startup_name": meta.get("startup_name", "unknown"),
            "industry":     meta.get("industry", "unknown"),
            "year":         meta.get("year"),
            "stage":        meta.get("stage", "unknown"),
            "source":       meta.get("source", "unknown"),
            "chunk_length": len(payload.get("page_content", "")),
        })
    return pd.DataFrame(rows)


def main():
    print("Fetching metadata from Qdrant...")
    df = fetch_metadata()

    if df.empty:
        print("No documents ingested yet. Add some post-mortems first.")
        return

    print(f"\n{'='*45}")
    print(f"  Total chunks indexed : {len(df)}")
    print(f"  Unique startups      : {df['startup_name'].nunique()}")
    print(f"  Unique sources       : {df['source'].nunique()}")
    print(f"{'='*45}")

    print("\n--- Industry distribution ---")
    print(df["industry"].value_counts().to_string())

    print("\n--- Stage distribution ---")
    print(df["stage"].value_counts().to_string())

    print("\n--- Chunk length statistics ---")
    cl = df["chunk_length"]
    print(f"  Mean   : {cl.mean():.0f} chars")
    print(f"  Median : {np.median(cl):.0f} chars")
    print(f"  Std    : {cl.std():.0f} chars")
    print(f"  Min    : {cl.min()} chars")
    print(f"  Max    : {cl.max()} chars")

    year_df = df[df["year"].notna()]
    if not year_df.empty:
        print("\n--- Failure count by year ---")
        year_counts = year_df["year"].astype(int).value_counts().sort_index()
        print(year_counts.to_string())

        # NumPy: year trend
        years = year_df["year"].astype(int).values
        print(f"\n  Earliest recorded failure : {np.min(years)}")
        print(f"  Latest recorded failure   : {np.max(years)}")
        print(f"  Mean failure year         : {np.mean(years):.0f}")

    print("\n--- Top 10 startups by chunk count ---")
    top = df.groupby("startup_name").size().sort_values(ascending=False).head(10)
    print(top.to_string())


if __name__ == "__main__":
    main()
