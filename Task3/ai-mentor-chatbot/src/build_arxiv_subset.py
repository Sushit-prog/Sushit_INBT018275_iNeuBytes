"""
Build a subset of ML-ArXiv papers tailored for the mentor chatbot's knowledge
base.

Filters the full dataset for papers with titles or abstracts that suggest
foundational or accessible content (surveys, tutorials, overviews, etc.).
Writes the selected papers as a markdown file into ``data/knowledge_base/``.
"""

import os
import math
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "raw_dataset" / "ml_arxiv_papers.csv"
OUTPUT_PATH = BASE_DIR / "data" / "knowledge_base" / "09_arxiv_ml_papers_subset.md"
OUTPUT_FILENAME = "09_arxiv_ml_papers_subset.md"

# Case-insensitive keywords targeting accessible / foundational papers
KEYWORDS = [
    "beginner",
    "survey",
    "review",
    "tutorial",
    "introduction to",
    "overview",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _contains_keyword(text: str) -> bool:
    """Return ``True`` if *text* contains any of the configured keywords."""
    lower = text.lower()
    return any(kw in lower for kw in KEYWORDS)


def format_paper(title: str, abstract: str) -> str:
    """Format a single paper as a markdown block."""
    return (
        f"## {title}\n"
        f"{abstract}\n"
        f"(Source: ML-ArXiv-Papers dataset)"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Load the ML-ArXiv dataset and produce a curated markdown subset."""
    # ---- Load ----------------------------------------------------------------
    print("=" * 60)
    print("Building arXiv ML papers subset for knowledge base")
    print("=" * 60)

    if not CSV_PATH.is_file():
        print(f"\n[WARNING] CSV not found at: {CSV_PATH}")
        print("Skipping arXiv papers subset generation.")
        return

    df: pd.DataFrame = pd.read_csv(CSV_PATH)
    total = len(df)
    print(f"\nLoaded {total:,} rows from {CSV_PATH.name}")

    # Drop rows where both title and abstract are missing
    df = df.dropna(subset=["title", "abstract"], how="all").reset_index(drop=True)
    print(f"After dropping rows with neither title nor abstract: {len(df):,}")

    # ---- Keyword filter ------------------------------------------------------
    mask = df["title"].fillna("").apply(_contains_keyword) | \
           df["abstract"].fillna("").apply(_contains_keyword)

    filtered = df[mask].copy().reset_index(drop=True)
    n_filtered = len(filtered)
    print(f"Keyword-filtered count: {n_filtered}")

    # ---- Selection logic -----------------------------------------------------
    if n_filtered < 150:
        print(
            f"  Fewer than 150 matches ({n_filtered}) — "
            f"falling back to random sample of 300 from full dataset."
        )
        selected: pd.DataFrame = df.sample(n=300, random_state=42).reset_index(drop=True)
    elif n_filtered > 300:
        print(f"  More than 300 matches ({n_filtered}) — taking first 300.")
        selected = filtered.head(300).reset_index(drop=True)
    else:
        print(f"  Using all {n_filtered} keyword-filtered papers.")
        selected = filtered

    n_selected = len(selected)
    print(f"\nSelected {n_selected} papers.")

    # ---- Build markdown content ---------------------------------------------
    blocks: list[str] = []
    for _, row in selected.iterrows():
        blocks.append(format_paper(row["title"], row["abstract"]))

    markdown_content = "\n\n".join(blocks)

    # ---- Write output --------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        fh.write(markdown_content)
        fh.write("\n")  # trailing newline

    file_size_kb = math.ceil(OUTPUT_PATH.stat().st_size / 1024)

    print()
    print("=" * 60)
    print("Done!")
    print(f"  Papers selected:   {n_selected}")
    print(f"  Output file:       {OUTPUT_FILENAME}")
    print(f"  File size:         ~{file_size_kb} KB")
    print("=" * 60)


if __name__ == "__main__":
    main()
