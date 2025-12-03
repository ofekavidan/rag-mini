# prepare_dataset.py
# Creates a small, clean dataset for retrieval where `text` = the question only.

import pandas as pd
from pathlib import Path

# Adjust file names if yours are different
RAW_PATH = Path("../data/JEOPARDY_CSV.csv")   # the raw Kaggle CSV you extracted
OUT_PATH = Path("../data/dataset.csv")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase and replace spaces with underscores for robust column access."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def pick_first(row: dict, keys: list[str]) -> str:
    """Return the first non-empty value among the given keys."""
    for k in keys:
        v = str(row.get(k, "")).strip()
        if v:
            return v
    return ""


def main():
    # Some Jeopardy CSVs are latin-1; ignore bad lines to be robust
    df = pd.read_csv(RAW_PATH, encoding="latin-1", on_bad_lines="skip")
    df = normalize_columns(df)

    # Try to pull canonical fields from common variants
    # question: "question" / "clue"
    # category: "category" / "cat"
    # answer:   "answer" / "response"
    df["__question__"] = df.apply(
        lambda r: pick_first(r.to_dict(), ["question", "clue"]), axis=1
    )
    df["__category__"] = df.apply(
        lambda r: pick_first(r.to_dict(), ["category", "cat"]), axis=1
    )
    df["__answer__"] = df.apply(
        lambda r: pick_first(r.to_dict(), ["answer", "response"]), axis=1
    )

    # Keep a tiny sample (≤ 200 rows as per the assignment)
    n = min(200, len(df))
    sample = df.sample(n=n, random_state=42).copy()

    # The key point: text = QUESTION ONLY  (better semantic retrieval)
    sample["text"] = sample["__question__"].fillna("").str.strip()

    # Optional light metadata to show in UI if needed
    out = sample[["text", "__category__", "__answer__"]].rename(
        columns={
            "__category__": "meta_category",
            "__answer__": "meta_answer",
        }
    )

    # Drop empty rows if any
    out = out[out["text"] != ""]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"✅ Saved {len(out)} rows to {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
