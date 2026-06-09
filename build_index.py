"""
build_index.py — Offline FAISS index builder for the Egyptian drugs dataset.

Run once locally (requires CSV + sentence-transformers), then commit faiss.index:

    python build_index.py

The index embeds the `combined` field (name_ar + name_en + ingredient) so runtime
query encoding matches indexed text.
"""

import gc
import os

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

CSV_PATH = os.getenv("EGYPT_DRUGS_CSV", "egypt_drugs_cleaned_utf8.csv")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss.index")
EMBED_MODEL_NAME = os.getenv(
    "EMBED_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))


def main() -> None:
    print(f"Loading {CSV_PATH}…")
    df = pd.read_csv(CSV_PATH).fillna("").astype(str)
    ingredient_col = "ingredient_clean" if "ingredient_clean" in df.columns else "active_ingredient"
    df["combined"] = (
        df.get("name_ar", pd.Series([""] * len(df)))
        + " "
        + df.get("name_en", pd.Series([""] * len(df)))
        + " "
        + df.get(ingredient_col, pd.Series([""] * len(df)))
    ).str.strip()

    texts = df["combined"].tolist()
    print(f"Encoding {len(texts)} rows with {EMBED_MODEL_NAME}…")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    vectors = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True).astype("float32")
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"✅ Wrote {FAISS_INDEX_PATH} — {index.ntotal} vectors")

    # Persist combined column back to CSV if missing (optional)
    if "combined" not in pd.read_csv(CSV_PATH, nrows=1).columns:
        out_csv = CSV_PATH.replace(".csv", "_with_combined.csv")
        df.to_csv(out_csv, index=False)
        print(f"ℹ️  Saved {out_csv} with combined column")

    del model, vectors, df
    gc.collect()


if __name__ == "__main__":
    main()
