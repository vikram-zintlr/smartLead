#!/usr/bin/env python3
import os
import json
import gc
from typing import Iterator, List

import numpy as np
import scipy.sparse as sp
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib
from sentence_transformers import SentenceTransformer

# =========================
# Config
# =========================
WEIGHTS = {
    "tfidf_cmp_name": 1.0,   # scale for TF-IDF block (cmp_name)
    "seniority": 2.0,        # scale for lvl_norm encoded column
    "minilm_title": 1.0,     # scale for MiniLM(job_title) block
}

MONGO_URI = os.getenv("MONGO_URI","mongodb://3402f86a8f1d7349340f6e2b155c193f90ef8d09a8287e960ee7dc46152bc23f:e4073cb35bd6a9f2219050739d4b2e3831e3e8a535533d8e557ee939399469fc@13.203.49.68:27720/?authMechanism=DEFAULT&authSource=admin")
DB_NAME = os.getenv("MONGO_DB_NAME", "betadb_06062023")
COLLECTION_NAME = os.getenv("MONGO_COLL", "person")

# Vectorizer sizing — tune for RAM/accuracy tradeoff
VECT_MAX_FEATURES = int(os.getenv("VECT_MAX_FEATURES", "200000"))
VECT_MIN_DF = int(os.getenv("VECT_MIN_DF", "2"))
VECT_MAX_DF = float(os.getenv("VECT_MAX_DF", "0.8"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "5000"))
MONGO_BATCH_SIZE = int(os.getenv("MONGO_BATCH_SIZE", "2000"))
SAVE_DIR = os.getenv("SAVE_DIR", "pickle_chunks_weighted_sparse_hybrid")

os.makedirs(SAVE_DIR, exist_ok=True)

print("mongo connection")
client = MongoClient(MONGO_URI)
print(client, "client")
db = client[DB_NAME]
print(db, "db")
collection = db[COLLECTION_NAME]
print(collection, "collection")

# =========================
# Helpers
# =========================
def stream_cmp_name_for_fit() -> Iterator[str]:
    """
    Stream ONLY cmp_name texts for vectorizer.fit (compact and RAM-safe).
    """
    cursor = collection.find(
        {},
        {"cmp_name": 1},
        no_cursor_timeout=True,
        batch_size=MONGO_BATCH_SIZE
    )
    try:
        for doc in cursor:
            yield str(doc.get("cmp_name", "")).lower()
    finally:
        cursor.close()

def stream_lvl_norm_for_fit() -> Iterator[str]:
    """
    Stream lvl_norm for LabelEncoder.fit (label vocab is small).
    """
    cursor = collection.find(
        {},
        {"lvl_norm": 1},
        no_cursor_timeout=True,
        batch_size=MONGO_BATCH_SIZE
    )
    try:
        for doc in cursor:
            yield str(doc.get("lvl_norm", "")).lower()
    finally:
        cursor.close()

def batched_docs(batch_size: int) -> Iterator[List[dict]]:
    """
    Stream documents in batches for transform/save.
    """
    batch = []
    cursor = collection.find(
        {},
        {"id": 1, "cmp_name": 1, "lvl_norm": 1, "jb_title": 1},
        no_cursor_timeout=True,
        batch_size=MONGO_BATCH_SIZE
    )
    try:
        for doc in cursor:
            batch.append(doc)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
    finally:
        cursor.close()

def save_sparse_chunk(X: sp.csr_matrix, ids: List, chunk_idx: int):
    """
    Save sparse embeddings and corresponding ids for this chunk.
    """
    base = os.path.join(SAVE_DIR, f"company_data_weighted_part_{chunk_idx+1}")
    npz_path = f"{base}.npz"
    ids_path = f"{base}_ids.json"

    sp.save_npz(npz_path, X.astype(np.float32), compressed=True)
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(ids, f)

    print(f"✅ Saved chunk {chunk_idx+1} -> {npz_path} (+ ids)")

# =========================
# 1) Fit TF-IDF(cmp_name) and LabelEncoder(lvl_norm)
# =========================
print("Fitting TF-IDF on cmp_name (streaming)...")
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=VECT_MAX_FEATURES,
    min_df=VECT_MIN_DF,
    max_df=VECT_MAX_DF,
    dtype=np.float32,
    lowercase=True,
)
vectorizer.fit(stream_cmp_name_for_fit())
print("✅ TF-IDF fitted.")

print("Fitting LabelEncoder on lvl_norm...")
seniority_encoder = LabelEncoder()
seniority_encoder.fit(list(stream_lvl_norm_for_fit()))
print(f"✅ LabelEncoder classes: {len(seniority_encoder.classes_)}")

# Persist artifacts
joblib.dump(vectorizer, "tfidf_vectorizer_cmpname11.pkl", compress=3)
joblib.dump(seniority_encoder, "seniority_encoder11.pkl", compress=3)
print("✅ Saved vectorizer + label encoder.")

# =========================
# 2) Prepare MiniLM for jb_title
# =========================
print("Loading MiniLM (all-MiniLM-L6-v2) for jb_title...")
minilm = SentenceTransformer("all-MiniLM-L6-v2")  # 384 dims
print("✅ MiniLM loaded.")

# =========================
# 3) Transform in batches and save hybrid CSR chunks
# =========================
print("Processing batches...")
chunk_idx = 0
for docs in batched_docs(CHUNK_SIZE):
    ids = []
    cmp_names = []
    lvl_norms = []
    jb_titles = []

    for d in docs:
        ids.append(d.get("id"))
        cmp_names.append(str(d.get("cmp_name", "")).lower())
        lvl_norms.append(str(d.get("lvl_norm", "")).lower())
        jb_titles.append(str(d.get("jb_title", "")).strip())

    # TF-IDF(cmp_name)
    tfidf_mat = vectorizer.transform(cmp_names)  # CSR [N x V]
    if WEIGHTS["tfidf_cmp_name"] != 1.0:
        tfidf_mat = tfidf_mat.multiply(WEIGHTS["tfidf_cmp_name"]).tocsr()

    # LabelEncoder(lvl_norm), scaled
    lvl_codes = seniority_encoder.transform(lvl_norms).astype(np.float32)
    lvl_codes *= np.float32(WEIGHTS["seniority"])
    lvl_col = sp.csr_matrix(
        (lvl_codes, (np.arange(len(lvl_codes)), np.zeros(len(lvl_codes), dtype=int))),
        shape=(len(lvl_codes), 1),
        dtype=np.float32,
    )

    # MiniLM(jb_title) dense -> scaled -> CSR
    if len(jb_titles) > 0:
        title_emb = minilm.encode(jb_titles, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False)
        title_emb = title_emb.astype(np.float32) * np.float32(WEIGHTS["minilm_title"])  # [N x 384]
        title_csr = sp.csr_matrix(title_emb)
        del title_emb
    else:
        title_csr = sp.csr_matrix((len(docs), 384), dtype=np.float32)

    # Final hybrid: [ TF-IDF | lvl_norm_col | MiniLM(title) ]
    X = sp.hstack([tfidf_mat, lvl_col, title_csr], format="csr", dtype=np.float32)

    # Save
    save_sparse_chunk(X, ids, chunk_idx)

    # Cleanup
    del ids, cmp_names, lvl_norms, jb_titles, tfidf_mat, lvl_codes, lvl_col, title_csr, X, docs
    gc.collect()

    chunk_idx += 1

print("✅ All chunks processed as hybrid embeddings (TF-IDF cmp_name + lvl_norm + MiniLM jb_title).")
