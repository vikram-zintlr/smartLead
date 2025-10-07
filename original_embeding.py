import os
import json
import gc
import pickle
import numpy as np
from typing import Iterator, Tuple, List
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib
import scipy.sparse as sp

# =========================
# Config
# =========================
WEIGHTS = {
    'company_name': 3.0,
    'seniority': 2.0,
    'job_title': 2.5,
}

MONGO_URI = "mongodb://d45b403107444ccf06594dca9f8e35ae87d5bbd2075c391359f3c0b8e610014e:8e246659231eb0f37024a28751b449c83782445d14cf103065efd095942adf48@172.31.25.128:28151/admin?authMechanism=admin"
DB_NAME = "betadb_06062023_1"
COLLECTION_NAME = "person"

# Vectorizer sizing — tune for RAM/accuracy tradeoff
VECT_MAX_FEATURES = 200_000   # was 1,000,000 (very heavy)
VECT_MIN_DF = 2               # ignore ultra-rare tokens
VECT_MAX_DF = 0.8             # ignore very common tokens

CHUNK_SIZE = 5000             # transform/save in this many docs at a time
MONGO_BATCH_SIZE = 2000       # how many docs server returns per batch over the wire
SAVE_DIR = "pickle_chunks_weighted_sparse"  # will use .npz for sparse, + ids.json

os.makedirs(SAVE_DIR, exist_ok=True)

print("mongo connection")
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# =========================
# Helpers
# =========================
def weighted_concat(company_name: str, job_title: str) -> str:
    """
    Apply weights by repeating tokens (cheap, works with TF-IDF).
    Keep strings short: basic repeat with trailing spaces.
    """
    cn = ((company_name + " ") * int(WEIGHTS['company_name'])) if company_name else ""
    jt = ((job_title + " ") * int(WEIGHTS['job_title'])) if job_title else ""
    return (cn + jt).strip()

def stream_for_fit() -> Iterator[str]:
    """
    Stream weighted texts over the entire collection for vectorizer.fit.
    Avoids storing all_texts in memory.
    """
    cursor = collection.find(
        {},
        {"cmp_name": 1, "lvl_norm": 1, "jb_title": 1},
        no_cursor_timeout=True,
        batch_size=MONGO_BATCH_SIZE
    )
    try:
        for doc in cursor:
            company_name = str(doc.get("cmp_name", "")).lower()
            job_title = str(doc.get("jb_title", "")).lower()
            yield weighted_concat(company_name, job_title)
    finally:
        cursor.close()

def stream_for_label_fit() -> Iterator[str]:
    """
    Stream seniority labels to fit LabelEncoder without storing all in RAM.
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
    npz_path = os.path.join(SAVE_DIR, f"company_data_weighted_part_{chunk_idx+1}.npz")
    ids_path = os.path.join(SAVE_DIR, f"company_data_weighted_part_{chunk_idx+1}_ids.json")

    # Save sparse matrix (CSR) as float32
    sp.save_npz(npz_path, X.astype(np.float32), compressed=True)

    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(ids, f)

    print(f"✅ Saved chunk {chunk_idx+1}  ->  {npz_path}  (+ ids)")

# =========================
# 1) Fit vectorizer and label encoder via streaming
# =========================
print("Collecting streams to fit TF-IDF (streaming, no big list in RAM)...")

# Fit TF-IDF
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=VECT_MAX_FEATURES,
    min_df=VECT_MIN_DF,
    max_df=VECT_MAX_DF,
    dtype=np.float32,          # memory-friendly
    lowercase=True
)
vectorizer.fit(stream_for_fit())
print("✅ TF-IDF fitted.")

# Fit LabelEncoder for seniority
seniority_encoder = LabelEncoder()
seniority_encoder.fit(list(stream_for_label_fit()))  # LabelEncoder needs a finite list, but labels are small
print(f"✅ Seniority encoder fitted on {len(seniority_encoder.classes_)} classes.")

# Persist models
joblib.dump(vectorizer, "tfidf_vectorizer_weighted.pkl", compress=3)
joblib.dump(seniority_encoder, "seniority_encoder.pkl", compress=3)
print("✅ Saved vectorizer + label encoder.")

gc.collect()

# =========================
# 2) Transform and save in sparse chunks
# =========================
print("Processing chunks and writing sparse embeddings...")

chunk_idx = 0
for docs in batched_docs(CHUNK_SIZE):
    # Build weighted texts + encoded seniority without DataFrame
    texts = []
    seniorities = []
    ids = []

    for d in docs:
        ids.append(d.get("id"))
        company_name = str(d.get("cmp_name", "")).lower()
        job_title = str(d.get("jb_title", "")).lower()
        texts.append(weighted_concat(company_name, job_title))
        seniorities.append(str(d.get("lvl_norm", "")).lower())

    # Transform TF-IDF (sparse)
    tfidf_matrix = vectorizer.transform(texts)          # CSR float32 (set above)
    # Encode seniority
    seniority_codes = seniority_encoder.transform(seniorities).astype(np.float32)
    # Weight seniority and add as a sparse column
    seniority_col = sp.csr_matrix((seniority_codes * WEIGHTS['seniority'], 
                                   (np.arange(len(seniority_codes)), np.zeros(len(seniority_codes), dtype=int))),
                                  shape=(len(seniority_codes), 1), dtype=np.float32)

    # Final sparse embedding = [TF-IDF | seniority_code]
    X = sp.hstack([tfidf_matrix, seniority_col], format="csr", dtype=np.float32)

    # Persist sparse chunk + ids
    save_sparse_chunk(X, ids, chunk_idx)

    # Cleanup
    del texts, seniorities, ids, tfidf_matrix, seniority_codes, seniority_col, X, docs
    gc.collect()

    chunk_idx += 1

print("✅ All chunks processed with weighted sparse embeddings.")
