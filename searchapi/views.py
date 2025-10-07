# views.py
from __future__ import annotations

import os
import gc
import glob
import json
from typing import List, Tuple, Any, Optional, Dict

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET

from pymongo import MongoClient
from dotenv import load_dotenv
import joblib
from sentence_transformers import SentenceTransformer


load_dotenv()

# =============================================================================
# Configuration (match training)
# =============================================================================
PICKLE_CHUNKS_DIR = getattr(
    settings,
    "PICKLE_CHUNKS_DIR",
    os.path.join(settings.BASE_DIR, ".", "pickle_chunks_weighted_sparse_hybrid"),
)

WEIGHTS = {
    "tfidf_cmp_name": 1.0,
    "seniority": 2.0,
    "minilm_title": 1.0,
}

VECTORIZER_PKL = getattr(
    settings, "VECTORIZER_PKL",
    os.path.join(settings.BASE_DIR, "tfidf_vectorizer_cmpname11.pkl")
)
LABELENC_PKL = getattr(
    settings, "LABELENC_PKL",
    os.path.join(settings.BASE_DIR, "seniority_encoder11.pkl")
)

# Mongo collection (prefer injected)
collection = getattr(settings, "COLLECTION", None)
if collection is None:
    MONGO_URI = getattr(settings, "MONGO_URI", os.getenv("MONGO_URI"))
    MONGO_DB_NAME = getattr(settings, "MONGO_DB_NAME", os.getenv("MONGO_DB_NAME", "betadb_06062023"))
    PERSON_COLL_NAME = getattr(settings, "PERSON_COLL_NAME", "person")
    if not MONGO_URI:
        raise RuntimeError("MONGO_URI not configured.")
    _client = MongoClient(MONGO_URI)
    _db = _client[MONGO_DB_NAME]
    collection = _db[PERSON_COLL_NAME]

# =============================================================================
# Load artifacts
# =============================================================================
loaded_tfidf = getattr(settings, "LOADED_TFIDF", None) or joblib.load(VECTORIZER_PKL)
label_encoder = getattr(settings, "label_encoder", None) or joblib.load(LABELENC_PKL)

# MiniLM model for jb_title
minilm = SentenceTransformer("all-MiniLM-L6-v2")

# def _load_npz_chunks(base_dir: str) -> Tuple[Optional[sp.csr_matrix], List[Any]]:
#     npz_paths = sorted(glob.glob(os.path.join(base_dir, "company_data_weighted_part_*.npz")))
#     if not npz_paths:
#         print(f"[loader] No .npz chunks found in {base_dir}")
#         return None, []

#     mats: List[sp.csr_matrix] = []
#     ids_all: List[Any] = []
#     for npz_path in npz_paths:
#         ids_path = npz_path.replace(".npz", "_ids.json")
#         try:
#             X_chunk = sp.load_npz(npz_path).astype(np.float32).tocsr()
#             with open(ids_path, "r", encoding="utf-8") as f:
#                 ids = json.load(f)
#             if X_chunk.shape[0] != len(ids):
#                 print(f"[loader] WARN size mismatch: {os.path.basename(npz_path)}  rows={X_chunk.shape[0]} vs ids={len(ids)}")
#             mats.append(X_chunk)
#             ids_all.extend(ids)
#             print(f"[loader] loaded {os.path.basename(npz_path)} rows={X_chunk.shape[0]} cols={X_chunk.shape[1]}")
#         except Exception as e:
#             print(f"[loader] ERROR loading {npz_path}: {e}")

#     if not mats:
#         return None, []
#     X = sp.vstack(mats, format="csr", dtype=np.float32)
#     return X, ids_all

# # Build KNN
# emb_matrix, uuid_list = _load_npz_chunks(PICKLE_CHUNKS_DIR)
# if emb_matrix is None or not uuid_list:
#     raise RuntimeError("No embeddings found. Check training output.")
# knn_model = NearestNeighbors(n_neighbors=50, metric="cosine", algorithm="auto")
# knn_model.fit(emb_matrix)
# del emb_matrix
# gc.collect()


def build_query_vector(cmp_name: str, lvl_norm: str, jb_title: str) -> sp.csr_matrix:
    """
    Hybrid query vector:
      Xq = [ TF-IDF(cmp_name) * w1 | LabelEnc(lvl_norm) * w2 | MiniLM(jb_title) * w3 ]
    """
    # TF-IDF(cmp_name)
    tfidf_vec = loaded_tfidf.transform([str(cmp_name or "").lower()])  # 1 x V
    if WEIGHTS["tfidf_cmp_name"] != 1.0:
        tfidf_vec = tfidf_vec.multiply(WEIGHTS["tfidf_cmp_name"]).tocsr()

    # LabelEncoder(lvl_norm)
    try:
        code = label_encoder.transform([str(lvl_norm or "").lower()]).astype(np.float32)[0]
    except Exception:
        code = np.float32(0.0)
    code *= np.float32(WEIGHTS["seniority"])
    lvl_col = sp.csr_matrix(([code], ([0], [0])), shape=(1, 1), dtype=np.float32)

    # MiniLM(jb_title) -> CSR
    title_emb = minilm.encode([str(jb_title or "").strip()], convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)
    title_emb *= np.float32(WEIGHTS["minilm_title"])  # [1 x 384]
    title_csr = sp.csr_matrix(title_emb)

    return sp.hstack([tfidf_vec, lvl_col, title_csr], format="csr", dtype=np.float32)

def _fetch_details(ids: List[Any]) -> Dict[Any, Dict[str, Any]]:
    if not ids: return {}
    projection = {
        "_id": 0,
        "id": 1,
        "cmp_name": 1,
        "jb_title": 1,
        "lvl_norm": 1,
        "cmp_keywords": 1,
        "relv_score": 1,
        "ln_url": 1,
        "cmp_ln_url": 1,
    }
    docs = list(collection.find({"id": {"$in": ids}}, projection))
    return {d.get("id"): d for d in docs}

# =============================================================================
# Endpoints
# =============================================================================
@csrf_exempt
def search_by_linkedin_api(request):
    """
    POST /aapi/search-by-linkedin/
    body: {"ln_url": "...", "k": 10}
    - Finds seed doc by LinkedIn URL (ln_url or cmp_ln_url)
    - Builds HYBRID query vector (TF-IDF cmp_name + lvl_norm + MiniLM jb_title)
    - KNN cosine search over trained hybrid embeddings
    - Fetches details from Mongo and returns results (sorted by relv_score, then similarity)
    """
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method is allowed."}, status=405)

    try:
        body = json.loads(request.body or "{}")
        ln_url = body.get("ln_url")
        k = int(body.get("k", 10))
        if not ln_url:
            return JsonResponse({"error": "ln_url is required."}, status=400)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)

    ln_url_norm = ln_url

    seed = collection.find_one(
        {"ln_url": ln_url_norm},
        {"_id": 0, "id": 1, "cmp_name": 1, "jb_title": 1, "lvl_norm": 1}
    ) or collection.find_one(
        {"cmp_ln_url": ln_url_norm},
        {"_id": 0, "id": 1, "cmp_name": 1, "jb_title": 1, "lvl_norm": 1}
    )
    if not seed:
        return JsonResponse({"error": f"No document found with LinkedIn URL {ln_url}"}, status=404)

    try:
        q_vec = build_query_vector(seed.get("cmp_name"), seed.get("lvl_norm"), seed.get("jb_title"))
    except Exception as e:
        return JsonResponse({"error": f"Failed to build query vector: {e}"}, status=500)

    try:
        distances, indices = knn_model.kneighbors(q_vec, n_neighbors=k)
    except Exception as e:
        return JsonResponse({"error": f"KNN failed: {e}"}, status=500)

    idxs = indices[0].tolist()
    dists = distances[0].tolist()
    sims = [float(1.0 - d) for d in dists]

    matched_ids = []
    for i in idxs:
        if 0 <= i < len(uuid_list):
            matched_ids.append(uuid_list[i])

    if not matched_ids:
        return JsonResponse({"error": "No similar profiles found."}, status=404)

    details_map = _fetch_details(matched_ids)

    results = []
    for rank, (row_idx, prof_id, sim) in enumerate(zip(idxs, matched_ids, sims), start=1):
        d = details_map.get(prof_id, {})
        results.append({
            "rank": rank,
            "uuid": prof_id,
            "similarity": sim,
            "cmp_name": d.get("cmp_name", ""),
            "jb_title": d.get("jb_title", ""),
            "lvl_norm": d.get("lvl_norm", ""),
            "cmp_keywords": d.get("cmp_keywords", ""),
            "relv_score": d.get("relv_score", 0),
            "ln_url": d.get("ln_url", "") or d.get("cmp_ln_url", ""),
        })

    return JsonResponse({"results": results}, status=200)

@require_GET
def api_usage_stats(request):
    try:
        middleware_path = next(m for m in settings.MIDDLEWARE if m.endswith("APIRequestMiddleware"))
        module_name, cls_name = middleware_path.rsplit(".", 1)
        middleware_mod = __import__(module_name, fromlist=[cls_name])
        middleware_instance = getattr(middleware_mod, cls_name)(None)
    except Exception as e:
        return JsonResponse({"error": f"Stats middleware not found/loaded: {e}"}, status=500)

    monitored = ["/aapi/search/", "/aapi/search-by-linkedin/", "/aapi/stats/"]
    try:
        stats = {"timestamp": np.datetime_as_string(np.datetime64("now"), unit="s"),
                 "requests_by_endpoint": []}
        for ep in monitored:
            s = middleware_instance.get_endpoint_stats(ep)
            if s:
                stats["requests_by_endpoint"].append(s)

        total = sum(x["request_count"] for x in stats["requests_by_endpoint"])
        succ = sum(x["metrics"].get("total_success", 0) for x in stats["requests_by_endpoint"])
        errs = sum(x["metrics"].get("total_errors", 0) for x in stats["requests_by_endpoint"])
        stats["summary"] = {
            "total_requests": total,
            "overall_success_rate": (succ / total) if total > 0 else 1.0,
            "overall_error_rate": (errs / total) if total > 0 else 0.0,
            "endpoints_monitored": len(monitored),
        }
        return JsonResponse(stats)
    except Exception as e:
        return JsonResponse({"error": f"Failed to compute stats: {e}"}, status=500)
