import os
import glob
import json
import time
from typing import List, Any, Tuple, Optional

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from django.conf import settings


_knn_model: Optional[NearestNeighbors] = None
_uuid_list: List[Any] = []


def _load_npz_chunks(base_dir: str) -> Tuple[Optional[sp.csr_matrix], List[Any]]:
    npz_paths = sorted(glob.glob(os.path.join(base_dir, "company_data_weighted_part_*.npz")))
    if not npz_paths:
        return None, []

    mats: List[sp.csr_matrix] = []
    ids_all: List[Any] = []
    total_rows = 0
    t0 = time.time()
    print(f"[knn_loader] Loading {len(npz_paths)} chunks from {base_dir}")
    for idx, npz_path in enumerate(npz_paths, start=1):
        ids_path = npz_path.replace(".npz", "_ids.json")
        try:
            c0 = time.time()
            X_chunk = sp.load_npz(npz_path).astype(np.float32).tocsr()
            with open(ids_path, "r", encoding="utf-8") as f:
                ids = json.load(f)
            mats.append(X_chunk)
            ids_all.extend(ids)
            total_rows += X_chunk.shape[0]
            c1 = time.time()
            print(f"[knn_loader] loaded {os.path.basename(npz_path)} rows={X_chunk.shape[0]} cols={X_chunk.shape[1]} in {c1 - c0:.2f}s ({idx}/{len(npz_paths)})")
        except Exception as e:
            raise RuntimeError(f"Failed loading chunk {os.path.basename(npz_path)}: {e}")

    if not mats:
        return None, []

    X = sp.vstack(mats, format="csr", dtype=np.float32)
    t1 = time.time()
    print(f"[knn_loader] Stacked matrix shape={X.shape} total_rows={total_rows} chunks={len(npz_paths)} load_time={t1 - t0:.2f}s")
    return X, ids_all


def load_knn_if_needed() -> Tuple[NearestNeighbors, List[Any]]:
    global _knn_model, _uuid_list
    if _knn_model is not None and _uuid_list:
        return _knn_model, _uuid_list

    base_dir = getattr(
        settings,
        "PICKLE_CHUNKS_DIR",
        os.path.join(settings.BASE_DIR, ".", "pickle_chunks_weighted_sparse_hybrid"),
    )

    emb_matrix, ids = _load_npz_chunks(base_dir)
    if emb_matrix is None or not ids:
        raise RuntimeError("No embeddings found. Ensure .npz chunks and *_ids.json exist.")

    model = NearestNeighbors(n_neighbors=50, metric="cosine", algorithm="auto")
    f0 = time.time()
    model.fit(emb_matrix)
    f1 = time.time()
    print(f"[knn_loader] Fitted NearestNeighbors(n_neighbors=50, metric='cosine') in {f1 - f0:.2f}s")

    # Free memory of the large matrix after fitting
    del emb_matrix

    _knn_model = model
    _uuid_list = ids
    return _knn_model, _uuid_list


# Initialize on module import once
knn_model, uuid_list = load_knn_if_needed()


