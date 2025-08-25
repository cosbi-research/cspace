import time
import json,re
import math
import requests
import numpy as np
from typing import Callable, Iterable, List, Tuple, Optional, Dict, Any

PUBTATOR_SEARCH_URL = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/search/"

def _slugify_for_pubtator(name: str) -> str:
    """
    Convert a concept name to a PubTator-friendly token:
    - remove punctuation (commas, quotes, etc.)
    - replace hyphens with underscores
    - replace whitespace with underscores
    - strip leading/trailing underscores
    """
    if not isinstance(name, str):
        name = str(name)

    # replace hyphens with underscores
    text = name.replace("-", "_")

    # remove punctuation except underscores and alphanumerics
    text = re.sub(r"[^\w\s_]", "", text)

    # collapse whitespace into single underscore
    text = re.sub(r"\s+", "_", text)

    # collapse multiple underscores
    text = re.sub(r"_+", "_", text)

    return text.strip("_")

def _build_query_token(name: str, entity_prefix: str) -> str:
    """
    Build a PubTator fielded token like '@DISEASE_Langerhans_Cell_Sarcoma'
    """
    return f"{entity_prefix}{_slugify_for_pubtator(name)}"

def _pubtator_pair_count(
    name1: str,
    name2: str,
    *,
    entity_prefix: str = "@DISEASE_",
    api_key: Optional[str] = None,
    timeout: float = 20.0,
) -> Optional[int]:
    """
    Query PubTator for (name1 AND name2) and return the total 'count' (number of articles).
    Returns None if the query fails (network/API issue).
    """
    token1 = _build_query_token(name1, entity_prefix)
    token2 = _build_query_token(name2, entity_prefix)
    # Build 'text' query like '@DISEASE_Name1 AND @DISEASE_Name2'
    text_query = f"{token1} AND {token2}"

    params = {
        "format": "json",
        "text": text_query,
        # leave page_size default (10) since we only need the total count
    }
    if api_key:
        # Included if you have one; PubTator may ignore it if not needed.
        params["api_key"] = api_key

    try:
        r = requests.get(PUBTATOR_SEARCH_URL, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        # PubTator3 returns a top-level "count"
        return int(data.get("count", 0))
    except Exception:
        return None

def count_pairs_in_literature(
    pairs: Iterable[Tuple[float, int, int, float, float]],
    *,
    name_lookup: Callable[[int], str],
    entity_prefix: str = "@DISEASE_",
    api_key: Optional[str] = None,
    per_request_delay_s: float = 0.25,
    retries: int = 2,
    backoff_factor: float = 2.0,
) -> List[Dict[str, Any]]:
    """
    For each (i, j, cosine) in `pairs`, query PubTator3 and return:
    [
      {
        "i": int, "j": int,
        "name_i": str, "name_j": str,
        "cosine": float,
        "count": Optional[int]  # None if request failed
      },
      ...
    ]

    - `name_lookup(i)` should return the human-readable concept name (e.g., "Thoracic Neoplasms").
    - `entity_prefix` defaults to '@DISEASE_' but you can change it if your concepts are another type.
    """
    out: List[Dict[str, Any]] = []
    for (_, i, j, cosine, _) in pairs:
        name_i = name_lookup(i)
        name_j = name_lookup(j)

        # Retry with exponential backoff if needed
        delay = per_request_delay_s
        count: Optional[int] = None
        for attempt in range(retries + 1):
            count = _pubtator_pair_count(
                name_i, name_j,
                entity_prefix=entity_prefix,
                api_key=api_key,
            )
            if count is not None:
                break
            time.sleep(delay)
            delay *= backoff_factor

        out.append({
            "i": int(i), "j": int(j),
            "name_i": name_i, "name_j": name_j,
            "cosine": float(cosine),
            "count": None if count is None else int(count),
        })

        # Friendly pacing so we don't hammer the API
        time.sleep(per_request_delay_s)

    return out

def _average_ranks_with_ties(x: np.ndarray) -> np.ndarray:
    """
    Compute average ranks (1..n) with ties receiving the average of their positions.
    Pure NumPy implementation to avoid SciPy dependency.
    """
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)

    # Handle ties: find groups of equal values and average their ranks
    sorted_x = x[order]
    i = 0
    while i < len(sorted_x):
        j = i + 1
        while j < len(sorted_x) and sorted_x[j] == sorted_x[i]:
            j += 1
        if j - i > 1:
            avg = (i + 1 + j) / 2.0  # average of integer positions (1-based)
            ranks[order[i:j]] = avg
        i = j
    return ranks

def correlate_similarity_with_counts(
    records: List[Dict[str, Any]],
    *,
    drop_missing: bool = True
) -> Dict[str, float]:
    """
    Compute correlation between cosine similarity and literature counts.
    Returns a dict with Pearson's r and Spearman's rho.

    `records` is the output of `count_pairs_in_literature`.
    """
    cosines = np.array([r["cosine"] for r in records], dtype=np.float64)
    counts  = np.array([np.nan if r["count"] is None else r["count"] for r in records], dtype=np.float64)

    if drop_missing:
        mask = np.isfinite(cosines) & np.isfinite(counts)
        cosines = cosines[mask]
        counts  = counts[mask]

    if cosines.size < 2:
        return {"pearson_r": np.nan, "spearman_rho": np.nan}

    # Pearson
    # Guard constant vectors
    if np.allclose(cosines, cosines[0]) or np.allclose(counts, counts[0]):
        pearson_r = np.nan
    else:
        pearson_r = float(np.corrcoef(cosines, counts)[0, 1])

    # Spearman (rank correlation, tie-aware)
    r1 = _average_ranks_with_ties(cosines)
    r2 = _average_ranks_with_ties(counts)
    if np.allclose(r1, r1[0]) or np.allclose(r2, r2[0]):
        spearman_rho = np.nan
    else:
        spearman_rho = float(np.corrcoef(r1, r2)[0, 1])

    return {"pearson_r": pearson_r, "spearman_rho": spearman_rho}