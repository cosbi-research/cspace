import numpy as np
import math
import heapq
import matplotlib.pyplot as plt
from typing import Tuple, Iterable, Optional

# ----------------------------
# Configuration (tune these)
# ----------------------------
BINS = 1000                  # histogram bins for similarity distribution
HIST_RANGE = (-1.0, 1.0)     # cosine similarity range (adjust if you know your bounds)
ROW_BLOCK = 1000             # number of rows to process at a time
TOPK = 50                    # how many near-1 and near-0 pairs to report
HI_Q = 0.999                 # high quantile threshold for "very similar"
LO_Q = 0.001                 # low quantile threshold for "very dissimilar"
NORMALIZED_DTYPE = np.float32  # dtype for normalized output on disk
OUT_MEMMAP_PATH = None         # e.g. "Wsq_normalized.dat" to write a memmap; None => return in RAM (careful!)

# ----------------------------
# Helpers
# ----------------------------
def iter_upper_triangle_blocks(n: int, row_block: int) -> Iterable[Tuple[int, int]]:
    """
    Yield [r0:r1) row blocks over the upper triangle (we'll still
    mask out the diagonal and lower part per-row).
    """
    for r0 in range(0, n, row_block):
        r1 = min(n, r0 + row_block)
        yield r0, r1

# We re-define with closure of global offsets for clarity:
def build_distribution(Wsq, bins=BINS, hist_range=HIST_RANGE, row_block=ROW_BLOCK):
    """
    Stream the upper triangle (j > i) of a symmetric cosine matrix Wsq to build a histogram.
    Filters NaNs/±inf. Returns (bin_edges, hist_counts, pdf, cdf, stats).
    """
    n = Wsq.shape[0]
    expected_pairs = n * (n - 1) // 2

    # Fixed bin edges; use these consistently everywhere (hist & normalization)
    bin_edges = np.linspace(hist_range[0], hist_range[1], bins + 1, dtype=np.float64)
    hist_counts = np.zeros(bins, dtype=np.int64)

    total_counted = 0
    nan_count = 0
    inf_count = 0
    min_seen = np.inf
    max_seen = -np.inf

    for r0 in range(0, n, row_block):
        r1 = min(n, r0 + row_block)
        block = Wsq[r0:r1, :]  # shape (r1-r0, n)

        # process each row i in [r0, r1)
        for i_local in range(r1 - r0):
            i = r0 + i_local
            if i + 1 >= n:
                continue
            vals = block[i_local, i + 1 : ]  # strict upper triangle for this row

            # Track raw min/max before filtering (for debugging)
            if vals.size:
                # Use finite mask to avoid NaN affecting min/max
                finite_mask = np.isfinite(vals)
                if finite_mask.any():
                    vfin = vals[finite_mask]
                    min_seen = min(min_seen, float(vfin.min()))
                    max_seen = max(max_seen, float(vfin.max()))
                nan_count += np.isnan(vals).sum()
                inf_count += np.isinf(vals).sum()

            # Keep only finite values
            mask = np.isfinite(vals)
            if not mask.any():
                continue
            v = vals[mask]

            # Accumulate histogram for this row slice
            c, _ = np.histogram(v, bins=bin_edges)
            hist_counts += c
            total_counted += v.size

    if total_counted == 0:
        raise ValueError("No finite off-diagonal entries found.")

    pdf = hist_counts.astype(np.float64) / float(total_counted)
    cdf = np.cumsum(pdf)

    # Numerical guard: clamp the last value to exactly 1.0 if it’s within tiny epsilon
    if 1.0 - cdf[-1] < 1e-12:
        cdf[-1] = 1.0

    # --- Diagnostics ---
    stats = {
        "n": n,
        "expected_pairs": expected_pairs,
        "counted_pairs": int(total_counted),
        "dropped_pairs": int(expected_pairs - total_counted),  # likely NaN/inf
        "nan_count": int(nan_count),
        "inf_count": int(inf_count),
        "min_seen": float(min_seen) if np.isfinite(min_seen) else None,
        "max_seen": float(max_seen) if np.isfinite(max_seen) else None,
        "cdf_last": float(cdf[-1]),
        "hist_range": tuple(hist_range),
        "bins": int(bins),
    }

    # Optional sanity plot
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    plt.figure()
    plt.plot(centers, pdf)
    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")
    plt.title("Distribution of cosine similarities (upper triangle, no diagonal)")
    plt.show()

    # Quick sanity prints (you can remove if too verbose)
    print("Distribution diagnostics:", stats)
    
    return bin_edges, hist_counts, pdf, cdf, stats
    
def cdf_normalizer(values: np.ndarray, bin_edges: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    """
    Map values to [0,1] using the empirical CDF defined by (bin_edges, cdf).
    Linear interpolation inside each bin.
    """
    # Find bin index for each value
    # np.searchsorted returns index in [0..len(bin_edges)]
    idx = np.searchsorted(bin_edges, values, side="right") - 1
    idx = np.clip(idx, 0, len(cdf) - 1)  # last bin aligns with last cdf entry

    # Compute left/right edges of the bin for interpolation
    left_edges = bin_edges[idx]
    right_edges = bin_edges[idx + 1]
    # cdf at left edge (previous bin)
    cdf_left = np.zeros_like(values, dtype=np.float64)
    cdf_left[idx > 0] = cdf[idx[idx > 0] - 1]
    cdf_right = cdf[idx]

    # Avoid divide-by-zero if bin width is zero (shouldn't happen with linspace)
    denom = np.where(right_edges > left_edges, right_edges - left_edges, 1.0)
    frac = (values - left_edges) / denom
    frac = np.clip(frac, 0.0, 1.0)

    # Linear interpolation between cdf_left and cdf_right inside the bin
    out = cdf_left + (cdf_right - cdf_left) * frac
    # Ensure bounds
    return np.clip(out, 0.0, 1.0).astype(np.float32)

def normalize_matrix_cdf(Wsq, bin_edges, cdf,
                         row_block=ROW_BLOCK,
                         out_memmap_path: Optional[str]=OUT_MEMMAP_PATH,
                         out_dtype=np.float32):
    """
    Write a normalized matrix (CDF mapping) into [0,1].
    - If out_memmap_path is provided: writes to disk (recommended for huge matrices).
    - Else returns a new in-RAM array (CAUTION: same size as Wsq!).
    """
    n = Wsq.shape[0]

    if out_memmap_path is not None:
        Wnorm = np.memmap(out_memmap_path, mode="w+", dtype=out_dtype, shape=(n, n))
    else:
        Wnorm = np.empty((n, n), dtype=out_dtype)

    for r0, r1 in iter_upper_triangle_blocks(n, row_block):
        block = Wsq[r0:r1, :]
        # Normalize full block first
        block_norm = cdf_normalizer(block.astype(np.float64, copy=False), bin_edges, cdf).astype(out_dtype, copy=False)
        # Copy into output
        Wnorm[r0:r1, :] = block_norm
        # Enforce symmetry & diagonal exactly 1 → normalized CDF of 1 should also be ~1, but set precisely:
        for i_local in range(r1 - r0):
            i = r0 + i_local
            Wnorm[i, i] = out_dtype(1.0)  # ensure exact
            # Mirror upper → lower to enforce symmetry (only need when r0==0? we just write entire rows anyway)
            # Already writing full row; symmetry holds if input was symmetric and mapping is monotonic.
            # If you need strict symmetry regardless, uncomment the following:
            # Wnorm[:, i] = Wnorm[i, :]

    if isinstance(Wnorm, np.memmap):
        Wnorm.flush()
    return Wnorm

def find_extremes(Wsq, k=TOPK, row_block=ROW_BLOCK) -> Tuple[list, list]:
    """
    Find k most similar (largest) and k most dissimilar (smallest) off-diagonal pairs (i, j, value).
    Scans only the upper triangle, returns lists sorted best→worst for each extreme.
    Uses two heaps for efficiency.
    """
    n = Wsq.shape[0]
    # For top-k largest, keep a min-heap of size k
    top_heap = []   # (value, i, j) as min-heap
    # For bottom-k smallest, keep a max-heap by pushing (-value, i, j)
    bot_heap = []   # (-value, i, j)

    def push_top(val, i, j):
        if len(top_heap) < k:
            heapq.heappush(top_heap, (val, i, j))
        else:
            if val > top_heap[0][0]:
                heapq.heapreplace(top_heap, (val, i, j))

    def push_bot(val, i, j):
        neg = -val
        if len(bot_heap) < k:
            heapq.heappush(bot_heap, (neg, i, j))
        else:
            if neg > bot_heap[0][0]:  # i.e., val < current worst (since neg bigger means smaller val)
                heapq.heapreplace(bot_heap, (neg, i, j))

    for r0, r1 in iter_upper_triangle_blocks(n, row_block):
        block = Wsq[r0:r1, :]
        for i_local in range(r1 - r0):
            i = r0 + i_local
            if i + 1 < n:
                row_vals = block[i_local, i + 1 : ]
                if row_vals.size:
                    # Top candidates in this slice
                    j_indices = np.arange(i + 1, n, dtype=np.int64)
                    for val, j in zip(row_vals, j_indices):
                        push_top(val, i, j)
                        push_bot(val, i, j)

    # Sort outputs: largest first for top, smallest first for bottom
    top = sorted(top_heap, key=lambda x: -x[0])
    bottom = sorted([(-neg, i, j) for (neg, i, j) in bot_heap], key=lambda x: x[0])

    return top, bottom

def quantile_thresholds_from_hist(bin_edges, hist_counts, cdf, lo_q=LO_Q, hi_q=HI_Q) -> Tuple[float, float]:
    """
    Compute approximate low/high similarity thresholds from histogram CDF.
    """
    pdf = hist_counts / hist_counts.sum()
    cdf_vals = np.cumsum(pdf)
    # Bin indices closest to quantiles
    lo_idx = np.searchsorted(cdf_vals, lo_q, side="left")
    hi_idx = np.searchsorted(cdf_vals, hi_q, side="left")
    lo_thr = bin_edges[max(lo_idx, 0)]
    hi_thr = bin_edges[min(hi_idx, len(bin_edges) - 2)]
    return lo_thr, hi_thr

def find_pairs_near_percentile(
    Wsq: np.ndarray,
    bin_edges: np.ndarray,
    cdf: np.ndarray,
    target_percentile: float,
    k: int = TOPK,
    row_block: int = ROW_BLOCK,
):
    """
    Find the top-k (i,j) off-diagonal pairs whose similarity is closest to a given percentile.
    Closeness is measured in percentile (CDF) space: abs(CDF(value) - target_percentile).

    Returns:
        results: list of tuples sorted by increasing percentile distance:
                 [(abs_diff, i, j, value, value_percentile), ...]
    """
    assert 0.0 <= target_percentile <= 1.0, "target_percentile must be in [0, 1]"
    n = Wsq.shape[0]

    # We'll keep the k best (smallest abs diff) using a max-heap via negative keys.
    # Heap entries: (-abs_diff, i, j, value, value_percentile)
    heap = []

    # Stream upper triangle in row blocks
    for r0, r1 in iter_upper_triangle_blocks(n, row_block):
        block = Wsq[r0:r1, :]  # shape (rows, n)

        # Compute percentile (CDF) for the whole block once, then slice rows
        block_q = cdf_normalizer(block.astype(np.float64, copy=False), bin_edges, cdf)  # float32 in [0,1]

        for i_local in range(r1 - r0):
            i = r0 + i_local
            if i + 1 >= n:
                continue

            vals = block[i_local, i + 1 :]
            qs   = block_q[i_local, i + 1 :]

            if vals.size == 0:
                continue

            # Mask non-finite values
            finite_mask = np.isfinite(vals)
            if not finite_mask.any():
                continue

            v = vals[finite_mask]
            q = qs[finite_mask]

            # Compute abs diff in percentile space
            diffs = np.abs(q.astype(np.float64) - float(target_percentile))

            # Push candidates
            j_indices = np.arange(i + 1, n, dtype=np.int64)[finite_mask]
            for d, val, qq, j in zip(diffs, v, q, j_indices):
                key = -d  # negative for max-heap behavior on a min-heap
                if len(heap) < k:
                    heapq.heappush(heap, (key, i, j, float(val), float(qq)))
                else:
                    # If this candidate is closer (smaller diff), replace the current worst
                    if key > heap[0][0]:  # since more negative = worse, greater means closer
                        heapq.heapreplace(heap, (key, i, j, float(val), float(qq)))

    # Convert heap to sorted list (best first)
    results = [(-key, i, j, val, qq) for (key, i, j, val, qq) in heap]
    results.sort(key=lambda x: x[0])  # sort by abs_diff ascending

    return results