"""
The 300-Year Old Tiger Nest Painting - shredded scroll reassembly.

Reads the shuffled vertical strips from a folder named `input`, finds their
original left-to-right order from edge compatibility, and writes next to this
script:
    reconstructed_image.png  - the reassembled image
    reconstructed_order.json - list of strip filenames, left to right

Method (same idea as the notebook):
  1. For every strip take its left-most and right-most pixel columns.
  2. Score every ordered pair (i -> j) by how well the right edge of i
     continues into the left edge of j:
       - zero-mean cosine similarity of the two edge columns (per channel),
       - plus a gradient-prediction error (extrapolate i's last two columns
         one pixel to the right, and j's first two columns one pixel to the
         left, and compare).
  3. Greedily accept the most compatible pairs, never giving a strip two
     right/left neighbours and never closing a cycle (union-find), until one
     chain covering all strips remains.
  4. Walk the chain from the strip with no left neighbour.
  5. Cut the chain at its weakest joins and re-order those pieces with an
     exact search (Held-Karp), which fixes pieces placed in the wrong global
     order on flat regions.

Dependencies: numpy, opencv-python (see requirements.txt).
"""

import glob
import json
import os
import sys

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def find_input_dir():
    for cand in (os.path.join(SCRIPT_DIR, "input"), os.path.join(os.getcwd(), "input")):
        if os.path.isdir(cand):
            return cand
    sys.exit("ERROR: folder 'input' not found next to main.py or in the working directory.")


def load_strips(input_dir):
    paths = sorted(glob.glob(os.path.join(input_dir, "**", "*.png"), recursive=True))
    if not paths:
        sys.exit(f"ERROR: no .png strips found in {input_dir}")
    names, imgs = [], []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            sys.exit(f"ERROR: could not read {p}")
        if img.ndim == 2:
            img = img[:, :, None]
        names.append(os.path.basename(p))
        imgs.append(img)
    shapes = {im.shape for im in imgs}
    if len(shapes) != 1:
        sys.exit(f"ERROR: strips have different shapes: {shapes}")
    return names, np.stack(imgs, axis=0)


def compatibility(strips):
    """Return an (N, N) cost matrix; cost[i, j] small => j sits right of i."""
    s = strips[..., :3].astype(np.float64)          # ignore alpha for matching
    n, h, w, c = s.shape

    left = s[:, :, 0, :]                            # (N, H, C)
    right = s[:, :, -1, :]

    # (a) zero-mean cosine similarity per channel (the notebook's metric)
    def unit(e):
        e = e - e.mean(axis=1, keepdims=True)
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)

    lu, ru = unit(left), unit(right)
    cos = np.zeros((n, n))
    for ch in range(c):
        cos += ru[:, :, ch] @ lu[:, :, ch].T
    cos /= c                                        # in [-1, 1], higher = better

    # (b) gradient-prediction error (symmetric), root-mean-square difference
    if w >= 2:
        pred_r = 2 * s[:, :, -1, :] - s[:, :, -2, :]  # i's guess of the next column
        pred_l = 2 * s[:, :, 0, :] - s[:, :, 1, :]    # j's guess of the previous column
    else:
        pred_r, pred_l = right, left
    R = pred_r.reshape(n, -1)
    L = left.reshape(n, -1)
    PL = pred_l.reshape(n, -1)
    RR = right.reshape(n, -1)
    # pairwise mean squared error via the expansion |a-b|^2 = |a|^2 + |b|^2 - 2ab
    def pair_mse(A, B):
        a2 = (A ** 2).sum(1)[:, None]
        b2 = (B ** 2).sum(1)[None, :]
        return np.maximum(a2 + b2 - 2 * A @ B.T, 0) / A.shape[1]

    pred_err = np.sqrt(pair_mse(R, L)) + np.sqrt(pair_mse(RR, PL))

    # Normalise both terms per row by the best alternative so they are comparable
    def rel(cost):
        c2 = cost.copy()
        np.fill_diagonal(c2, np.inf)
        srt = np.sort(c2, axis=1)
        scale = srt[:, 1:2] + 1e-9                  # 2nd best for each i
        return c2 / scale

    cost = rel(pred_err) + rel(1.0 - cos + 1e-6)
    np.fill_diagonal(cost, np.inf)
    return cost


def greedy_chain(cost):
    n = cost.shape[0]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    right_of = np.full(n, -1)
    left_of = np.full(n, -1)
    links = 0
    for flat in np.argsort(cost, axis=None):
        if links == n - 1:
            break
        i, j = divmod(int(flat), n)
        if i == j or right_of[i] != -1 or left_of[j] != -1:
            continue
        ri, rj = find(i), find(j)
        if ri == rj:                                # would close a cycle
            continue
        right_of[i], left_of[j] = j, i
        parent[ri] = rj
        links += 1

    start = int(np.where(left_of == -1)[0][0])
    order = [start]
    while right_of[order[-1]] != -1:
        order.append(int(right_of[order[-1]]))
    assert len(order) == n, "chain does not cover all strips"
    return order


def refine_segments(order, cost, k=12):
    """Cut the chain at its k weakest joins and re-order the pieces optimally
    (Held-Karp over segments). Fixes pieces that were chained in the wrong
    global order, e.g. on flat regions such as sky."""
    n = len(order)
    if n < 3:
        return order
    joins = np.array([cost[order[t], order[t + 1]] for t in range(n - 1)])
    k = min(k, n - 1)
    cuts = sorted(np.argsort(joins)[::-1][:k])
    segs, prev = [], 0
    for c in cuts:
        segs.append(order[prev:c + 1])
        prev = c + 1
    segs.append(order[prev:])
    m = len(segs)

    jc = np.array([[cost[segs[a][-1], segs[b][0]] if a != b else np.inf
                    for b in range(m)] for a in range(m)])
    full = (1 << m) - 1
    INF = np.inf
    dp = np.full((1 << m, m), INF)
    back = np.full((1 << m, m), -1, dtype=np.int64)
    for a in range(m):
        dp[1 << a, a] = 0.0
    for mask in range(1, 1 << m):
        row = dp[mask]
        for last in range(m):
            v = row[last]
            if v == INF:
                continue
            for nxt in range(m):
                if mask & (1 << nxt):
                    continue
                nm = mask | (1 << nxt)
                nv = v + jc[last, nxt]
                if nv < dp[nm, nxt]:
                    dp[nm, nxt] = nv
                    back[nm, nxt] = last
    last = int(np.argmin(dp[full]))
    seq, mask = [], full
    while last != -1:
        seq.append(last)
        prev_last = int(back[mask, last])
        mask ^= 1 << last
        last = prev_last
    seq.reverse()
    return [s for idx in seq for s in segs[idx]]


def main():
    input_dir = find_input_dir()
    names, strips = load_strips(input_dir)
    print(f"Loaded {len(names)} strips of shape {strips.shape[1:]} from {input_dir}")

    cost = compatibility(strips)
    order = refine_segments(greedy_chain(cost), cost)
    ordered_names = [names[k] for k in order]
    image = np.concatenate([strips[k] for k in order], axis=1)

    img_path = os.path.join(SCRIPT_DIR, "reconstructed_image.png")
    json_path = os.path.join(SCRIPT_DIR, "reconstructed_order.json")
    cv2.imwrite(img_path, image)
    with open(json_path, "w") as f:
        json.dump(ordered_names, f, indent=4)

    print(f"Saved {img_path}  ({image.shape[1]}x{image.shape[0]})")
    print(f"Saved {json_path}")
    print("First strips:", ordered_names[:5])


if __name__ == "__main__":
    main()
