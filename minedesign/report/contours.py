"""Marching-squares contour tracing on a raster grid.

Used to slice the pit floor surface into elevation-banded outline rings for the
3D viewer -- the report's equivalent of digitizing a series of "quick planes"
through a solid at fixed elevation intervals, a standard way to read a pit
shell's shape level by level rather than as one undifferentiated surface.
"""
from __future__ import annotations

import numpy as np

# Edge midpoints of a unit cell, keyed by corner letter.
_EDGE = {"L": (0.0, 0.5), "R": (1.0, 0.5), "B": (0.5, 0.0), "T": (0.5, 1.0)}

# Marching-squares case table: which edge pairs a boundary segment connects,
# indexed by the 4-bit corner-inside pattern (bl=1, br=2, tr=4, tl=8).
_CASES: dict[int, list[tuple[str, str]]] = {
    1: [("L", "B")], 2: [("B", "R")], 3: [("L", "R")],
    4: [("R", "T")], 5: [("L", "T"), ("B", "R")], 6: [("B", "T")],
    7: [("L", "T")], 8: [("T", "L")], 9: [("T", "B")],
    10: [("T", "R"), ("B", "L")], 11: [("T", "R")], 12: [("R", "L")],
    13: [("R", "B")], 14: [("B", "L")],
}


def trace_contours(mask: np.ndarray, min_points: int = 4) -> list[np.ndarray]:
    """Closed boundary loops of a boolean raster, in (row, col) grid coordinates.

    Returns loops longest first, so a ring's main outline comes before any
    small island the same level happens to also enclose.
    """
    mask = np.asarray(mask, bool)
    if not mask.any():
        return []

    padded = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), bool)
    padded[1:-1, 1:-1] = mask

    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    rows, cols = padded.shape
    for r in range(rows - 1):
        for c in range(cols - 1):
            case = (int(padded[r, c]) | (int(padded[r, c + 1]) << 1)
                    | (int(padded[r + 1, c + 1]) << 2) | (int(padded[r + 1, c]) << 3))
            for a, b in _CASES.get(case, []):
                pa = (r + _EDGE[a][1], c + _EDGE[a][0])
                pb = (r + _EDGE[b][1], c + _EDGE[b][0])
                segments.append((pa, pb))

    if not segments:
        return []

    def key(p):
        return (round(p[0], 4), round(p[1], 4))

    adjacency: dict[tuple, list[tuple]] = {}
    for pa, pb in segments:
        adjacency.setdefault(key(pa), []).append(key(pb))

    unvisited = dict(adjacency)
    loops: list[np.ndarray] = []
    while unvisited:
        start = next(iter(unvisited))
        loop = [start]
        seen = {start}
        current = start
        while True:
            options = unvisited.get(current)
            if not options:
                break
            nxt = options.pop()
            if not options:
                unvisited.pop(current, None)
            if nxt == start or nxt in seen:
                break
            loop.append(nxt)
            seen.add(nxt)
            current = nxt
        unvisited.pop(current, None)
        if len(loop) >= min_points:
            loops.append(np.array(loop, float) - 1.0)  # undo the padding offset

    loops.sort(key=len, reverse=True)
    return loops


def smooth_loop(loop: np.ndarray, window: int = 5, passes: int = 2) -> np.ndarray:
    """Moving-average smoothing around a closed loop.

    A raster boundary is a staircase; a contour line is not. This rounds the
    staircase without moving the line more than a fraction of a cell.
    """
    if len(loop) < window or window < 3:
        return loop
    out = np.asarray(loop, float)
    for _ in range(passes):
        padded = np.vstack([out[-(window // 2):], out, out[:window // 2]])
        kernel = np.ones(window) / window
        smoothed = np.column_stack([
            np.convolve(padded[:, 0], kernel, mode="valid"),
            np.convolve(padded[:, 1], kernel, mode="valid"),
        ])
        out = smoothed[:len(out)]
    return out


def loop_area(loop: np.ndarray) -> float:
    """Shoelace area of a closed loop, in squared grid units."""
    if len(loop) < 3:
        return 0.0
    a, b = loop[:, 0], loop[:, 1]
    return float(abs(np.dot(a, np.roll(b, -1)) - np.dot(b, np.roll(a, -1))) / 2.0)
