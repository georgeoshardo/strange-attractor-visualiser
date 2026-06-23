import numpy as np


def binned_density(x: np.ndarray, y: np.ndarray, bins: int = 128) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0:
        return np.empty(0, dtype=np.float32)

    density = np.zeros(len(x), dtype=np.float32)
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        return density

    x_finite = x[finite]
    y_finite = y[finite]
    x_span = float(np.max(x_finite) - np.min(x_finite))
    y_span = float(np.max(y_finite) - np.min(y_finite))
    if x_span == 0.0 or y_span == 0.0:
        return density

    x_bins = _bin_indices(x_finite, float(np.min(x_finite)), x_span, bins)
    y_bins = _bin_indices(y_finite, float(np.min(y_finite)), y_span, bins)

    counts = np.zeros((bins, bins), dtype=np.float32)
    np.add.at(counts, (x_bins, y_bins), 1.0)
    smoothed = _smooth_counts(counts)

    finite_density = smoothed[x_bins, y_bins]
    finite_density = finite_density - float(np.min(finite_density))
    max_density = float(np.max(finite_density))
    if max_density > 0.0:
        finite_density = finite_density / max_density
    density[finite] = finite_density.astype(np.float32, copy=False)
    return density


def _bin_indices(values: np.ndarray, minimum: float, span: float, bins: int) -> np.ndarray:
    scaled = (values - minimum) / span
    indices = np.floor(scaled * bins).astype(int)
    return np.clip(indices, 0, bins - 1)


def _smooth_counts(counts: np.ndarray) -> np.ndarray:
    padded = np.pad(counts, 1, mode="edge")
    return (
        padded[:-2, :-2]
        + 2.0 * padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + 2.0 * padded[1:-1, :-2]
        + 4.0 * padded[1:-1, 1:-1]
        + 2.0 * padded[1:-1, 2:]
        + padded[2:, :-2]
        + 2.0 * padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / 16.0
