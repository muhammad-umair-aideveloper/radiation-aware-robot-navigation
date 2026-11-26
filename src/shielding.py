import numpy as np


def attenuation_matrix(wall, mu, samples=120):
    """Return atten[p, c] = exp(-mu * wall_length(p -> c)) for all flat cell pairs.

    p and c are flat cell indices (row * width + col). wall_length is the length of the
    straight segment from cell p to cell c that lies inside wall cells (sampled at `samples`
    points). Radiation is attenuated only by walls; mu is the attenuation per cell of wall.
    """
    wall = np.asarray(wall, dtype=bool)
    if wall.ndim != 2:
        raise ValueError("wall must be two-dimensional")
    if not np.isfinite(mu) or mu < 0.0:
        raise ValueError("mu must be non-negative and finite")
    if not isinstance(samples, (int, np.integer)) or samples < 1:
        raise ValueError("samples must be a positive integer")
    h, w = wall.shape
    pts = np.argwhere(np.ones((h, w), dtype=bool)).astype(float)
    ts = np.linspace(0.0, 1.0, samples)[None, :, None]
    out = np.empty((len(pts), len(pts)), dtype=np.float32)
    for p in range(len(pts)):
        d = pts - pts[p]
        pos = pts[p][None, None, :] + d[:, None, :] * ts
        ri = np.clip(np.rint(pos[..., 0]).astype(int), 0, h - 1)
        ci = np.clip(np.rint(pos[..., 1]).astype(int), 0, w - 1)
        length = np.hypot(d[:, 0], d[:, 1]) * wall[ri, ci].mean(axis=1)
        out[p] = np.exp(-mu * length)
    return out


def shielded_field(sources, wall, atten, softening=2.0):
    """Radiation field with wall shielding. sources = [(row, col, intensity), ...]."""
    wall = np.asarray(wall, dtype=bool)
    if wall.ndim != 2:
        raise ValueError("wall must be two-dimensional")
    if not np.isfinite(softening) or softening <= 0.0:
        raise ValueError("softening must be positive and finite")
    h, w = wall.shape
    atten = np.asarray(atten)
    if atten.shape != (h * w, h * w):
        raise ValueError(f"atten must have shape ({h * w}, {h * w})")
    rows, cols = np.indices((h, w))
    field = np.zeros((h, w), dtype=float)
    for sr, sc, i0 in sources:
        if not (0 <= sr < h and 0 <= sc < w):
            raise ValueError("source coordinates must be within the wall shape")
        if not np.isfinite(i0) or i0 < 0.0:
            raise ValueError("source intensity must be non-negative and finite")
        field += i0 / ((rows - sr) ** 2 + (cols - sc) ** 2 + softening) * atten[sr * w + sc].reshape(h, w)
    return field
