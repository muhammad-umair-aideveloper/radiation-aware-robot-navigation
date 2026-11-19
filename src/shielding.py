import numpy as np


def attenuation_matrix(wall, mu, samples=120):
    """Return atten[p, c] = exp(-mu * wall_length(p -> c)) for all flat cell pairs.

    p and c are flat cell indices (row * width + col). wall_length is the length of the
    straight segment from cell p to cell c that lies inside wall cells (sampled at `samples`
    points). Radiation is attenuated only by walls; mu is the attenuation per cell of wall.
    """
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
    h, w = wall.shape
    rows, cols = np.indices((h, w))
    field = np.zeros((h, w), dtype=float)
    for sr, sc, i0 in sources:
        field += i0 / ((rows - sr) ** 2 + (cols - sc) ** 2 + softening) * atten[sr * w + sc].reshape(h, w)
    return field
