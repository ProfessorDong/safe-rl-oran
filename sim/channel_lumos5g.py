"""
channel_lumos5g.py
------------------
Loads the Lumos5G dataset (Narayanan et al., IMC 2020) and exposes a
per-slot SINR/throughput multiplier used by the env's channel realization.

Without this loader, env.step() uses log-normal jitter on mu_max
(channel_synthetic baseline). With Lumos5G, env.step() draws from real
mmWave 5G throughput-variation patterns calibrated to a normalized
multiplier in [0.1, 2.0].
"""
from __future__ import annotations
import os
import numpy as np
from typing import Tuple
from .config import SimCfg


LUMOS_REL = "lumos5g.csv"
CACHE_NAME = "lumos5g_cache.npz"


def _try_load(cfg: SimCfg) -> np.ndarray | None:
    """Return a 1-D NumPy array of normalized throughput multipliers
    (mean = 1.0, capped at [0.1, 2.0]), or None if the dataset is absent."""
    csv = os.path.join(cfg.data_dir, LUMOS_REL)
    if not os.path.exists(csv):
        return None
    try:
        import pandas as pd
    except ImportError:
        return None
    try:
        df = pd.read_csv(csv)
        # Use Throughput column. Median ~80 Mbps.
        thr = df["Throughput"].dropna().astype(float).values
        if len(thr) == 0:
            return None
        m = np.median(thr) if np.median(thr) > 0 else thr.mean()
        mult = np.clip(thr / max(m, 1e-6), 0.1, 2.0).astype(np.float32)
        return mult
    except Exception as e:
        print(f"[channel_lumos5g] load failed: {e}")
        return None


def load_channel_multipliers(cfg: SimCfg) -> Tuple[np.ndarray, str]:
    """Return (channel_mult_pool, source_tag).
    channel_mult_pool is a 1-D array to be sampled per-slot per-cell.
    """
    cache = os.path.join(cfg.data_dir, CACHE_NAME)
    if os.path.exists(cache):
        try:
            d = np.load(cache, allow_pickle=False)
            return d["pool"], str(d["source"])
        except Exception:
            pass

    real = _try_load(cfg)
    if real is not None:
        os.makedirs(cfg.data_dir, exist_ok=True)
        np.savez(cache, pool=real, source=np.array("lumos5g"))
        return real, "lumos5g"

    # Synthetic fallback: log-normal noise approximating mmWave variation.
    rng = np.random.default_rng(0)
    pool = np.exp(rng.normal(0.0, 0.4, size=10000)).astype(np.float32)
    pool = np.clip(pool, 0.1, 2.0)
    pool = pool / pool.mean()
    os.makedirs(cfg.data_dir, exist_ok=True)
    np.savez(cache, pool=pool, source=np.array("synthetic"))
    return pool, "synthetic"
