"""
arrivals.py
-----------
Per-cell arrival-rate loader. Three sources, in priority order:

  1. Shanghai Telecom (Yu et al. 2019, 6-month session-level dataset from
     a major Chinese city; per-row (start, end, lat, lon, user_id)).
     Aggregates session-start counts per cell per hour into a (B, 24)
     diurnal profile.

  2. C2TM (City Cellular Traffic Map) -- SJTU 2015, week-long hourly
     traffic for a Chinese city (Aug 2012). Per-base-station x per-hour
     records of bytes and active users.

  3. Synthetic fallback -- calibrated Poisson + Pareto-bursty with
     diurnal modulation tuned to look like real cellular busy-hour data.

The returned object is a per-slot (B, T) arrival array in Mbits.
"""
from __future__ import annotations
import os
import numpy as np
from typing import Tuple
from .config import SimCfg


CACHE_NAME = "arrivals_cache.npz"
C2TM_REL = "c2tm/traceset/cellular_traffic.csv"
SHANGHAI_DIR = "shanghai_telecom"
SHANGHAI_DEFAULT_FILE = "data_6.1~6.15.xlsx"


def _calibrated_synthetic_profile(B: int, rng: np.random.Generator) -> np.ndarray:
    """24-hour per-cell arrival-rate profile (B, 24)."""
    hours = np.arange(24)
    base = 0.35 + 0.30 * np.cos((hours - 20.0) * 2 * np.pi / 24.0) + \
        0.10 * np.sin((hours - 8.0) * 2 * np.pi / 24.0)
    base = np.clip(base, 0.10, 1.0)
    cell_scale = rng.uniform(0.7, 1.4, size=B)[:, None]
    return base[None, :] * cell_scale


def _try_load_shanghai(dir_path: str, B: int) -> np.ndarray | None:
    """Load one half-month of Shanghai Telecom and produce a (B, 24)
    per-cell hourly session-start profile, normalized to per-cell mean 1.
    """
    if not os.path.isdir(dir_path):
        return None
    try:
        import pandas as pd
    except ImportError:
        return None
    candidates = [SHANGHAI_DEFAULT_FILE] + sorted(
        f for f in os.listdir(dir_path) if f.endswith(".xlsx"))
    path = None
    for c in candidates:
        p = os.path.join(dir_path, c)
        if os.path.exists(p):
            path = p
            break
    if path is None:
        return None
    try:
        print(f"[arrivals] reading Shanghai file {os.path.basename(path)} "
              f"(this can take ~30-60s)")
        df = pd.read_excel(path,
                            usecols=["start time", "latitude", "longitude"])
        df.columns = ["start_time", "lat", "lon"]
        df = df.dropna(subset=["start_time", "lat", "lon"])
        df["hour"] = df["start_time"].dt.hour
        # Round (lat, lon) to 4 decimals (~11 m) for cell identification.
        df["cell"] = (df["lat"].round(4).astype(str) + "_" +
                       df["lon"].round(4).astype(str))
        # Pick top-B cells by total session count.
        totals = df.groupby("cell").size().sort_values(ascending=False)
        top_cells = totals.head(B).index.tolist()
        df = df[df["cell"].isin(top_cells)]
        prof = (df.groupby(["cell", "hour"]).size()
                  .unstack(fill_value=0)
                  .reindex(top_cells)
                  .reindex(columns=range(24), fill_value=0)
                  .values.astype(float))
        # Normalize so per-cell mean profile averages to 1.
        prof = prof / np.maximum(prof.mean(axis=1, keepdims=True), 1.0)
        return prof
    except Exception as e:
        print(f"[arrivals] Shanghai load failed: {e}; falling back.")
        return None


def _try_load_c2tm(csv_path: str, B: int) -> np.ndarray | None:
    """Load C2TM hourly traffic and return (B, 24) profile of mean bytes/hour
    for the top-B base stations. Normalize to mean 1."""
    if not os.path.exists(csv_path):
        return None
    try:
        import pandas as pd
    except ImportError:
        return None
    try:
        df = pd.read_csv(csv_path)
        totals = df.groupby("bs")["bytes"].sum().sort_values(ascending=False)
        top_bs = totals.head(B).index.tolist()
        df = df[df["bs"].isin(top_bs)].copy()
        df["dt"] = pd.to_datetime(df["time_hour"], unit="s", utc=True)
        df["hour"] = df["dt"].dt.hour
        prof = df.groupby(["bs", "hour"])["bytes"].mean().unstack(fill_value=0)
        prof = prof.reindex(top_bs).reindex(columns=range(24), fill_value=0)
        arr = prof.values.astype(float)
        arr = arr / np.maximum(arr.mean(axis=1, keepdims=True), 1.0)
        return arr
    except Exception as e:
        print(f"[arrivals] C2TM load failed: {e}; falling back.")
        return None


def load_profile(cfg: SimCfg, rng: np.random.Generator) -> Tuple[np.ndarray, str]:
    """Return (B, 24) per-cell hourly arrival-rate multiplier profile and a
    source tag ('shanghai', 'c2tm', or 'synthetic')."""
    B = cfg.topo.B
    cache = os.path.join(cfg.data_dir, CACHE_NAME)
    if os.path.exists(cache):
        try:
            d = np.load(cache, allow_pickle=False)
            if d["profile"].shape == (B, 24):
                return d["profile"], str(d["source"])
        except Exception:
            pass

    # Try Shanghai Telecom first (highest priority).
    sh_dir = os.path.join(cfg.data_dir, SHANGHAI_DIR)
    real = _try_load_shanghai(sh_dir, B)
    if real is not None:
        os.makedirs(cfg.data_dir, exist_ok=True)
        np.savez(cache, profile=real, source=np.array("shanghai"))
        return real, "shanghai"

    # Then C2TM.
    c2tm_path = os.path.join(cfg.data_dir, C2TM_REL)
    real = _try_load_c2tm(c2tm_path, B)
    if real is not None:
        os.makedirs(cfg.data_dir, exist_ok=True)
        np.savez(cache, profile=real, source=np.array("c2tm"))
        return real, "c2tm"

    syn = _calibrated_synthetic_profile(B, rng)
    os.makedirs(cfg.data_dir, exist_ok=True)
    np.savez(cache, profile=syn, source=np.array("synthetic"))
    return syn, "synthetic"


def generate_arrivals(cfg: SimCfg, T: int, seed: int) -> Tuple[np.ndarray, str]:
    """Produce (B, T) arrival sequence (Mbits per slot) for one episode.

    Episode time compression: an episode spans the full 24-hour profile,
    so each slot maps to (slot / T) * 24 hours. This lets the controller
    experience diurnal variation within a single training rollout.
    Per-cell rate-normalization ensures all cells reach the configured
    base_rate during their busy hour.
    """
    rng = np.random.default_rng(seed)
    profile, source = load_profile(cfg, rng)
    B = cfg.topo.B

    hours = (np.arange(T) * 24.0 / max(T, 1)).astype(int) % 24
    mult = profile[:, hours]
    cell_peak = profile.max(axis=1, keepdims=True)
    mult = mult / np.maximum(cell_peak, 1e-3)

    mean = cfg.arr.base_rate_Mb_per_slot * mult
    arrivals = rng.poisson(lam=np.maximum(mean, 1e-6) * 10.0).astype(float) / 10.0

    burst_mask = rng.random((B, T)) < cfg.arr.burst_prob
    burst_amp = (rng.pareto(cfg.arr.pareto_shape, size=(B, T)) + 1.0) * \
        cfg.arr.burst_scale
    arrivals = arrivals + burst_mask * burst_amp * mean

    return arrivals.astype(np.float32), source
