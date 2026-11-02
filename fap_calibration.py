#!/usr/bin/env python
"""Significance -> false-alarm-probability tables for the ZVAR short and long searches.

A source's significance is the maximum periodogram peak over many trial frequencies,
so for pure noise it follows a Gumbel distribution (GEV, shape 0). One (mu, beta) is
fitted per (band, n_bins), and FAP = the Gumbel survival function.

  short (5 min - 10 d)   Least-squares Gumbel fit to the noise hump of the best-peak
                         density, over the contiguous bins >= 2e-3 of the peak.
  long  (10 - 1000 d)    The files stop at significance 10, so there is no hump. The
                         fit is a Poisson MLE on absolute counts, n * width * pdf, over
                         a fitted window. The window tolerance is the one whose beta
                         best matches a window-free Gumbel + power-law reference.

Each table has columns significance, FAP, N_sigma (the one-sided Gaussian
equivalent), on a 0.1 grid: 0-200 (short) or 0-1000 (long).
    <out>/fap_{g,r}band_{05,10,20}bin.csv

Usage:
    python fap_calibration.py                 # both searches
    python fap_calibration.py --search short
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
from scipy.stats import genextreme, norm

ZVAR_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHORT_HIST = os.path.join(ZVAR_TOOLS, "histogram_data.csv")
LONG_HIST = os.path.join(ZVAR_TOOLS, "data", "zvar_long_sig_hist.csv")
SHORT_OUT = os.path.join(ZVAR_TOOLS, "fap_lookup_tables")
LONG_OUT = os.path.join(ZVAR_TOOLS, "fap_lookup_tables_long_fitwin")

KEYS = [(b, c) for b in "gr" for c in ("20bin", "10bin", "5bin")]
NBINS = {"20bin": 20, "10bin": 10, "5bin": 5}

# short
PEAK_FRAC = 2e-3          # fit window: contiguous bins with density >= PEAK_FRAC * peak

# long
MIN_LAMBDA = 25.0         # bins expecting fewer counts are ignored in window tests
MIX_DEV_MAX = 3.0         # reference fits worse than this (deviance per bin) are unused
TOL_GRID = np.array([0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 0.35, 0.50])
ZLO_GRID = np.round(np.arange(10.0, 16.01, 0.5), 3)
ZHI_GRID = np.round(np.arange(14.0, 60.01, 0.5), 3)


def load_hist(path):
    """(band, config) -> (bin centres, density, n_total)."""
    parse = lambda s: np.fromstring(s.strip().strip("[]"), sep=" ")
    return {(r.band, r.config): (parse(r.bin_centers), parse(r.density), int(r.n_total))
            for r in pd.read_csv(path).itertuples()}


def write_table(out_dir, band, cfg, grid, fap, floor):
    os.makedirs(out_dir, exist_ok=True)
    nsig = norm.isf(np.clip(fap, floor, 1.0 - 1e-15))
    pd.DataFrame({"significance": grid, "FAP": fap, "N_sigma": nsig}).to_csv(
        os.path.join(out_dir, f"fap_{band}band_{NBINS[cfg]:02d}bin.csv"), index=False)


def gumbel_pdf(z, mu, beta):
    t = (np.asarray(z, dtype=float) - mu) / beta
    return np.exp(-t - np.exp(-t)) / beta


# ------------------------------------------------------------ short period
def fit_short(bc, d, n):
    """Weighted least-squares Gumbel fit to the hump of the density histogram."""
    w, pk = float(bc[1] - bc[0]), int(np.argmax(d))
    lo = hi = pk
    while lo - 1 >= 0 and d[lo - 1] >= d[pk] * PEAK_FRAC:
        lo -= 1
    while hi + 1 < len(d) and d[hi + 1] >= d[pk] * PEAK_FRAC:
        hi += 1
    x, y = bc[lo:hi + 1], d[lo:hi + 1]
    sigma = np.sqrt(np.clip(y * w * n, 1.0, None)) / (w * n)   # Poisson, density units
    wt = y / y.sum()
    std = np.sqrt(np.sum(wt * (x - np.sum(x * wt)) ** 2))       # starting beta
    (mu, beta), _ = curve_fit(lambda z, mu, beta: genextreme.pdf(z, 0.0, loc=mu, scale=beta),
                              x, y, p0=[bc[pk], max(std, 0.3)], sigma=sigma,
                              bounds=([-np.inf, 1e-3], [np.inf, np.inf]), maxfev=40000)
    return mu, beta


def short_tables(hist_path, out_dir):
    grid = np.round(np.arange(0.0, 200.0 + 0.1, 0.1), 1)
    for (band, cfg), (bc, d, n) in load_hist(hist_path).items():
        mu, beta = fit_short(bc, d, n)
        sf = lambda z: np.clip(genextreme.sf(z, 0.0, loc=mu, scale=beta), 1e-300, 1.0)
        fap = np.where(grid < mu, sf(mu), sf(grid))   # below the mode, hold SF(mu)
        write_table(out_dir, band, cfg, grid, fap, 1e-300)
        print(f"short {band} {cfg:5s}  mu={mu:.4f}  beta={beta:.4f}")


# ------------------------------------------------------------- long period
def fit_long(h, z_lo, z_hi, p0=(0.0, 4.3)):
    """Poisson MLE of (mu, beta) on counts in [z_lo, z_hi]. beta is fitted in log."""
    c, k, n, w = h
    sel = (c >= z_lo) & (c <= z_hi)
    if sel.sum() < 8:
        return np.nan, np.nan
    x, kk = c[sel], k[sel].astype(float)

    def nll(p):
        lam = np.maximum(n * w * gumbel_pdf(x, p[0], np.exp(p[1])), 1e-300)
        return float(-np.sum(kk * np.log(lam) - lam))

    res = minimize(nll, [p0[0], np.log(p0[1])], method="Powell",
                   options=dict(xtol=1e-7, ftol=1e-9, maxfev=20000))
    return float(res.x[0]), float(np.exp(res.x[1]))


def window_scan(h):
    """(z_lo, z_hi) -> (mu, beta) over the window grid, warm-started along z_hi."""
    out = {}
    for z_lo in ZLO_GRID:
        p0 = (0.0, 4.3)
        for z_hi in ZHI_GRID[ZHI_GRID >= z_lo + 2.0]:
            mu, beta = fit_long(h, z_lo, z_hi, p0)
            if np.isfinite(beta):
                out[(z_lo, z_hi)] = p0 = (mu, beta)
    return out


def fitted_window(h, scan, tol, n_iter=8):
    """Iterate the window edges to a fixed point, then refit.

    The upper edge is the first bin more than tol above the model. The lower edge is
    the smallest edge whose lower half the model matches to within tol.
    """
    c, k, n, w = h
    snap = lambda g, z: g[np.argmin(np.abs(g - z))]
    get = lambda lo, hi: scan.get((snap(ZLO_GRID, lo), snap(ZHI_GRID, hi)), (np.nan, np.nan))
    z_lo, z_hi = ZLO_GRID[0], 20.0
    for _ in range(n_iter):
        mu, beta = get(z_lo, z_hi)
        if not np.isfinite(beta):
            break
        lam = n * w * gumbel_pdf(c, mu, beta)
        out = (c > z_lo + 2.0) & (lam >= MIN_LAMBDA)
        over = np.flatnonzero(k[out] / lam[out] - 1.0 > tol)
        new_hi = float(c[out][over[0]]) if over.size else ZHI_GRID[-1]
        new_hi = snap(ZHI_GRID, float(np.clip(new_hi, z_lo + 4.0, ZHI_GRID[-1])))

        new_lo = ZLO_GRID[-1]
        for cand in ZLO_GRID:
            mu2, beta2 = get(cand, new_hi)
            if not np.isfinite(beta2):
                continue
            sel = (c >= cand) & (c <= new_hi)
            x, kk = c[sel], k[sel].astype(float)
            lam2 = n * w * gumbel_pdf(x, mu2, beta2)
            half = (x <= cand + 0.5 * (new_hi - cand)) & (lam2 >= MIN_LAMBDA)
            if half.any() and np.max(np.abs(kk[half] / lam2[half] - 1.0)) <= tol:
                new_lo = float(cand)
                break
        if abs(new_hi - z_hi) < 1e-9 and abs(new_lo - z_lo) < 1e-9:
            break
        z_lo, z_hi = new_lo, new_hi
    return fit_long(h, max(z_lo, float(c[np.flatnonzero(k)[0]])), z_hi)


def mixture_reference(h):
    """Window-free Gumbel + power-law fit over the whole recorded range.

    Returns (beta, Poisson deviance per bin).
    """
    c, k, n, w = h
    nz = np.flatnonzero(k)
    sel = (c >= c[nz[0]]) & (np.arange(len(c)) <= nz[-1])
    x, kk = c[sel], k[sel].astype(float)
    lam_of = lambda p: n * w * (gumbel_pdf(x, p[0], np.exp(p[1]))
                                + np.exp(p[2]) * (x / 20.0) ** (-np.exp(p[3])))
    nll = lambda p: float(-np.sum(kk * np.log(np.maximum(lam_of(p), 1e-300)) - lam_of(p)))
    best = min((minimize(nll, [0.0, np.log(b0), a0, np.log(2.5)], method="Powell",
                         options=dict(xtol=1e-8, ftol=1e-10, maxfev=100000))
                for b0 in (3.8, 4.3, 4.8) for a0 in (-12.0, -15.0)),   # multi-start
               key=lambda r: r.fun)
    lam = np.maximum(lam_of(best.x), 1e-300)
    dev = 2 * np.sum(np.where(kk > 0, kk * np.log(np.maximum(kk, 1e-300) / lam), 0.0)
                     - (kk - lam)) / max(int(sel.sum()) - 4, 1)
    return float(np.exp(best.x[1])), float(dev)


def long_tables(hist_path, out_dir):
    hist = {}
    for key, (c, d, n) in load_hist(hist_path).items():
        w = float(c[1] - c[0])
        hist[key] = (c, np.round(d * w * n).astype(np.int64), n, w)   # density -> counts

    # pick the window tolerance that best reproduces the reference betas
    scans = {key: window_scan(hist[key]) for key in KEYS}
    fits = {tol: {key: fitted_window(hist[key], scans[key], tol) for key in KEYS}
            for tol in TOL_GRID}
    mix = {key: mixture_reference(hist[key]) for key in KEYS}
    usable = [key for key in KEYS if mix[key][1] < MIX_DEV_MAX]
    rms = [np.sqrt(np.mean([np.log(fits[tol][key][1] / mix[key][0]) ** 2 for key in usable]))
           for tol in TOL_GRID]
    tol_star = TOL_GRID[int(np.argmin(rms))]
    print(f"long tol = {tol_star:g} (reference: {usable})")

    grid = np.round(np.arange(0.0, 1000.0 + 1e-9, 0.1), 1)
    tiny = np.finfo(float).tiny
    for (band, cfg), (mu, beta) in fits[tol_star].items():
        fap = np.clip(-np.expm1(-np.exp(-(grid - mu) / beta)), tiny, 1.0)   # Gumbel SF
        write_table(out_dir, band, cfg, grid, fap, tiny)
        print(f"long  {band} {cfg:5s}  mu={mu:.4f}  beta={beta:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search", choices=["short", "long", "both"], default="both")
    ap.add_argument("--short-hist", default=SHORT_HIST)
    ap.add_argument("--long-hist", default=LONG_HIST)
    ap.add_argument("--short-out", default=SHORT_OUT)
    ap.add_argument("--long-out", default=LONG_OUT)
    a = ap.parse_args()
    if a.search in ("short", "both"):
        short_tables(a.short_hist, a.short_out)
    if a.search in ("long", "both"):
        long_tables(a.long_hist, a.long_out)


if __name__ == "__main__":
    main()
