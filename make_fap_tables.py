#!/usr/bin/env python
"""Significance -> FAP lookup tables for the ZVAR short- and long-period searches.
Both model the false alarms as a Gumbel (GEV, c = 0) per (band, n_bins); FAP is its
survival function. Inputs are best peak histograms.
"""
import os

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
from scipy.stats import genextreme, norm

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = [(b, c) for b in "gr" for c in ("20bin", "10bin", "5bin")]
NBINS = {"20bin": 20, "10bin": 10, "5bin": 5}


def load_hist(path):
    """(band, config) -> (bin centres, density, n_total)."""
    parse = lambda s: np.fromstring(s.strip().strip("[]"), sep=" ")
    return {(r.band, r.config): (parse(r.bin_centers), parse(r.density), int(r.n_total))
            for r in pd.read_csv(path).itertuples()}


def write_table(out_dir, band, cfg, grid, fap, floor):
    nsig = norm.isf(np.clip(fap, floor, 1.0 - 1e-15))       # one-sided Gaussian equivalent
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({"significance": grid, "FAP": fap, "N_sigma": nsig}).to_csv(
        os.path.join(out_dir, f"fap_{band}band_{NBINS[cfg]:02d}bin.csv"), index=False)


# ── Short period (5 min - 10 d): least-squares Gumbel fit to the noise hump ──────
PEAK_FRAC = 2e-3     # fit the contiguous bins with density >= PEAK_FRAC * peak


def fit_short(bc, d, n):
    w, pk = float(bc[1] - bc[0]), int(np.argmax(d))
    lo = hi = pk
    while lo - 1 >= 0 and d[lo - 1] >= d[pk] * PEAK_FRAC:
        lo -= 1
    while hi + 1 < len(d) and d[hi + 1] >= d[pk] * PEAK_FRAC:
        hi += 1
    x, y = bc[lo:hi + 1], d[lo:hi + 1]
    sigma = np.sqrt(np.clip(y * w * n, 1.0, None)) / (w * n)   # Poisson, in density units
    wt = y / y.sum()                                            # moment guess for beta
    std = np.sqrt(np.sum(wt * (x - np.sum(x * wt)) ** 2))
    (mu, beta), _ = curve_fit(lambda z, mu, beta: genextreme.pdf(z, 0.0, loc=mu, scale=beta),
                              x, y, p0=[bc[pk], max(std, 0.3)], sigma=sigma,
                              bounds=([-np.inf, 1e-3], [np.inf, np.inf]), maxfev=40000)
    return mu, beta


def short_tables():
    grid = np.round(np.arange(0.0, 200.0 + 0.1, 0.1), 1)
    for (band, cfg), (bc, d, n) in load_hist(os.path.join(HERE, "histogram_data.csv")).items():
        mu, beta = fit_short(bc, d, n)
        sf = lambda z: np.clip(genextreme.sf(z, 0.0, loc=mu, scale=beta), 1e-300, 1.0)
        fap = np.where(grid < mu, sf(mu), sf(grid))   # below the mode, clamp to SF(mu) = 1 - 1/e
        write_table(os.path.join(HERE, "fap_lookup_tables"), band, cfg, grid, fap, 1e-300)
        print(f"short {band} {cfg:5s}  mu={mu:.4f}  beta={beta:.4f}")


# ── Long period (10-1000 d): Poisson MLE on absolute counts, fitted window ───────
# The files are truncated at significance 10 (no hump), so the model is
# n_searched * width * gumbel_pdf(z), fitted on a window chosen by tolerance `tol`,
# with `tol` itself chosen to match a window-free Gumbel + power-law fit.
MIN_LAMBDA = 25.0    # bins expecting fewer counts are skipped in relative-deviation tests
MIX_DEV_MAX = 3.0    # mixture fits worse than this (deviance/bin) are not a reference
TOL_GRID = np.array([0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 0.35, 0.50])
ZLO_GRID = np.round(np.arange(10.0, 16.01, 0.5), 3)
ZHI_GRID = np.round(np.arange(14.0, 60.01, 0.5), 3)


def gumbel_pdf(z, mu, beta):
    t = (np.asarray(z, dtype=float) - mu) / beta
    return np.exp(-t - np.exp(-t)) / beta


def fit_long(h, z_lo, z_hi, p0=(0.0, 4.3)):
    """Poisson MLE for (mu, beta) on [z_lo, z_hi]; beta fitted in log."""
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
    """(z_lo, z_hi) -> (mu, beta) on the window grid, warm-started along z_hi."""
    out = {}
    for z_lo in ZLO_GRID:
        p0 = (0.0, 4.3)
        for z_hi in ZHI_GRID[ZHI_GRID >= z_lo + 2.0]:
            mu, beta = fit_long(h, z_lo, z_hi, p0)
            if np.isfinite(beta):
                out[(z_lo, z_hi)] = p0 = (mu, beta)
    return out


def fitted_window(h, scan, tol, n_iter=8):
    """Iterate the edges to a fixed point, then refit: upper = first bin > tol above the
    model; lower = smallest edge whose lower half the model matches within tol."""
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
    """Window-free Gumbel + power-law excess over the whole recorded range.
    Returns (beta, Poisson deviance per bin)."""
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


def long_tables():
    hist = {}
    for key, (c, d, n) in load_hist(os.path.join(HERE, "data", "zvar_long_sig_hist.csv")).items():
        w = float(c[1] - c[0])
        hist[key] = (c, np.round(d * w * n).astype(np.int64), n, w)   # density -> counts

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
        write_table(os.path.join(HERE, "fap_lookup_tables_long_fitwin"),
                    band, cfg, grid, fap, tiny)
        print(f"long  {band} {cfg:5s}  mu={mu:.4f}  beta={beta:.4f}")


if __name__ == "__main__":
    short_tables()
    long_tables()
