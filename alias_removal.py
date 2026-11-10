#!/usr/bin/env python
"""Alias-removal blacklists for the ZVAR FPW period search.

Every source's 150 stored peaks sit on its quadrant's search grid. A grid frequency
shared by more distinct sources than the local background allows is an alias, so it
is blacklisted. Each (field, ccd, quadrant, band) gets this procedure:

  1. Fit the grid step and phase, and map every peak to an integer grid key.
  2. Count the distinct sources per key. Runs of >= 20 empty keys were blanked by the
     pipeline and are not counted as searched.
  3. Split the keys into log-frequency windows of at least 2000 searched keys. In each
     window, fit a negative binomial NB(mu, r) by truncated maximum likelihood to the
     keys with count < T. Here T = max(median + 1, Poisson 3-sigma threshold at the
     clipped mean - 1, 3), so the alias tail cannot widen its own null.
  4. Interpolate log mu and log r in log f. Blacklist every key whose count reaches
     the NB 3-sigma quantile, P(M >= n_cut) < Phi(-3).

A fit that diverges (mu runs to the parameter bound) takes its neighbours' threshold.

Output, under --out:
    <field>/f<field>_c<ccd>_q<quad>_<band>.csv   freq, n_dup, n_cut   (blacklisted keys)
    <field>/nb_profile.csv                       the per-window NB fits
    manifest.csv                                 one row per quadrant-band

Usage:
    python alias_removal.py --field 281 --band zg     # one field, one band
    python alias_removal.py --field 281               # one field, both bands
    python alias_removal.py                           # every field, in turn
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one BLAS thread per worker

import argparse
import glob
import time
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import nbinom, norm, poisson

DATA_ROOT = "/data/zvar/zvar_results"
OUT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "cut_lookup_tables_v3m_nb")
KEYS = ["field", "ccd", "quad", "band"]

N_PEAKS = 150             # stored peaks per source
ALPHA = norm.sf(3.0)      # 3-sigma one-sided tail
MIN_PEAKS = 1000          # fewer peaks than this: no table
GRID_FIT_HI = (5.0, 20.0) # fit the grid on f < 5 c/d, widened to 20 if too few
GRID_FIT_MIN = 200        # distinct frequencies needed to fit the grid
GRID_R_MIN = 0.8          # minimum grid coherence
BLANK_RUN = 20            # empty runs at least this long were blanked by the pipeline
N_WINDOWS = 24            # log-spaced windows before merging
WINDOW_MIN_KEYS = 2000    # merge windows until each has this many searched keys
CLIP_NSIG, CLIP_ITERS = 3.0, 3
R_MAX = 1e4               # NB r cap (Poisson limit)
D_DIVERGED = 1e6          # var/mean = 1 + mu/r above this: the fit diverged
N_FINE = 3000             # log-f grid the threshold is evaluated on


# ---------------------------------------------------------------- the search grid
def measure_grid_step(freq, span=3e-3, npts=4001, iters=4, coarse_n=2000):
    """Grid step maximising the Rayleigh coherence |<exp(2 pi i f / step)>|.

    The low-percentile gap is the starting guess. The scan then zooms 80x per pass
    on a subsample. Returns (step, R, phase).
    """
    x = np.asarray(freq, float)
    xc = x if len(x) <= coarse_n else np.random.default_rng(0).choice(x, coarse_n,
                                                                       replace=False)
    d = np.diff(np.unique(x))
    step = float(np.percentile(d[d > 0], 2.0))
    for _ in range(iters):
        grid = np.linspace(step * (1 - span), step * (1 + span), npts)
        coh = np.empty(npts)
        for i in range(0, npts, 256):                       # chunked to bound memory
            g = grid[i:i + 256]
            coh[i:i + 256] = np.abs(np.exp(1j * (2 * np.pi * (xc[None, :] / g[:, None])))
                                    .mean(1))
        step = float(grid[coh.argmax()])
        span /= 80.0
    z = np.mean(np.exp(2j * np.pi * (x / step)))
    return step, float(np.abs(z)), float((np.angle(z) / (2 * np.pi)) % 1.0)


def multiplicity(key, psid):
    """Distinct sources per key. Returns (unique keys, counts, inverse index)."""
    order = np.lexsort((psid, key))
    ks, ps = key[order], psid[order]
    new_key = np.r_[True, ks[1:] != ks[:-1]]
    new_pair = new_key | np.r_[True, ps[1:] != ps[:-1]]
    uk = ks[new_key]
    cnt = np.add.reduceat(new_pair.astype(np.int64), np.flatnonzero(new_key))
    return uk, cnt, np.searchsorted(uk, key)


def searched_mask(uk):
    """Dense mask over [uk.min(), uk.max()]: False on empty runs >= BLANK_RUN."""
    occupied = np.zeros(uk.max() - uk.min() + 1, bool)
    occupied[uk - uk.min()] = True
    searched = np.ones_like(occupied)
    idx = np.flatnonzero(~occupied)
    if idx.size:
        brk = np.flatnonzero(np.diff(idx) > 1)
        for a, b in zip(np.r_[idx[0], idx[brk + 1]], np.r_[idx[brk], idx[-1]]):
            if b - a + 1 >= BLANK_RUN:
                searched[a:b + 1] = False
    return searched


def windows(searched, k_lo, step, phase):
    """Log-spaced key windows, merged left to right to >= WINDOW_MIN_KEYS searched keys."""
    k_hi = k_lo + searched.size - 1
    f_lo, f_hi = max((k_lo + phase) * step, 1e-6), (k_hi + phase) * step
    edges = np.unique(np.clip(np.rint(np.geomspace(f_lo, f_hi, N_WINDOWS + 1) / step
                                      - phase).astype(np.int64), k_lo, k_hi))
    csum = np.r_[0, np.cumsum(searched)]
    ns = lambda a, b: int(csum[b - k_lo + 1] - csum[a - k_lo])
    out, a = [], edges[0]
    for b in edges[1:]:
        if ns(a, b) >= WINDOW_MIN_KEYS or b == edges[-1]:
            out.append((a, b))
            a = b
    if len(out) > 1 and ns(*out[-1]) < WINDOW_MIN_KEYS:     # fold a thin last window back
        out[-2] = (out[-2][0], out[-1][1])
        out.pop()
    return [(a, b, ns(a, b)) for a, b in out if ns(a, b) > 0]


# ------------------------------------------------------------------ the null model
def poisson_threshold(lam):
    return int(poisson.isf(ALPHA, lam)) + 1


def nb_threshold(mu, r):
    """Smallest count with NB upper tail below ALPHA (vectorised)."""
    mu, r = np.asarray(mu, float), np.asarray(r, float)
    return (nbinom.isf(ALPHA, r, r / (r + np.maximum(mu, 1e-12))) + 1).astype(np.int64)


def clipped_mean(counts, n_searched):
    """Sigma-clipped mean count over searched keys. Empty keys enter as zeros."""
    n_zero = max(int(n_searched) - counts.size, 0)
    if counts.size + n_zero == 0:
        return 0.0
    lam = float(counts.sum()) / (counts.size + n_zero)
    for _ in range(CLIP_ITERS):
        keep = counts <= lam + CLIP_NSIG * np.sqrt(max(lam, 1.0))
        new = float(counts[keep].sum()) / max(int(keep.sum()) + n_zero, 1)
        if not np.isfinite(new) or new <= 0:
            break
        done = abs(new - lam) < 1e-4 * max(lam, 1e-6)
        lam = new
        if done:
            break
    return lam


def fit_nb_truncated(hist, T):
    """Truncated ML fit of NB(mu, r) to hist[:T]. Returns (mu, r); NaN if it fails."""
    T = int(min(T, hist.size))
    w, m = hist[:T].astype(float), np.arange(T)
    if w.sum() == 0:
        return np.nan, np.nan
    mean = (w * m).sum() / w.sum()
    if mean <= 0:
        return 1e-6, R_MAX                                  # nothing below T but zeros

    def nll(x):
        x = np.clip(x, -30.0, 30.0)
        mu, r = np.exp(x[0]), min(np.exp(x[1]), R_MAX)
        p = r / (r + mu)
        norm_ = max(nbinom.cdf(T - 1, r, p), 1e-300)
        return -(w * nbinom.logpmf(m, r, p)).sum() + w.sum() * np.log(norm_)

    best = min((minimize(nll, [np.log(mean), np.log(r0)], method="Nelder-Mead",
                         options=dict(xatol=1e-5, fatol=1e-7, maxiter=3000))
                for r0 in (1.0, 10.0, 1000.0)), key=lambda res: res.fun)
    x = np.clip(best.x, -30.0, 30.0)
    mu, r = np.exp(x[0]), min(np.exp(x[1]), R_MAX)
    if 1.0 + mu / r > D_DIVERGED:                          # mu ran to the bound
        return np.nan, np.nan
    return float(mu), float(r)


def nb_profile(uk, cnt, searched, k_lo, step, phase):
    """One NB fit per window. Failed windows record the threshold interpolated there."""
    rows = []
    for node, (a, b, n_s) in enumerate(windows(searched, k_lo, step, phase)):
        c = cnt[(uk >= a) & (uk <= b)]
        hist = np.bincount(c, minlength=1)
        hist[0] += max(n_s - c.size, 0)                     # empty searched keys
        lam_clip = clipped_mean(c, n_s)
        n_pois = poisson_threshold(lam_clip)
        q50 = int(np.searchsorted(np.cumsum(hist) / hist.sum(), 0.5))
        T = max(q50 + 1, n_pois - 1, 3)
        mu, r = fit_nb_truncated(hist, T)
        rows.append(dict(node=node, freq=np.sqrt(max((a + phase) * step, 1e-9)
                                                 * (b + phase) * step),
                         n_keys=n_s, lam_clip=lam_clip, n_cut_pois=n_pois, T=T, mu=mu,
                         r=r, n_cut=int(nb_threshold(mu, r)) if np.isfinite(mu) else -1,
                         converged=bool(np.isfinite(mu))))
    prof = pd.DataFrame(rows)
    bad = ~prof.converged.to_numpy()
    if bad.any() and not bad.all():
        prof.loc[bad, "n_cut"] = threshold_at(prof, np.log(prof.freq.to_numpy()[bad]))
    return prof


def threshold_at(prof, lf):
    """NB threshold at log-frequencies lf, from log mu and log r interpolated in log f."""
    good = prof[prof.converged]
    x = np.log(good.freq.to_numpy())
    mu = np.exp(np.interp(lf, x, np.log(np.maximum(good.mu.to_numpy(), 1e-12))))
    return nb_threshold(mu, np.exp(np.interp(lf, x, np.log(good.r.to_numpy()))))


def per_key_threshold(prof, f_key):
    """Threshold for every key, evaluated on an N_FINE log-f grid (nearest point)."""
    grid = np.linspace(np.log(f_key.min()), np.log(f_key.max()), N_FINE)
    idx = np.clip(np.rint((np.log(f_key) - grid[0]) / (grid[1] - grid[0])), 0,
                  N_FINE - 1).astype(np.int64)
    return threshold_at(prof, grid)[idx]


# ------------------------------------------------------------- one quadrant-band
def process(args):
    """Build one quadrant-band's blacklist. Returns (manifest row, nb profile or None)."""
    path, out_dir = args
    field, ccd, quad, band = os.path.basename(path)[4:-3].split("_")
    rec = dict(field=int(field), ccd=int(ccd), quad=int(quad), band=band, status="ok")
    try:
        with h5py.File(path, "r") as h:
            psids = h["psids"][:]
            freq = h["bestFreqs"][:].reshape(len(psids), N_PEAKS)
            sig = h["significance"][:].reshape(len(psids), N_PEAKS)
    except Exception as exc:
        rec["status"] = f"unreadable:{type(exc).__name__}"
        return rec, None
    row, col = np.nonzero((sig > 0) & (freq > 0) & np.isfinite(freq) & np.isfinite(sig))
    fr, ps = freq[row, col].astype(np.float64), psids[row].astype(np.uint64)
    rec.update(n_src=int(np.unique(ps).size) if fr.size else 0, n_peaks=int(fr.size))
    if fr.size < MIN_PEAKS:
        rec["status"] = "too_few_peaks"
        return rec, None

    for f_hi in GRID_FIT_HI:
        lowf = np.unique(fr[(fr > 0.1) & (fr < f_hi)])
        if len(lowf) >= GRID_FIT_MIN:
            break
    else:
        rec["status"] = "grid_too_sparse"
        return rec, None
    step, R, phase = measure_grid_step(lowf)
    rec.update(step=step, baseline_d=1.0 / step, grid_R=R, phase=phase)
    if R < GRID_R_MIN:
        rec["status"] = f"grid_incoherent_R{R:.2f}"
        return rec, None

    key = np.round(fr / step - phase).astype(np.int64)
    uk, cnt, inv = multiplicity(key, ps)
    searched = searched_mask(uk)
    lam = float(cnt.sum()) / max(int(searched.sum()), 1)
    rec.update(n_searched=int(searched.sum()), n_pairs=int(cnt.sum()), lam=lam,
               n_cut_flat=poisson_threshold(lam))

    prof = nb_profile(uk, cnt, searched, int(uk.min()), step, phase)
    if not prof.converged.any():
        rec["status"] = "nb_profile_failed"
        return rec, None
    f_key = (uk + phase) * step
    n_cut = per_key_threshold(prof, f_key)
    kill = cnt >= n_cut
    pk = kill[inv]
    rec.update(n_nodes=len(prof), n_cut_lo=int(prof.n_cut.min()),
               n_cut_hi=int(prof.n_cut.max()), n_unconverged=int((~prof.converged).sum()),
               n_freq_cut=int(kill.sum()), n_peaks_cut=int(pk.sum()),
               frac_peaks_cut=float(pk.mean()), n_src_kept=int(np.unique(ps[~pk]).size))
    rec["src_retention"] = rec["n_src_kept"] / max(rec["n_src"], 1)

    csv = os.path.join(out_dir, f"f{field}_c{ccd}_q{quad}_{band}.csv")
    pd.DataFrame({"freq": f_key[kill], "n_dup": cnt[kill], "n_cut": n_cut[kill]}
                 ).sort_values("freq").to_csv(csv + ".part", index=False,
                                              float_format="%.10g")
    os.replace(csv + ".part", csv)                          # never leave a partial table
    return rec, prof.assign(**{k: rec[k] for k in KEYS})[KEYS + list(prof.columns)]


# ---------------------------------------------------------------------- driver
def merge_csv(path, new, keys, float_format=None):
    """Write `new` to path, replacing any existing rows with the same keys."""
    if os.path.exists(path):
        old = pd.read_csv(path)
        idx = pd.MultiIndex.from_frame(new[keys].drop_duplicates())
        new = pd.concat([old[~pd.MultiIndex.from_frame(old[keys]).isin(idx)], new])
    new.sort_values(keys + [c for c in ("node",) if c in new]).to_csv(
        path, index=False, float_format=float_format)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", type=int, help="one field (default: every field)")
    ap.add_argument("--band", choices=["zg", "zr", "g", "r"], help="default: both")
    ap.add_argument("--out", default=OUT_ROOT)
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--overwrite", action="store_true",
                    help="rebuild field-bands already in the manifest")
    a = ap.parse_args()
    bands = ["z" + a.band[-1]] if a.band else ["zg", "zr"]
    fields = [a.field] if a.field is not None else sorted(
        int(os.path.basename(d)) for d in glob.glob(f"{DATA_ROOT}/[0-9][0-9][0-9][0-9]"))
    manifest = os.path.join(a.out, "manifest.csv")
    done = (set(pd.read_csv(manifest, usecols=["field", "band"]).itertuples(index=False,
                                                                            name=None))
            if os.path.exists(manifest) and not a.overwrite else set())

    with Pool(a.jobs) as pool:
        for i, field in enumerate(fields, 1):
            todo = [b for b in bands if (field, b) not in done]
            files = [p for b in todo for p in sorted(glob.glob(
                f"{DATA_ROOT}/{field:04d}/fpw_{field:04d}_*_{b}.h5"))]
            if not files:
                print(f"[{i}/{len(fields)}] {field:04d} nothing to do", flush=True)
                continue
            t0 = time.time()
            out_dir = os.path.join(a.out, f"{field:04d}")
            os.makedirs(out_dir, exist_ok=True)
            out = pool.map(process, [(p, out_dir) for p in files], chunksize=1)
            recs = pd.DataFrame([r for r, _ in out])
            profs = [p for _, p in out if p is not None]
            if profs:
                merge_csv(os.path.join(out_dir, "nb_profile.csv"), pd.concat(profs), KEYS,
                          float_format="%.8g")
            merge_csv(manifest, recs, KEYS)     # after each field, so a crash loses one
            ok = recs[recs.status == "ok"]
            print(f"[{i}/{len(fields)}] {field:04d} {'+'.join(todo)}: {len(ok)} tables, "
                  f"{len(recs) - len(ok)} skipped, {int(ok.n_freq_cut.sum()):,} keys "
                  f"blacklisted ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
