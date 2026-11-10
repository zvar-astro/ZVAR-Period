# ZVAR V1 Build Documentation - Version 1

This document describes the V1-branch build of the ZVAR data runs ("ZTF Variable Source Project"). It explains the pipeline architecture, primary data products, HDF5 layout, period-finding strategy, operational considerations, and caveats for users of the V1 data build. This code was used to run the short period periodicity analysis on the image difference data as released in ZVAR DR1 (Whitebook et al 2026c). This README is documentation only (not an installation guide).

## Purpose and scope

This file documents how the V1 branch of the ZVAR pipeline (the codebase used to produce the short period data of ZVAR DR1 described in the accompanying paper) runs, what data products it produces, and how to interpret those products. It is aimed at collaborators and archival users who need to understand the V1 processing choices, file format, and code specifics.

Notes:
- Much of the code as specified here is also described in a linear fashion in the Appendix of the ZVAR DR1 paper (Whitebook et al 2026c). Where the Appendix gives field- or implementation-specific details those are summarized here.
- This README documents the V1 pipeline implementation and expected outputs; the repository also contains a separate installation/usage README.md which explains how to build/run the code locally.

## High-level pipeline overview

Periodicity detection was done using the Fast Periodic Weighting (FPW) periodogram implemented on NVIDIA GPUs with CUDA for parallel performance.

The pipeline was written for single-machine GPU runs - a GPU handles processing one forced-photometry matchfile at a time - and is orchestrated by lightweight Python scripts that delegate work, queue fields, and manage outputs.

Key design goals of the V1 build:
- Waveform-agnostic period search (detect sinusoids, eclipses, sharp features).  
- GPU-accelerated periodograms to handle very large frequency grids (Nfreq ~ 1e6) and many light curves per file (Nsrc ~ 1e3-1e5).  
- Keep period-finding outputs as compact, queryable HDF5 datasets for downstream analysis.

## Period-finding stage (FPW)
- This version of the code executes the short-period search: P_min = 5 minutes, P_max = 10 days, oversample n_over = 3. Baseline removal (detrending) is applied to suppress long-timescale trends.
- FPW (Fast Periodic Weighting) computes a binned-phase statistic equivalent to fitting piecewise-constant boxes (M bins) to the folded light curve and evaluating a chi-squared-like score per test frequency. The V1 pipeline computes FPW for multiple binning configurations (e.g., Nbin = 5, 10, 20) in order to be sensitive both to sinusoidal and sharp-profile signals (eclipses).
- FPW complexity scales as O(Nfreq x Nexp) and is implemented with CUDA kernels to keep memory and arithmetic efficient on GPU hardware.

Batching and memory management:
- Because Nfreq and Nsrc are large, the implementation batches across frequencies and sources such that GPU memory utilization does not exceed ~90% of device memory. The code pre-allocates GPU buffers and reuses them between batches.
- The code computes phase indices (frequency x timestamp -> phase bin) once per frequency grid (timestamps are shared across many sources in a quadrant) and re-uses those indices when computing FPW sums.

Peak finding and outputs:
- The FPW implementation runs a GPU-friendly peak finder on chunked periodogram data. The per-source peak-finder retains up to Npeak peaks per Nbin configuration and sorts them by significance (V1 short-period run: Npeak = 50, i.e. 150 peaks per source across the three Nbin configurations; long-period run: Npeak = 32).
- The V1 pipeline applies an in-chunk distance criterion (d = 300 indices by default) when choosing independent peaks to avoid saving close aliases of the same signal.
- A minimum significance threshold (V1 chosen z >= 10 for initial peak selection) is applied in peak finding. Significances are computed as z(f) = (chi^2(f) - mu)/sigma where mu and sigma are the periodogram mean and standard deviation (sigma-clipped for long-period mode).

## Periodogram HDF5 format (period-finding outputs)

The periodogram (peaks) HDF5 files created by the V1 run include one entry per light curve with the following primary fields (flattened arrays for multi-dimensional fields):
- psids - Pan-STARRS source identifier
- ra, dec - source coordinates (degrees)
- valid - fraction of exposures retained after outlier filtering (0-1)
- bestFreqs - top detected frequencies (day^-1) (for each saved peak and binning configuration)
- significance - the periodogram significance z for each saved peak
- means - mean periodogram power used in significance calculation
- stds - periodogram standard deviation used in significance calculation

Flattening convention: For a logical index (i, j, k) for source i, binning j, and peak rank k, the flattened index ordering is `i * Nsamp * Npeak + j * Npeak + k` (Nsamp = number of Nbin ensambles). Missing peaks are stored with a frequency of -1 and significance -inf.

## Frequency excision (known systematic harmonics)

To reduce the number of spurious peaks due to observing cadence and known time scales, the V1 build excises a set of frequency ranges before running FPW. These include harmonics associated with:
- Sidereal day and rational fractions (1 day, 1/2 day, 1/3 day, ...)
- Lunar month
- Annual and multi-year harmonics (1 year, 1/2 year, ...)

The exact central frequencies and half-widths used in the V1 cuts are encoded in the code and summarized in the paper (see the excision table). Users should consult the code and the paper if they need to reproduce or alter these excision windows.

## Preprocessing and outlier handling

- Each light curve is median-subtracted and sigma-clipped using MAD-derived sigma (sigma ~ 1.4826 x MAD). An asymmetric sigma-clipping step is used: values > +5 sigma or < -10 sigma are flagged (upper clip more conservative to avoid removing flares; lower clip less strict to avoid removing eclipses).
- For short-period searches a Chebyshev polynomial baseline (default degree = 7) is fit with GPU-accelerated least-squares (Gpufit). Fits use tolerance 1e-4 and up to 1000 iterations. If baseline removal fails for a small fraction of sources (< ~1% V1), the source is flagged and the periodogram is computed without baseline removal.

## Alias removal (alias_removal.py)

`alias_removal.py` builds per-quadrant frequency blacklists from the periodogram HDF5 files. It reads all 150 stored peaks per source of the short-period run, each of which lies on its quadrant's frequency grid; a grid frequency shared by more distinct sources than the local background allows is treated as an alias and blacklisted. Each (field, CCD, quadrant, band) is processed independently:
- The number of distinct sources per key is counted. Runs of >= 20 empty keys were excised by the pipeline and are not counted as searched.
- Keys are split into log-frequency windows of at least 2000 searched keys. In each window a negative binomial NB(mu, r) is fit by truncated maximum likelihood to the keys with count < T, where T = max(median + 1, Poisson 3-sigma threshold at the clipped mean - 1, 3).
- log mu and log r are interpolated in log f, and every key whose count reaches the NB one-sided 3-sigma quantile is blacklisted. Windows whose fit diverges take the threshold interpolated from their neighbours.

Outputs (under `--out`):
- `<field>/f<field>_c<ccd>_q<quad>_<band>.csv` - blacklisted frequencies with columns freq, n_dup (distinct sources), n_cut (threshold)
- `<field>/nb_profile.csv` - the per-window NB fits
- `manifest.csv` - one row per quadrant-band with status and summary statistics. Quadrants with fewer than 1000 peaks, a sparse or incoherent grid, or no converged NB fit are recorded here and skipped.

Fields are processed in turn with a multiprocessing pool (`--jobs`, default 24); `--field` and `--band` restrict the run, and completed field-bands in the manifest are skipped unless `--overwrite` is given. The input root (`/data/zvar/zvar_results`) is set in the script.

## Significance to FAP calibration (fap_calibration.py)

`fap_calibration.py` builds the lookup tables that map periodogram significance z to false-alarm probability (FAP). A source's significance is the maximum peak over many trial frequencies, so for pure white noise it follows a Gumbel distribution. One (mu, beta) pair is fit per band and Nbin configuration, and FAP is the Gumbel survival function.
- Short-period search (5 min - 10 days): weighted least-squares Gumbel fit to the noise hump of the best-peak density histogram, over the contiguous bins with density >= 2e-3 of the peak.
- Long-period search (10 - 1000 days): the input histograms stop at z = 10, so there is no hump. The fit is a Poisson maximum-likelihood fit to absolute counts over an iteratively fitted window; the window tolerance is the one whose beta best matches a window-free Gumbel + power-law reference fit.

Outputs are `fap_{g,r}band_{05,10,20}bin.csv` with columns significance, FAP, and N_sigma (the one-sided Gaussian equivalent), on a 0.1 significance grid over 0-200 (short) or 0-1000 (long). `--search short|long|both` selects which tables are built (default both).

## Operational recommendations and hardware

- V1 runs were executed on single nodes with a modern NVIDIA GPU (compute capability and available memory determine batching sizes). The pipeline assumes CUDA support and was tuned for GPUs with multiple GBs of device memory; allocating ~90% of available device memory is the default for batch sizing.
- Expect that very large fields (many PS1 sources) will require multiple days of GPU runtime if run serially; pipeline orchestration should be used to distribute fields across worker nodes if available.
- Memory/performance tuning knobs are present in the V1 scripts (batch sizes, shared-memory allocation fractions, Npeak, d distance for alias masking, frequency grid oversampling). Check the code defaults before modifying for production runs.

## Reproducing V1 results

- Reproducing the V1 results requires the following:
  - The same forced-photometry matchfile generation parameters (PS1 seed catalog, aperture/DoPHOT settings, quality cut thresholds)
  - The same FPW frequency-grids (short and long configurations, oversampling factors), Nbin configurations, and excision lists
  - The same peak-finding parameters (Npeak, d alias distance, z threshold)
  - The same alias-removal blacklists (`alias_removal.py`: NB null model, 3-sigma threshold, window sizes)
  - The same Gumbel (GEV) fit per band and Nbin to map z -> FAP (`fap_calibration.py`)

- All of the above parameters are available in the V1 branch code. Consult configuration scripts for exact numeric defaults; the defaults used to produce the published results are encoded in the pipeline control scripts.

## Contact

- Primary contacts: If you have an issue please submit an issue at https://github.com/zvar-astro/zvar-help/issues or contact Samuel Whitebook (sewhitebook@astro.caltech.edu) or Thomas Culino (tculino@caltech.edu) with questions.
