# ZVAR Legacy Build Documentation — Version 1

This document describes the legacy-branch build of the ZVAR data runs ("ZTF Variable Source Project"). It explains the pipeline architecture, primary data products, HDF5 layout, period-finding strategy, operational considerations, and caveats for users of the legacy data build. This README is documentation only (not an installation guide) and is intentionally labeled "Version 1".

## Purpose and scope

This file documents how the legacy branch of the ZVAR pipeline (the codebase used to produce the Phase 1 data release described in the accompanying paper) runs, what data products it produces, and how to interpret those products. It is aimed at collaborators and archival users who need to understand the legacy processing choices, file formats, and known limitations.

Notes:
- The Appendix ("Pipeline and Methodology") of the ZVAR paper (main(3).tex) is the primary reference used to create this document. Where the Appendix gives field- or implementation-specific details those are summarized here.
- This README documents the legacy pipeline implementation and expected outputs; the repository also contains a separate installation/usage README.md which explains how to build/run the code locally. Do not replace that file with this document.

## High-level pipeline overview

The legacy ZVAR pipeline comprises two major stages:

1. Forced photometry extraction (image-difference forced photometry and aperture-based extraction on reference images).  
2. Periodicity detection using the Fast Periodic Weighting (FPW) periodogram implemented on NVIDIA GPUs with CUDA for parallel performance.

The pipeline was written for single-machine GPU runs — a GPU handles processing one forced-photometry matchfile at a time — and is orchestrated by lightweight Python scripts that delegate work, queue fields, and manage outputs.

Key design goals of the legacy build:
- Waveform-agnostic period search (detect sinusoids, eclipses, sharp features).  
- GPU-accelerated periodograms to handle very large frequency grids (Nfreq ~ 1e6) and many light curves per file (Nsrc ~ 1e3–1e5).  
- Keep forced-photometry and period-finding outputs as compact, queryable HDF5 datasets for downstream analysis.

## Input catalogs, selection, and forced photometry

- The pipeline seeds forced photometry positions from the Pan-STARRS1 (PS1) catalog for objects within the ZTF footprint. Only PS1 sources brighter than the legacy magnitude cut (m ≈ 21.5) are included for forced photometry.
- Forced photometry is computed on ZTF template-subtracted difference images taken from the IRSA/IPAC repository (ZTF pipeline products) and, in some cases, on reference images for aperture photometry.
- Photometry is performed per-field / per-CCD / per-quadrant (ZTF camera: 16 CCDs × 4 quadrants → 64 quadrants per field). Each quadrant produces one HDF5 "matchfile" containing exposures, source metadata, and per-epoch measured difference flux values.
- Basic per-image quality cuts are applied before a difference-image epoch is used. The legacy selection values live in the code and include checks on INFOBITS, seeing, exposure time, number of PS1 calibrators, zeropoint range, and zeropoint RMS.

## Forced-photometry HDF5 format (matchfile)

Each forced-photometry matchfile contains a group `/data` and three tables:
- `/data/exposures` — one row per epoch; columns (non-exhaustive):
  - jd, bjd (barycentric-corrected per quadrant), filterid, exptime, pid (program id), field, ccd, quad, imstat, infobits, seeing, mzpsci, mzpsciunc, maglim, airmass, nps1matches, ...
- `/data/sources` — one row per source; columns:
  - ps1_id, ra, decl, mag_ref (DoPHOT ref-mag), mag_err_ref, objtype, ...
- `/data/sourcedata` — one row per (source, epoch) pair, stored such that each light curve occupies a contiguous block on disk. Columns:
  - flux (difference flux), flux_err, flag

Indexing convention: for Nsrc sources and Nexp exposures, the entry for source j at exposure i appears at row (j * Nexp + i) of the `sourcedata` table. This contiguous layout is chosen to maximize I/O throughput when streaming light curves.

## Period-finding stage (FPW)

- Two preset frequency-search modes are executed in the legacy build:
  - Short-period search: P_min = 5 minutes, P_max = 10 days, oversample n_over = 3. Baseline removal (detrending) is applied to suppress long-timescale trends.
  - Long-period search: P_min = 10 days, P_max = ~half the light-curve baseline (typical T ~ 4 years), oversample n_over = 300. Baseline removal is NOT applied for long-period searches; instead, a sigma-clipped two-pass mean/std estimation is used to handle periodogram trends.
- FPW (Fast Periodic Weighting) computes a binned-phase statistic equivalent to fitting piecewise-constant boxes (M bins) to the folded light curve and evaluating a chi-squared-like score per test frequency. The legacy pipeline computes FPW for multiple binning configurations (e.g., Nbin = 5, 10, 20) in order to be sensitive both to sinusoidal and sharp-profile signals (eclipses).
- FPW complexity scales as O(Nfreq × Nexp) and is implemented with CUDA kernels to keep memory and arithmetic efficient on GPU hardware.

Batching and memory management:
- Because Nfreq and Nsrc are large, the implementation batches across frequencies and sources such that GPU memory utilization does not exceed ~90% of device memory. The code pre-allocates GPU buffers and reuses them between batches.
- The code computes phase indices (frequency × timestamp → phase bin) once per frequency grid (timestamps are shared across many sources in a quadrant) and re-uses those indices when computing FPW sums.

Peak finding and outputs:
- The FPW implementation runs a GPU-friendly peak finder on chunked periodogram data. The per-source peak-finder retains up to Npeak (legacy: 32) peaks per Nbin configuration and sorts them by significance.
- The legacy pipeline applies an in-chunk distance criterion (d = 300 indices by default) when choosing independent peaks to avoid saving close aliases of the same signal.
- A minimum significance threshold (legacy chosen z >= 10 for initial peak selection) is applied in peak finding. Significances are computed as z(f) = (χ^2(f) − μ)/σ where μ and σ are the periodogram mean and standard deviation (sigma-clipped for long-period mode).

## Periodogram HDF5 format (period-finding outputs)

The periodogram (peaks) HDF5 files created by the legacy run include one entry per light curve with the following primary fields (flattened arrays for multi-dimensional fields):
- psids — Pan-STARRS source identifier
- ra, dec — source coordinates (degrees)
- valid — fraction of exposures retained after outlier filtering (0–1)
- bestFreqs — top detected frequencies (day^-1) (for each saved peak and binning configuration)
- significance — the periodogram significance z for each saved peak
- means — mean periodogram power used in significance calculation
- stds — periodogram standard deviation used in significance calculation

Flattening convention: For a logical index (i, j, k) for source i, binning j, and peak rank k, the flattened index ordering is `i * Nsamp * Npeak + j * Npeak + k` (Nsamp = number of Nbin ensambles). Missing peaks are stored with a frequency of -1 and significance -inf.

## Calibration of significances and false alarm probabilities (FAP)

- The legacy build computes per-field, per-band, per-binning GEV/Gumbel fits to the empirical distribution of maximum significances in the noise-dominated regime. These fits produce field-specific conversion from the measured z to a false-alarm probability (FAP) and to an equivalent Gaussian N_sigma value.
- Table outputs in the paper list example μ and β parameters for each band and Nbin. Users should treat these FAPs as internal to the legacy build: they are not directly comparable to FAPs published by other surveys or produced by other algorithms.
- Caveats: ZTF sampling is highly non-uniform and astrophysical sources are rarely pure white noise. Sampling aliases (sidereal harmonics, lunar cycle, seasonal effects) may cause spurious peaks even when FAP is low. Users should validate candidate periodicities, especially at low significance and near excised frequency ranges.

## Frequency excision (known systematic harmonics)

To reduce the number of spurious peaks due to observing cadence and known time scales, the legacy build excises a set of frequency ranges before running FPW. These include harmonics associated with:
- Sidereal day and rational fractions (1 day, 1/2 day, 1/3 day, ...)
- Lunar month
- Annual and multi-year harmonics (1 year, 1/2 year, ...)

The exact central frequencies and half-widths used in the legacy cuts are encoded in the code and summarized in the paper (see the excision table). Users should consult the code and the paper if they need to reproduce or alter these excision windows.

## Preprocessing and outlier handling

- Each light curve is median-subtracted and sigma-clipped using MAD-derived σ (σ ≈ 1.4826 × MAD). An asymmetric sigma-clipping step is used: values > +5σ or < −10σ are flagged (upper clip more conservative to avoid removing flares; lower clip less strict to avoid removing eclipses).
- For short-period searches a Chebyshev polynomial baseline (default degree = 7) is fit with GPU-accelerated least-squares (Gpufit). Fits use tolerance 1e-4 and up to 1000 iterations. If baseline removal fails for a small fraction of sources (< ~1% legacy), the source is flagged and the periodogram is computed without baseline removal.

## Operational recommendations and hardware

- Legacy runs were executed on single nodes with a modern NVIDIA GPU (compute capability and available memory determine batching sizes). The pipeline assumes CUDA support and was tuned for GPUs with multiple GBs of device memory; allocating ~90% of available device memory is the default for batch sizing.
- Expect that very large fields (many PS1 sources) will require multiple days of GPU runtime if run serially; pipeline orchestration should be used to distribute fields across worker nodes if available.
- Memory/performance tuning knobs are present in the legacy scripts (batch sizes, shared-memory allocation fractions, Npeak, d distance for alias masking, frequency grid oversampling). Check the code defaults before modifying for production runs.

## Known limitations and caveats (legacy build)

- Heterogeneous sampling: ZTF has strong field-to-field and declination-to-declination sampling variation; detection sensitivity varies accordingly. Low-declination fields often have fewer epochs.
- False positives at survey limits: objects near the photometric limit (m ≈ 21.5) show inflated false positive rates due to rapidly increasing photometric errors; treat dim-object candidates with extra scrutiny.
- Harmonic aliases and window-function artifacts remain a significant source of false detections despite excision windows. Multi-peak checks and cross-validation across bands/seasons are recommended.
- Not all astrophysical variable types are well-detected by single-mode periodograms: multimode pulsators and stochastic variables (e.g., dwarf novae) are often missed or detected at lower significance.
- Timestamp uncertainties: shutter open/close timing and a legacy pipeline timing offset (reported in the paper) introduce small timing systematics; shortest tested periods (5 minutes) can be affected at ~1% phase level by shutter timing.

## Querying and data access (legacy outputs)

- Legacy pipeline outputs are HDF5 and intended to be ingested into query systems. The ZVAR project published an API (see paper text): endpoints for photometry, objects, cone searches, crossmatches, etc. If you are using legacy HDF5 files directly, use the stated table layouts in this README to extract light curves and periodogram peaks.

Example (pseudocode) to read a forced photometry light curve for a source j:

```python
# pseudocode (the repository contains code utilities for reading these files)
h5 = h5py.File('matchfile.h5','r')
Nexp = len(h5['/data/exposures'])
# row offset for source j:
offset = j * Nexp
flux = h5['/data/sourcedata']['flux'][offset:offset+Nexp]
flux_err = h5['/data/sourcedata']['flux_err'][offset:offset+Nexp]
exposure_meta = h5['/data/exposures'][:]  # timestamps, seeing, etc.
```

## Reproducing legacy results

- Reproducing the Phase 1 legacy results requires the following:
  - The same forced-photometry matchfile generation parameters (PS1 seed catalog, aperture/DoPHOT settings, quality cut thresholds)
  - The same FPW frequency-grids (short and long configurations, oversampling factors), Nbin configurations, and excision lists
  - The same peak-finding parameters (Npeak, d alias distance, z threshold)
  - The same per-field GEV fitting procedure to map z → FAP

- All of the above parameters are available in the legacy branch code. Consult configuration scripts and the Appendix for exact numeric defaults; the defaults used to produce the published results are encoded in the pipeline control scripts.

## Contact and provenance

- Primary contacts: authors listed in main(3).tex (e.g., Samuel Whitebook, Thomas Prince) and the ZVAR collaboration. The paper's corresponding author is listed as Thomas Prince (prince@caltech.edu) in the manuscript header.
- Provenance: every HDF5 matchfile and period-file contains header metadata fields (timestamps, field/ccd/quad, program id) useful for tracing the specific exposures and runs.

## Appendix references

This README summarizes the Appendix "Pipeline and Methodology" (see main(3).tex in the repo root) and the main body where photometric limits, periodogram calibration, and data availability are discussed. Refer to those sections for the mathematically precise definitions of FPW, the false alarm probability conversion, and the forced-photometry I/O formats.

-- End of Version 1 --
