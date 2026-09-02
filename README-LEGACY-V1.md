# ZVAR V1 Build Documentation — Version 1

This document describes the V1-branch build of the ZVAR data runs ("ZTF Variable Source Project"). It explains the pipeline architecture, primary data products, HDF5 layout, period-finding strategy[...]

## Purpose and scope

This file documents how the V1 branch of the ZVAR pipeline (the codebase used to produce the short period data of ZVAR DR1 described in the accompanying paper) runs, what data products it produces,[...]

Notes:
- Much of the code as specified here is also described in a linear fashion in the Appendix of the ZVAR DR1 paper (Whitebook et al 2026c). Where the Appendix gives field- or implementation-specific[...]
- This README documents the V1 pipeline implementation and expected outputs; the repository also contains a separate installation/usage README.md which explains how to build/run the code locally.

## High-level pipeline overview

Periodicity detection was done using the Fast Periodic Weighting (FPW) periodogram implemented on NVIDIA GPUs with CUDA for parallel performance.

The pipeline was written for single-machine GPU runs — a GPU handles processing one forced-photometry matchfile at a time — and is orchestrated by lightweight Python scripts that delegate work[...]

Key design goals of the V1 build:
- Waveform-agnostic period search (detect sinusoids, eclipses, sharp features).  
- GPU-accelerated periodograms to handle very large frequency grids (Nfreq ~ 1e6) and many light curves per file (Nsrc ~ 1e3–1e5).  
- Keep period-finding outputs as compact, queryable HDF5 datasets for downstream analysis.

## Period-finding stage (FPW)
- This version of the code executes the short-period search: P_min = 5 minutes, P_max = 10 days, oversample n_over = 3. Baseline removal (detrending) is applied to suppress long-timescale trends.
- FPW (Fast Periodic Weighting) computes a binned-phase statistic equivalent to fitting piecewise-constant boxes (M bins) to the folded light curve and evaluating a chi-squared-like score per test[...]
- FPW complexity scales as O(Nfreq × Nexp) and is implemented with CUDA kernels to keep memory and arithmetic efficient on GPU hardware.

Batching and memory management:
- Because Nfreq and Nsrc are large, the implementation batches across frequencies and sources such that GPU memory utilization does not exceed ~90% of device memory. The code pre-allocates GPU buf[...]
- The code computes phase indices (frequency × timestamp → phase bin) once per frequency grid (timestamps are shared across many sources in a quadrant) and re-uses those indices when computing [...]

Peak finding and outputs:
- The FPW implementation runs a GPU-friendly peak finder on chunked periodogram data. The per-source peak-finder retains up to Npeak (V1: 32) peaks per Nbin configuration and sorts them by signifi[...]
- The V1 pipeline applies an in-chunk distance criterion (d = 300 indices by default) when choosing independent peaks to avoid saving close aliases of the same signal.
- A minimum significance threshold (V1 chosen z >= 10 for initial peak selection) is applied in peak finding. Significances are computed as z(f) = (χ^2(f) − μ)/σ where μ and σ are the perio[...]

## Periodogram HDF5 format (period-finding outputs)

The periodogram (peaks) HDF5 files created by the V1 run include one entry per light curve with the following primary fields (flattened arrays for multi-dimensional fields):
- psids — Pan-STARRS source identifier
- ra, dec — source coordinates (degrees)
- valid — fraction of exposures retained after outlier filtering (0–1)
- bestFreqs — top detected frequencies (day^-1) (for each saved peak and binning configuration)
- significance — the periodogram significance z for each saved peak
- means — mean periodogram power used in significance calculation
- stds — periodogram standard deviation used in significance calculation

Flattening convention: For a logical index (i, j, k) for source i, binning j, and peak rank k, the flattened index ordering is `i * Nsamp * Npeak + j * Npeak + k` (Nsamp = number of Nbin ensambles[...]

## Frequency excision (known systematic harmonics)

To reduce the number of spurious peaks due to observing cadence and known time scales, the V1 build excises a set of frequency ranges before running FPW. These include harmonics associated with:
- Sidereal day and rational fractions (1 day, 1/2 day, 1/3 day, ...)
- Lunar month
- Annual and multi-year harmonics (1 year, 1/2 year, ...)

The exact central frequencies and half-widths used in the V1 cuts are encoded in the code and summarized in the paper (see the excision table). Users should consult the code and the paper if they [...]

## Preprocessing and outlier handling

- Each light curve is median-subtracted and sigma-clipped using MAD-derived σ (σ ≈ 1.4826 × MAD). An asymmetric sigma-clipping step is used: values > +5σ or < −10σ are flagged (upper clip[...]
- For short-period searches a Chebyshev polynomial baseline (default degree = 7) is fit with GPU-accelerated least-squares (Gpufit). Fits use tolerance 1e-4 and up to 1000 iterations. If baseline [...]

## Operational recommendations and hardware

- V1 runs were executed on single nodes with a modern NVIDIA GPU (compute capability and available memory determine batching sizes). The pipeline assumes CUDA support and was tuned for GPUs with m[...]
- Expect that very large fields (many PS1 sources) will require multiple days of GPU runtime if run serially; pipeline orchestration should be used to distribute fields across worker nodes if avai[...]
- Memory/performance tuning knobs are present in the V1 scripts (batch sizes, shared-memory allocation fractions, Npeak, d distance for alias masking, frequency grid oversampling). Check the code [...]

## Reproducing V1 results

- Reproducing the V1 results requires the following:
  - The same forced-photometry matchfile generation parameters (PS1 seed catalog, aperture/DoPHOT settings, quality cut thresholds)
  - The same FPW frequency-grids (short and long configurations, oversampling factors), Nbin configurations, and excision lists
  - The same peak-finding parameters (Npeak, d alias distance, z threshold)
  - The same per-field GEV fitting procedure to map z -> FAP

- All of the above parameters are available in the V1 branch code. Consult configuration scripts and the Appendix of Whitebook et al 2026c for exact numeric defaults; the defaults used to produce [...]

## Contact

- Primary contacts: If you have an issue please submit an issue at https://github.com/zvar-astro/zvar-help/issues or contact Samuel Whitebook (sewhitebook@astro.caltech.edu) or Thomas Culino (tcul[...]

-- End of Version 1 --
