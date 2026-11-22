# PCA (EOF) analysis of NCEP/NCAR surface air temperature, 2013-2014

Course project for Mathematical Foundation for AI, Department of Artificial Intelligence and Data Science,
FAST-NUCES Karachi (instructor: Dr. Fahad Raiz, PhD).

Team: Jahan Zaib (MS, 26K-7626) and Kariz Kamal (PhD, 24K-9014). Started 18 November 2026.

We apply principal component analysis (called EOF analysis in climate science) to daily near-surface air
temperature over North America. The project explains eigenvalues, eigenvectors (EOFs) and principal components,
compares PCA of the full field with PCA of the anomalies, uses North's rule to see which modes can be told apart,
and checks how few components are needed to rebuild the data.

## Data
- NCEP/NCAR Reanalysis 1, four-times-daily air temperature at sigma level 0.995, 15N-75N, 160W-30W, 2.5 degree
  grid, 1 Jan 2013 to 31 Dec 2014 (25 x 53 grid points). Source: NOAA Physical Sciences Laboratory,
  <https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html>.
- `data/air_temperature.nc` is the copy that comes with the [xarray-data](https://github.com/pydata/xarray-data)
  repository.
- Daily means give a data matrix of 730 days x 1325 grid points.

## Running it
On Kaggle: Create > New Notebook > File > Import Notebook, upload `pca_ncep_temperature_kaggle.ipynb`, turn
Internet on in the settings panel and press Run All. If Internet is not available, upload `data/air_temperature.nc`
as a dataset and attach it. The notebook looks for it under `/kaggle/input`.

On your own machine, open the notebook in Jupyter (needs numpy, pandas, scipy, matplotlib, xarray and netCDF4).
Figures go to `outputs/figures/`, tables to `outputs/tables/` and every number used in the report to
`outputs/results_summary.json`.

## What is in the repository
| Path | What it is |
|---|---|
| `pca_ncep_temperature_kaggle.ipynb` | the analysis notebook (already run) |
| `data/air_temperature.nc` | NCEP/NCAR temperature sample (7.7 MB) |
| `outputs/figures`, `outputs/tables` | figures and tables used in the report |
| `report/Project_Report_PCA_NCEP.docx` | the written report |

## Method in short
1. Daily mean temperature in degC, each grid point weighted by sqrt(cos latitude).
2. Analysis A uses the full field. Analysis B uses anomalies, after removing the mean and the annual and
   semi-annual harmonics (two years are too short for a day-by-day climatology).
3. PCA by SVD of the centred, weighted data matrix: eigenvalues = s^2/(n-1), EOFs = right singular vectors,
   PCs = U*S. The result is checked against `numpy.linalg.eigh` of the covariance matrix.
4. Number of modes: scree plot, cumulative variance and North's rule, with the effective sample size taken from
   the lag-1 autocorrelation.
5. Interpretation: regression maps (degC per +1 SD of the PC), correlation with seven regional temperature series
   (5 % level, effective sample size) and reconstruction RMSE.

## Main results
| | Analysis A (full field) | Analysis B (anomalies) |
|---|---|---|
| Mode 1 variance | 80.7 % (annual cycle, R = 0.99) | 14.9 % |
| Modes 2, 3, 4 | 2.6 %, 2.4 %, 2.0 % | 9.9 %, 9.7 %, 6.2 % |
| Modes needed for 80 % | 1 | 22 |

- EOF1-4 of the anomalies are a north-central continental mode, a west-east dipole, an Alaska/NW Canada mode
  and an Arctic Canada mode.
- Modes 2 and 3 cannot be separated by North's rule (eigenvalues 906 and 887).

## References
- Kalnay et al. (1996), Bull. Amer. Meteor. Soc., 77, 437-471. NCEP/NCAR reanalysis.
- North et al. (1982), Mon. Wea. Rev., 110, 699-706. Sampling errors of EOFs.
- Hannachi, Jolliffe and Stephenson (2007), Int. J. Climatol., 27, 1119-1152. EOF review.
