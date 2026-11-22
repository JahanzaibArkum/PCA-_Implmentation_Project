# PCA (EOF) analysis of NCEP/NCAR Reanalysis surface air temperature, 2013-2014

## Project at a glance
* **Department:** Department of Artificial Intelligence and Data Science, FAST-NUCES Karachi
* **Course:** Mathematical Foundation for AI, with Dr. Fahad Raiz, PhD
* **Team:** Jahan Zaib (MS student, 26k-7626) and Kariz Kamal (PhD student, 24k-9014)
* **Start:** 18 November 2026

Principal component analysis (called *EOF analysis* in climate science) of daily near-surface air temperature
over North America. The project explains and interprets **eigenvalues, eigenvectors (EOFs) and principal components**,
compares PCA of the full field with PCA of anomalies, tests which modes are statistically distinct (North's rule),
and shows how few components are needed to reconstruct the data.

## Data
* **NCEP/NCAR Reanalysis 1**, 4x-daily air temperature at sigma level 0.995, 15N-75N, 160W-30W, 2.5 degree grid,
  1 Jan 2013 - 31 Dec 2014 (25 x 53 grid points). Source: NOAA Physical Sciences Laboratory,
  <https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html>.
* The sample file `data/air_temperature.nc` is the copy distributed with the
  [xarray-data](https://github.com/pydata/xarray-data) repository.
* Daily means -> data matrix of **730 days x 1325 grid points**.

## Run it
**Kaggle:** *Create -> New Notebook -> File -> Import Notebook* and upload `pca_ncep_temperature_kaggle.ipynb`.
Turn **Internet ON** (Settings, right panel) and press *Run All*. (If Internet is not available to your account,
upload `data/air_temperature.nc` as a Kaggle dataset and attach it - the notebook finds it under `/kaggle/input`.)

**Locally:**
```bash
pip install -r requirements.txt
python pca_ncep_analysis.py        # same code as the notebook
```
Figures are written to `outputs/figures/`, tables to `outputs/tables/`, all numbers used in the report to
`outputs/results_summary.json`. The coastline file is downloaded automatically (maps still work without it).

## Repository contents
| Path | Description |
|---|---|
| `pca_ncep_temperature_kaggle.ipynb` | Notebook with explanations and results (executed) |
| `pca_ncep_analysis.py` | Same analysis as a plain script |
| `data/air_temperature.nc` | NCEP/NCAR temperature sample (7.7 MB) |
| `outputs/figures`, `outputs/tables` | Everything shown in the report |
| `report/Project_Report_PCA_NCEP.docx` | Written report |

## Method in brief
1. Daily mean temperature (degC); each grid point weighted by sqrt(cos latitude).
2. **Analysis A** - full field (seasonal cycle kept). **Analysis B** - anomalies after removing the mean, annual and
   semi-annual harmonics (only two years of data, so no day-by-day climatology).
3. PCA by SVD of the centred, weighted data matrix: eigenvalues = s^2/(n-1), EOFs = right singular vectors,
   PCs = U*S. Verified against `numpy.linalg.eigh` of the covariance matrix.
4. Number of modes: scree plot, cumulative variance, **North's rule** with effective sample size from lag-1 autocorrelation.
5. Interpretation: regression maps (degC per +1 SD of PC), correlation with seven regional temperature indices
   (5 % significance with effective sample size), reconstruction RMSE.

## Main results
| | Analysis A (full field) | Analysis B (anomalies) |
|---|---|---|
| Mode 1 variance | **80.7 %** (annual cycle, R = 0.99) | 14.9 % |
| Modes 2, 3, 4 | 2.6 %, 2.4 %, 2.0 % | 9.9 %, 9.7 %, 6.2 % |
| Modes needed for 80 % | 1 | 22 |

* EOF1-4 of the anomalies: north-central continental mode, west-east dipole, Alaska/NW Canada mode, Arctic Canada mode.
* Modes 2 and 3 are not separable by North's rule (eigenvalues 906 and 887).

## References
Kalnay et al. (1996), Bull. Amer. Meteor. Soc., 77, 437-471 - NCEP/NCAR reanalysis.
North et al. (1982), Mon. Wea. Rev., 110, 699-706 - sampling errors of EOFs.
Hannachi, Jolliffe and Stephenson (2007), Int. J. Climatol., 27, 1119-1152 - EOF review.
