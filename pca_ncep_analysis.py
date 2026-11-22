# %% [markdown]
# # Principal Component Analysis (EOF analysis) of NCEP/NCAR Reanalysis Surface Air Temperature
#
# **Authors:** Jahan Zaib (26k-7626, MS student) and Kariz Kamal (24k-9014, PhD student)
#
# **Department:** Department of Artificial Intelligence and Data Science, FAST-NUCES Karachi  
# **Course:** Mathematical Foundation for AI, with Dr. Fahad Raiz, PhD  
# **Project start:** 18 November 2026
#
# **Data:** NCEP/NCAR Reanalysis 1, 4x-daily air temperature at sigma level 0.995 (near surface),
# 2013-2014, 15N-75N, 160W-30W (North America), 2.5 degree grid.
# Source: NOAA Physical Sciences Laboratory (https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html).
# The sample file used here is the copy distributed with the `xarray-data` repository on GitHub.
#
# **What this notebook does**
# 1. Loads the data, converts Kelvin to degC and averages the four daily values into daily means.
# 2. **Analysis A** - PCA on the full field (seasonal cycle kept).
# 3. **Analysis B** - PCA on anomalies (seasonal cycle removed) -> the "weather" modes.
# 4. Interprets eigenvalues, eigenvectors (EOFs), principal components (scores), North's rule of thumb,
#    correlations with regional temperature indices, and reconstruction with few components.
# 5. Saves every figure and table used in the written report.
#
# *Kaggle tip:* switch **Internet ON** in the notebook settings (right panel) so the two small data files can be
# downloaded, or upload `air_temperature.nc` and `countries.geo.json` as a Kaggle dataset and the notebook will
# find them automatically under `/kaggle/input`.

# %%
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
})

DATA_DIR = Path("data")
FIG_DIR = Path("outputs/figures")
TAB_DIR = Path("outputs/tables")
for d in (DATA_DIR, FIG_DIR, TAB_DIR):
    d.mkdir(parents=True, exist_ok=True)

NC_URL = "https://raw.githubusercontent.com/pydata/xarray-data/master/air_temperature.nc"
GEO_URL = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"


def get_file(name, url):
    """Return a local path to `name`: look in data/, then in /kaggle/input, else download it."""
    local = DATA_DIR / name
    if local.exists():
        return local
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        hits = list(kaggle_input.glob(f"**/{name}"))
        if hits:
            return hits[0]
    print(f"Downloading {name} ...")
    urllib.request.urlretrieve(url, local)
    return local


# %% [markdown]
# ## 1. Load and prepare the data
# * Kelvin -> degC.
# * Four values per day (00, 06, 12, 18 UTC) are averaged into one **daily mean** (removes the day/night cycle),
#   giving n = 730 days (observations).
# * Each of the 25 x 53 = 1325 grid points is one **variable**.
#
# So the data matrix has shape **730 x 1325**.

# %%
ds = xr.open_dataset(get_file("air_temperature.nc", NC_URL))
T = (ds["air"] - 273.15).resample(time="1D").mean()          # daily mean, degC
lat, lon = T["lat"].values, T["lon"].values
time = pd.to_datetime(T["time"].values)
n, ny, nx = T.shape
p = ny * nx
field = T.values                                              # (730, 25, 53)
print(f"days n = {n}, grid points p = {p}, missing values = {int(np.isnan(field).sum())}")

# Area weights: grid boxes shrink towards the pole, so weight each point by sqrt(cos(latitude)).
w = np.sqrt(np.cos(np.deg2rad(lat)))[:, None] * np.ones((1, nx))      # (25, 53)

# Climatological (2-year) mean and the day index used for harmonics
clim_mean = field.mean(axis=0)
day = np.arange(n)
omega = 2 * np.pi / 365.25

# %% [markdown]
# ## 2. Helper functions: PCA by Singular Value Decomposition (SVD)
# For a centred, weighted data matrix `X` (n x p):  `X = U S Vt`
# * eigenvalues of the covariance matrix: `lambda_k = S_k^2 / (n-1)`
# * eigenvectors (EOFs):                  rows of `Vt` (unit length)
# * principal components (scores):        columns of `U S`

# %%
def eof_analysis(F, weights=w):
    """PCA of a (time, lat, lon) field. Returns eigenvalues, PCs (scores), EOFs (eigenvectors)."""
    nt = F.shape[0]
    X = (F * weights).reshape(nt, -1)
    X = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    eigvals = S ** 2 / (nt - 1)
    pcs = U * S
    # sign convention (the sign of an eigenvector is arbitrary): largest |coefficient| positive
    for k in range(len(S)):
        if Vt[k][np.argmax(np.abs(Vt[k]))] < 0:
            Vt[k] *= -1
            pcs[:, k] *= -1
    return eigvals, pcs, Vt


def lag1(x):
    return np.corrcoef(x[:-1], x[1:])[0, 1]


def n_eff(r1x, r1y=None):
    """Effective number of independent samples when consecutive days are correlated (AR(1) correction).
    One series: n(1-r1)/(1+r1) (Wilks 2011).  Two series: n(1-r1x*r1y)/(1+r1x*r1y) (Bretherton et al. 1999)."""
    if r1y is None:
        return n * (1 - r1x) / (1 + r1x)
    return n * (1 - r1x * r1y) / (1 + r1x * r1y)


def regression_maps(anom, z, k):
    """Regress anomaly field onto standardised PCs -> degC per +1 standard deviation of the PC."""
    A = anom.reshape(len(anom), -1)
    A = A - A.mean(0)
    return ((z[:, :k].T @ A) / (len(A) - 1)).reshape(k, ny, nx)


def box_mean(anom, lat_rng, lon_rng):
    """Area-weighted mean of an anomaly field over a lat/lon box -> time series."""
    la = (lat >= lat_rng[0]) & (lat <= lat_rng[1])
    lo = (lon >= lon_rng[0]) & (lon <= lon_rng[1])
    ww = np.cos(np.deg2rad(lat[la]))[:, None] * np.ones((1, lo.sum()))
    return (anom[:, la][:, :, lo] * ww).sum((1, 2)) / ww.sum()


# %% [markdown]
# ## 3. Analysis A - PCA of the full field (seasonal cycle kept)

# %%
lamA, pcA, eofA = eof_analysis(field)
# make PC1 positive in summer (warm) for readability
if np.corrcoef(pcA[:, 0], np.cos(omega * (day - 200)))[0, 1] < 0:
    pcA[:, 0] *= -1
    eofA[0] *= -1
varA = 100 * lamA / lamA.sum()
zA = pcA / pcA.std(axis=0, ddof=1)                     # standardised PCs
regA = regression_maps(field, zA, 3)

# How much of PC1 is the annual cycle?  Regress PC1 on cos & sin of the annual frequency.
H1 = np.column_stack([np.ones(n), np.cos(omega * day), np.sin(omega * day)])
coef, *_ = np.linalg.lstsq(H1, zA[:, 0], rcond=None)
fit1 = H1 @ coef
R_annual = np.corrcoef(fit1, zA[:, 0])[0, 1]
peak_doy = int(np.argmax(fit1[:366]) + 1)
print("Analysis A: variance explained (%) of PC1-5:", np.round(varA[:5], 2))
print(f"  PC1 vs annual harmonic: R = {R_annual:.3f}; fitted maximum on day-of-year {peak_doy}")

# %% [markdown]
# ## 4. Analysis B - PCA of anomalies (seasonal cycle removed)
# With only two years we cannot compute a reliable day-by-day climatology (the anomaly of year 1 would simply be
# minus the anomaly of year 2). Instead the seasonal cycle is removed by **least-squares harmonic regression**:
# a constant + annual harmonic + semi-annual harmonic fitted at every grid point and subtracted.

# %%
H = np.column_stack([np.ones(n)] + [f(k * omega * day) for k in (1, 2) for f in (np.cos, np.sin)])
beta, *_ = np.linalg.lstsq(H, field.reshape(n, -1), rcond=None)
seasonal = (H @ beta).reshape(n, ny, nx)
anom = field - seasonal                                          # degC anomalies

wvar_full = ((field - field.mean(0)) ** 2 * w ** 2).sum() / (n - 1)
wvar_anom = ((anom - anom.mean(0)) ** 2 * w ** 2).sum() / (n - 1)
seasonal_share = 100 * (1 - wvar_anom / wvar_full)
print(f"Seasonal cycle (annual+semi-annual harmonics) explains {seasonal_share:.1f}% of the total variance")

lamB, pcB, eofB = eof_analysis(anom)
varB = 100 * lamB / lamB.sum()
cumB = np.cumsum(varB)
zB = pcB / pcB.std(axis=0, ddof=1)
regB = regression_maps(anom, zB, 6)
print("Analysis B: variance explained (%) of PC1-10:", np.round(varB[:10], 2))
print("            cumulative (%):", np.round(cumB[:10], 1))

# %% [markdown]
# ## 5. Checks that the SVD result really is the textbook PCA
# 1. Eigenvalues from the **covariance matrix** (`numpy.linalg.eigh`) equal the SVD eigenvalues.
# 2. Eigenvalues add up to the total variance (trace of the covariance matrix).
# 3. EOFs are orthonormal; PCs are mutually uncorrelated.

# %%
Xw = (anom * w).reshape(n, -1)
Xw = Xw - Xw.mean(0)
C = Xw.T @ Xw / (n - 1)                                       # 1325 x 1325 covariance matrix
ev_cov = np.sort(np.linalg.eigvalsh(C))[::-1]
checks = {
    "max |eigenvalue(cov) - eigenvalue(SVD)| (first 20)": float(np.max(np.abs(ev_cov[:20] - lamB[:20]))),
    "trace(C) - sum(eigenvalues)": float(np.trace(C) - lamB.sum()),
    "max |EOF.EOF^T - I| (first 10)": float(np.max(np.abs(eofB[:10] @ eofB[:10].T - np.eye(10)))),
    "max |corr(PCi,PCj)|, i!=j (first 10)": float(np.max(np.abs(np.corrcoef(pcB[:, :10].T) - np.eye(10)))),
    "number of non-zero eigenvalues": int(np.sum(lamB > 1e-10 * lamB[0])),
    "total variance (weighted degC^2)": float(lamB.sum()),
}
for k, v in checks.items():
    print(f"{k:55s} {v:.3e}" if isinstance(v, float) else f"{k:55s} {v}")

# %% [markdown]
# ## 6. How many modes are physically meaningful?  North's rule of thumb
# Sampling error of an eigenvalue: `d(lambda) = lambda * sqrt(2 / N_eff)` (North et al., 1982).
# Neighbouring modes are considered *separable* only if their spacing exceeds this error.
# `N_eff = n (1 - r1) / (1 + r1)` where `r1` is the lag-1 autocorrelation of the PC (days are not independent).

# %%
K = 15
r1B = np.array([lag1(pcB[:, k]) for k in range(K)])
neffB = n_eff(r1B)
errB = lamB[:K] * np.sqrt(2 / neffB)
gap = lamB[:K - 1] - lamB[1:K]
sep = gap > np.maximum(errB[:K - 1], errB[1:K])
tblB = pd.DataFrame({
    "Mode": np.arange(1, K + 1),
    "Eigenvalue (degC^2)": lamB[:K],
    "Variance (%)": varB[:K],
    "Cumulative (%)": cumB[:K],
    "Lag-1 autocorr": r1B,
    "N_eff": neffB,
    "North error (degC^2)": errB,
    "Separated from next": list(sep) + [np.nan],
})
tblB.round(3).to_csv(TAB_DIR / "table_eigenvalues_anomaly_PCA.csv", index=False)
print(tblB.round(3).head(8).to_string(index=False))

r1A = np.array([lag1(pcA[:, k]) for k in range(10)])
tblA = pd.DataFrame({"Mode": np.arange(1, 11), "Eigenvalue (degC^2)": lamA[:10], "Variance (%)": varA[:10],
                     "Cumulative (%)": np.cumsum(varA[:10]), "Lag-1 autocorr": r1A})
tblA.round(3).to_csv(TAB_DIR / "table_eigenvalues_full_field_PCA.csv", index=False)

# %% [markdown]
# ## 7. What does each principal component represent?  Correlation with regional temperature indices
# Area-mean temperature anomalies are computed for seven regions; each is correlated with PC1-PC4.
# Significance (5 % level, two-sided t-test) uses the effective sample size of the *pair* of series.

# %%
regions = {
    "Alaska / NW Canada (60-70N, 155-125W)": ((60, 70), (205, 235)),
    "Western USA (32-45N, 125-110W)": ((32, 45), (235, 250)),
    "Central USA / Plains (35-47.5N, 105-92W)": ((35, 47.5), (255, 268)),
    "Eastern USA / Gt Lakes (35-47.5N, 88-75W)": ((35, 47.5), (272, 285)),
    "Hudson Bay / Arctic Canada (55-70N, 95-65W)": ((55, 70), (265, 295)),
    "Greenland / N. Atlantic (60-75N, 60-30W)": ((60, 75), (300, 330)),
    "Mexico / Gulf (17.5-27.5N, 105-85W)": ((17.5, 27.5), (255, 275)),
}
rows = []
for name, (la, lo) in regions.items():
    ts = box_mean(anom, la, lo)
    row = {"Region": name}
    for k in range(4):
        r = np.corrcoef(ts, zB[:, k])[0, 1]
        ne = n_eff(lag1(ts), lag1(zB[:, k]))
        t = r * np.sqrt((ne - 2) / (1 - r ** 2))
        pval = 2 * stats.t.sf(abs(t), ne - 2)
        row[f"PC{k + 1}"] = r
        row[f"PC{k + 1} sig"] = bool(pval < 0.05)
    rows.append(row)
tblR = pd.DataFrame(rows)
tblR.round(3).to_csv(TAB_DIR / "table_region_correlations.csv", index=False)
print(tblR.round(2).to_string(index=False))

# %% [markdown]
# ## 8. Reconstruction: how well do k components describe the 1325-point field?
# `X_k = sum_{j<=k} PC_j * EOF_j`  (done on the weighted anomalies, then un-weighted to degC).

# %%
Xanom = (anom - anom.mean(0)).reshape(n, -1)
ks = [1, 2, 3, 4, 5, 10, 20, 50, 100, 200]
rec_rows = []
for k in ks:
    Xk = (pcB[:, :k] @ eofB[:k]).reshape(n, ny, nx) / w      # back to degC
    rmse = np.sqrt(np.mean((Xk - (anom - anom.mean(0))) ** 2))
    rec_rows.append({"k": k, "Cumulative variance (%)": cumB[k - 1], "RMSE (degC)": rmse})
tblK = pd.DataFrame(rec_rows)
tblK.round(3).to_csv(TAB_DIR / "table_reconstruction.csv", index=False)
print(tblK.round(3).to_string(index=False))
n_for = {pct: int(np.searchsorted(cumB, pct) + 1) for pct in (50, 70, 80, 90)}
print("components needed for 50/70/80/90 % variance:", n_for)
rmse_all = [np.sqrt(np.mean(((pcB[:, :k] @ eofB[:k]).reshape(n, ny, nx) / w - (anom - anom.mean(0))) ** 2))
            for k in range(1, 101)]
std_anom = anom.std()
print(f"overall std of anomalies = {std_anom:.2f} degC")

# %% [markdown]
# ## 8b. Worked examples: eigenvector vs. loading vs. regression vs. correlation, and one day's score
# For selected grid points we print, for EOF1 and EOF2 of the anomaly PCA:
# * **eigenvector coefficient** - entry of the unit-length EOF (weighted space, no units),
# * **regression** - degC change at that point for +1 standard deviation of the PC,
# * **correlation** - r between the PC and the local temperature anomaly.

# %%
points = {"Winnipeg area (50N, 97.5W)": (50.0, 262.5), "Chicago area (42.5N, 87.5W)": (42.5, 272.5),
          "Denver area (40N, 105W)": (40.0, 255.0), "Anchorage area (62.5N, 150W)": (62.5, 210.0),
          "Gulf of Mexico (27.5N, 92.5W)": (27.5, 267.5)}
prow = []
for name, (la, lo) in points.items():
    i, j = int(np.argmin(np.abs(lat - la))), int(np.argmin(np.abs(lon - lo)))
    row = {"Point": name}
    for k in range(2):
        r = np.corrcoef(zB[:, k], anom[:, i, j])[0, 1]
        row[f"EOF{k + 1} eigenvector"] = eofB[k].reshape(ny, nx)[i, j]
        row[f"EOF{k + 1} regression (degC/SD)"] = regB[k][i, j]
        row[f"EOF{k + 1} correlation"] = r
    prow.append(row)
tblP = pd.DataFrame(prow)
tblP.round(3).to_csv(TAB_DIR / "table_eof_values_at_points.csv", index=False)
print(tblP.round(3).to_string(index=False))

# Worked example for the day with the largest |PC1|
i_day = int(np.argmax(np.abs(zB[:, 0])))
ig, jg = np.unravel_index(np.argmax(np.abs(regB[0])), (ny, nx))
obs = anom[i_day, ig, jg] - anom[:, ig, jg].mean()
rec = {k: float(((pcB[i_day:i_day + 1, :k] @ eofB[:k]).reshape(ny, nx) / w)[ig, jg]) for k in (1, 4, 20)}
worked = {"date": str(time[i_day].date()), "lat": float(lat[ig]), "lon": float(lon[jg]), "z_pc1": float(zB[i_day, 0]),
          "regression": float(regB[0][ig, jg]), "observed": float(obs), "reconstructed": rec,
          "eof1_coeff": float(eofB[0].reshape(ny, nx)[ig, jg])}
print("Worked example:", worked)

# Which temporal pattern is PC2 of Analysis A?  (correlate with annual quadrature & semi-annual harmonics)
H2 = np.column_stack([np.ones(n), np.cos(omega * day), np.sin(omega * day), np.cos(2 * omega * day), np.sin(2 * omega * day)])
pcA_fit = {}
for k in range(1, 4):
    c_, *_ = np.linalg.lstsq(H2, zA[:, k], rcond=None)
    pcA_fit[f"PC{k + 1}"] = float(np.corrcoef(H2 @ c_, zA[:, k])[0, 1])
print("Analysis A, PC2-4: multiple R with annual+semiannual harmonics:", pcA_fit)

# Spatial (area-weighted) pattern correlation between Analysis A modes 2-5 and Analysis B modes 1-4
def pattern_corr(a, b):
    a, b = (a * w).ravel(), (b * w).ravel()
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))

regA5 = regression_maps(field, zA, 5)
pat = pd.DataFrame([[pattern_corr(regA5[i], regB[j]) for j in range(4)] for i in range(1, 5)],
                   index=[f"A-EOF{i + 1}" for i in range(1, 5)], columns=[f"B-EOF{j + 1}" for j in range(4)])
pat.round(2).to_csv(TAB_DIR / "table_pattern_correlation_A_vs_B.csv")
print(pat.round(2))

# %% [markdown]
# ## 9. Figures

# %%
import matplotlib.dates as mdates

# --- coastlines (country borders from GeoJSON) ---------------------------------------------------
try:
    geo = json.load(open(get_file("countries.geo.json", GEO_URL)))
    lines = []
    for feat in geo["features"]:
        g = feat["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            for ring in poly:
                a = np.array(ring)
                a[:, 0] = np.where(a[:, 0] < 0, a[:, 0] + 360, a[:, 0])
                lines.append(a)
except Exception as err:                                            # maps still work without outlines
    print("Coastlines unavailable:", err)
    lines = []

DEG_C = "\u00b0C"


def base_map(ax):
    for a in lines:
        if np.abs(np.diff(a[:, 0])).max() < 180:
            ax.plot(a[:, 0], a[:, 1], color="0.25", lw=0.5)
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())
    ax.set_aspect(1 / np.cos(np.deg2rad(45)))
    ax.set_xticks([200, 220, 240, 260, 280, 300, 320])
    ax.set_xticklabels([f"{360 - x}\u00b0W" for x in (200, 220, 240, 260, 280, 300, 320)])
    ax.set_yticks([20, 30, 40, 50, 60, 70])
    ax.set_yticklabels([f"{y}\u00b0N" for y in (20, 30, 40, 50, 60, 70)])


def nice_limit(m):
    """Round a maximum up to a 'nice' number so colour-bar labels are readable."""
    step = 0.5 if m < 5 else 1 if m < 10 else 2 if m < 20 else 4
    return float(np.ceil(m / step) * step)


def show_map(ax, data, cmap, sym=False, vmax=None, levels=None):
    """Filled-contour map. sym=True -> symmetric diverging scale centred on zero."""
    if sym:
        m = nice_limit(np.abs(data).max() if vmax is None else vmax)
        cf = ax.contourf(lon, lat, data, levels=np.linspace(-m, m, 21), cmap=cmap,
                         norm=TwoSlopeNorm(vcenter=0, vmin=-m, vmax=m))
        cf.ticks = np.linspace(-m, m, 5)
    else:
        cf = ax.contourf(lon, lat, data, levels=levels, cmap=cmap)
        cf.ticks = None
    base_map(ax)
    return cf


def add_cbar(fig, cf, ax, label, **kw):
    cb = fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.06, label=label, format="%g", **kw)
    if cf.ticks is not None:
        cb.set_ticks(cf.ticks)
    return cb


def date_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))


# --- Fig 1: climatological mean and variability ----------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
cf = show_map(axs[0], clim_mean, "RdYlBu_r", levels=np.arange(-30, 36, 3))
add_cbar(fig, cf, axs[0], DEG_C, shrink=0.9)
axs[0].set_title("(a) Mean temperature, 2013\u20132014")
cf = show_map(axs[1], anom.std(0), "viridis", levels=np.arange(0, 8.5, 0.5))
add_cbar(fig, cf, axs[1], DEG_C, shrink=0.9)
axs[1].set_title("(b) Std. dev. of daily anomalies")
fig.savefig(FIG_DIR / "fig1_climatology_and_variability.png")

# --- Fig 2: scree plots -------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
m = np.arange(1, 11)
axs[0].bar(m, varA[:10], color="0.55")
axs[0].plot(m, varA[:10], "ko-", ms=3)
axs[0].set_title("(a) Analysis A: full field")
axs[0].set_xlabel("Mode number")
axs[0].set_ylabel("Variance explained (%)")
axs[0].set_xticks(m)
axs[0].text(1.5, varA[0] - 5, f"{varA[0]:.1f}%", fontsize=9)
m = np.arange(1, K + 1)
axs[1].errorbar(m, varB[:K], yerr=100 * errB / lamB.sum(), fmt="ko-", ms=3.5, capsize=3, lw=1)
axs[1].set_title("(b) Analysis B: anomalies (North error bars)")
axs[1].set_xlabel("Mode number")
axs[1].set_ylabel("Variance explained (%)")
axs[1].set_xticks(m)
axs[1].grid(alpha=0.3)
fig.savefig(FIG_DIR / "fig2_scree_plots.png")

# --- Fig 3: Analysis A, EOF1 map and PC1 --------------------------------------------------------
fig = plt.figure(figsize=(7.2, 3.3), constrained_layout=True)
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.1])
ax = fig.add_subplot(gs[0])
cf = show_map(ax, regA[0], "RdBu_r", sym=True)
add_cbar(fig, cf, ax, f"{DEG_C} per +1 SD of PC1", shrink=0.9)
ax.set_title(f"(a) EOF1 ({varA[0]:.1f}% of variance)")
ax = fig.add_subplot(gs[1])
ax.plot(time, zA[:, 0], color="0.2", lw=0.9, label="PC1 (standardised)")
ax.plot(time, fit1, color="tab:red", lw=1.1, ls="--", label=f"annual harmonic fit (R = {R_annual:.2f})")
ax.set_title("(b) PC1 time series")
ax.set_ylabel("Standardised score")
ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
date_axis(ax)
ax.grid(alpha=0.3)
fig.savefig(FIG_DIR / "fig3_analysisA_EOF1_PC1.png")

# --- Fig 4: Analysis B, EOF1-4 (regression maps) ---------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
vm = np.abs(regB[:4]).max()
for k, ax in enumerate(axs.ravel()):
    cf = show_map(ax, regB[k], "RdBu_r", sym=True, vmax=vm)
    ax.set_title(f"EOF{k + 1} ({varB[k]:.1f}%)")
cb = fig.colorbar(cf, ax=axs, orientation="horizontal", pad=0.03, shrink=0.55, format="%g",
                  label=f"{DEG_C} per +1 SD of the PC")
cb.set_ticks(cf.ticks)
fig.savefig(FIG_DIR / "fig4_analysisB_EOF1to4.png")

# --- Fig 5: PC time series ---------------------------------------------------------------------
fig, axs = plt.subplots(4, 1, figsize=(7.2, 5.8), sharex=True, constrained_layout=True)
for k, ax in enumerate(axs):
    ax.plot(time, zB[:, k], color="0.15", lw=0.8)
    ax.axhline(0, color="0.5", lw=0.5)
    ax.fill_between(time, 0, zB[:, k], where=zB[:, k] > 0, color="tab:red", alpha=0.35)
    ax.fill_between(time, 0, zB[:, k], where=zB[:, k] < 0, color="tab:blue", alpha=0.35)
    ax.set_ylabel(f"PC{k + 1}")
    ax.grid(alpha=0.25)
date_axis(axs[-1])
fig.savefig(FIG_DIR / "fig5_analysisB_PC1to4.png")

# --- Fig 6: score plot (PC1-PC2) by season, and seasonal spread of the scores ----------------------
season = np.array([{12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA",
                    9: "SON", 10: "SON", 11: "SON"}[mth] for mth in time.month])
cols = {"DJF": "tab:blue", "MAM": "tab:green", "JJA": "tab:red", "SON": "tab:orange"}
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True)
for s_, c in cols.items():
    sel = season == s_
    axs[0].scatter(zB[sel, 0], zB[sel, 1], s=9, color=c, alpha=0.7, label=s_)
axs[0].axhline(0, color="0.6", lw=0.5)
axs[0].axvline(0, color="0.6", lw=0.5)
axs[0].set_xlabel("PC1 score (standardised)")
axs[0].set_ylabel("PC2 score (standardised)")
axs[0].set_title("(a) Score plot: each dot is one day")
axs[0].legend(frameon=False, title="Season", fontsize=8)
axs[0].grid(alpha=0.25)
bw = 0.2
for j, (s_, c) in enumerate(cols.items()):
    axs[1].bar(np.arange(4) + (j - 1.5) * bw, [zB[season == s_, k].std() for k in range(4)], bw, color=c, label=s_)
axs[1].set_xticks(np.arange(4))
axs[1].set_xticklabels([f"PC{k + 1}" for k in range(4)])
axs[1].set_ylabel("Std. dev. of standardised score")
axs[1].set_title("(b) Spread of scores by season")
axs[1].grid(alpha=0.25, axis="y")
fig.savefig(FIG_DIR / "fig6_score_plots.png")

# --- Fig 7: reconstruction of one day ------------------------------------------------------------
i_ex = int(np.argmax(np.abs(zB[:, 0])))
ex_date = time[i_ex]
print("Example day (largest |PC1|):", ex_date.date())
orig = anom[i_ex] - anom.mean(0)
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
vm = np.abs(orig).max()
for ax, (lab, kk) in zip(axs.ravel(), [("Original anomaly", None), ("1 PC", 1), ("4 PCs", 4), ("20 PCs", 20)]):
    d = orig if kk is None else (pcB[i_ex:i_ex + 1, :kk] @ eofB[:kk]).reshape(ny, nx) / w
    cf = show_map(ax, d, "RdBu_r", sym=True, vmax=vm)
    ax.set_title(lab if kk is None else f"Reconstruction with {lab} ({cumB[kk - 1]:.0f}% variance)")
cb = fig.colorbar(cf, ax=axs, orientation="horizontal", pad=0.03, shrink=0.55, format="%g", label=DEG_C)
cb.set_ticks(cf.ticks)
fig.suptitle(f"Temperature anomaly on {ex_date:%d %B %Y}", y=1.03)
fig.savefig(FIG_DIR / "fig7_reconstruction_example.png")

# --- Fig 8: cumulative variance and RMSE ---------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
mm = np.arange(1, 101)
axs[0].plot(mm, np.cumsum(varB)[:100], "k-", label="B: anomalies")
axs[0].plot(mm, np.cumsum(varA)[:100], "k--", label="A: full field")
for pct in (80, 90):
    axs[0].axhline(pct, color="0.6", lw=0.6)
axs[0].set_xlabel("Number of components retained")
axs[0].set_ylabel("Cumulative variance (%)")
axs[0].set_title("(a) Cumulative variance")
axs[0].legend(frameon=False, loc="lower right")
axs[0].grid(alpha=0.3)
axs[1].plot(mm, rmse_all, "k-")
axs[1].set_xlabel("Number of components retained")
axs[1].set_ylabel(f"Reconstruction RMSE ({DEG_C})")
axs[1].set_title("(b) Error of the reconstruction (B)")
axs[1].grid(alpha=0.3)
fig.savefig(FIG_DIR / "fig8_cumulative_variance_and_rmse.png")
plt.show()

# %% [markdown]
# ## 10. Save every number used in the written report

# %%
# date of the largest values of PC1-PC4 and pattern summary (used when interpreting results)
extremes = {}
for k in range(4):
    i_max, i_min = int(np.argmax(zB[:, k])), int(np.argmin(zB[:, k]))
    extremes[f"PC{k + 1}"] = {"max_date": str(time[i_max].date()), "max_val": float(zB[i_max, k]),
                              "min_date": str(time[i_min].date()), "min_val": float(zB[i_min, k])}

def season_std(k):
    return {s: float(zB[season == s, k].std()) for s in cols}

summary = {
    "n_days": n, "n_points": p, "n_lat": ny, "n_lon": nx,
    "mean_temp_degC": float(clim_mean.mean()), "min_clim": float(clim_mean.min()), "max_clim": float(clim_mean.max()),
    "seasonal_share_pct": float(seasonal_share),
    "A": {"var": varA[:10].tolist(), "eig": lamA[:10].tolist(), "R_annual": float(R_annual), "peak_doy": peak_doy,
          "total_var": float(lamA.sum()), "lag1_pc1": float(r1A[0]),
          "n_nonzero": int(np.sum(lamA > 1e-10 * lamA[0]))},
    "B": {"var": varB[:15].tolist(), "eig": lamB[:15].tolist(), "cum": cumB[:15].tolist(),
          "total_var": float(lamB.sum()), "n_for": n_for, "std_anom": float(std_anom),
          "r1": r1B.tolist(), "neff": neffB.tolist(), "err": errB.tolist(), "sep": [bool(s) for s in sep]},
    "checks": checks, "extremes": extremes, "example_day": str(ex_date.date()),
    "season_std_pc1": season_std(0), "season_std_pc2": season_std(1),
    "season_std_pc3": season_std(2), "season_std_pc4": season_std(3),
    "rmse_k": {int(r["k"]): float(r["RMSE (degC)"]) for r in rec_rows},
    "std_by_season": {s: float(anom[season == s].std()) for s in cols},
    "worked": worked, "pcA_harmonic_R": pcA_fit, "pattern_corr": pat.round(3).to_dict(),
}
def peak(m):
    i, j = np.unravel_index(np.argmax(np.abs(m)), m.shape)
    return {"lat": float(lat[i]), "lon": float(lon[j]), "value": float(m[i, j])}

summary["peaks"] = {
    "A_EOF1": peak(regA[0]), "A_EOF1_min": float(regA[0].min()), "A_EOF1_max": float(regA[0].max()),
    "A_EOF1_min_loc": [float(lat[np.unravel_index(np.argmin(regA[0]), regA[0].shape)[0]]),
                       float(lon[np.unravel_index(np.argmin(regA[0]), regA[0].shape)[1]])],
    "A_EOF1_mean_south_of_25N": float(regA[0][lat <= 25].mean()),
    "A_EOF1_mean_north_of_55N": float(regA[0][lat >= 55].mean()),
    "B_EOF": {f"EOF{k + 1}": peak(regB[k]) for k in range(4)},
    "B_EOF_min_max": {f"EOF{k + 1}": [float(regB[k].min()), float(regB[k].max())] for k in range(4)},
    "B_std_max": peak(anom.std(0)),
}
with open("outputs/results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Saved figures to", FIG_DIR, "tables to", TAB_DIR)
