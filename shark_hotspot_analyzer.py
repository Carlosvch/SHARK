#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from pykalman import KalmanFilter
from scipy import signal, stats
from sklearn.cluster import DBSCAN
from sklearn.metrics import (accuracy_score, classification_report, f1_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.simplefilter("ignore", UserWarning)
pd.options.mode.copy_on_write = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

@dataclass
class Config:
    movebank_csv: Path = Path("Whale shark movements in Gulf of Mexico.csv")
    data_dir: Path = Path("data")
    results_dir: Path = Path("results")

    sst_patterns: Tuple[str, ...] = ("sst/sst.day.mean.*.nc", "sst/sst.day.mean.*")
    chl_pattern: str = "chlorophyll_a/*chlor_a*"
    ssh_patterns: Tuple[str, ...] = (
        "altimetry/*duacs*0.125deg*P1D*.nc",
        "altimetry/*duacs*0.25deg*P1D*.nc",
        "altimetry/*.nc",
        "altimetry/*",
    )

    resample_rule: str = "6h"
    region_margin_deg: float = 5.0
    id_col_preference: Tuple[str, ...] = ("individual-local-identifier", "tag-local-identifier")
    min_points_per_individual: int = 5

    fft_window: int = 24
    speed_quantile_for_foraging: float = 0.3
    turn_angle_th_deg: float = 45.0

    dbscan_eps_km: float = 40.0
    dbscan_min_samples: int = 25
    unknown_class_index: int = 0
    max_radius_km: float = 110.0
    radius_quantile: int = 80

    window_size: int = 12
    batch_size: int = 32
    gru_hidden_size: int = 128
    gru_num_layers: int = 1
    gru_bidirectional: bool = True
    gru_dropout: float = 0.5
    epochs: int = 25
    learning_rate: float = 1e-3
    early_stopping_patience: int = 8
    gradient_clip_value: float = 1.0
    lambda_zone: float = 0.4

    split_ratios: Dict[str, float] = field(default_factory=lambda: {"train": 0.7, "val": 0.15})
    seed: int = 42

    enable_corr_analysis: bool = True
    corr_compute_on: str = "train"
    corr_add_as_features: bool = True
    corr_spatial_res_deg: float = 0.25
    corr_temporal_rule: str = "1D"
    corr_min_obs_time: int = 5
    corr_min_obs_freq: int = 12
    corr_max_lag_steps: int = 5

    enable_undersampling: bool = True
    undersample_factor: int = 3
    undersample_exclude_classes: List[int] = field(default_factory=lambda: [0])

    enable_qna: bool = True
    
    map_plot_tracks: bool = True
    map_gap_hours: int = 24
    map_downsample_step: int = 1
    map_max_labeled_ids: int = 20

    def __post_init__(self):
        self.qna_output_dir = self.results_dir / "qna"
        self.corr_output_dir = self.results_dir / "corr"
        self.results_dir.mkdir(exist_ok=True)
        self.qna_output_dir.mkdir(exist_ok=True)
        self.corr_output_dir.mkdir(exist_ok=True)

CONFIG = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(CONFIG.seed)
torch.manual_seed(CONFIG.seed)

def to_360(lon):
    a = np.asarray(lon)
    a = np.where(a < 0, a + 360.0, a)
    a = np.where(a >= 360.0, a - 360.0, a)
    return a

def _to_180(lon):
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1 = np.radians(lat1); p2 = np.radians(lat2)
    dlat = p2 - p1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2.0)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlon/2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def earth_dx_dy_m(lat_deg: np.ndarray, dlon_deg: float, dlat_deg: float):
    R = 6371000.0
    lat_rad = np.radians(lat_deg)
    dy = dlat_deg * (np.pi/180.0) * R
    dx = dlon_deg * (np.pi/180.0) * R * np.cos(lat_rad)
    return dx, dy

def lon_diff_min_deg(lon_deg: pd.Series) -> pd.Series:
    d = lon_deg.diff()
    d = ((d + 180.0) % 360.0) - 180.0
    return d

def detect_coords(ds: xr.Dataset) -> Tuple[str, str, str]:
    time_names = ["time", "TIME", "t"]
    lat_names  = ["lat", "latitude", "y", "nav_lat"]
    lon_names  = ["lon", "longitude", "x", "nav_lon"]
    def pick(cands):
        for c in cands:
            if c in ds.coords or c in ds.dims or c in ds.variables:
                return c
        return None
    t = pick(time_names); la = pick(lat_names); lo = pick(lon_names)
    if t is None or la is None or lo is None:
        raise ValueError("Could not detect time/lat/lon coords.")
    return t, la, lo

def ensure_sorted(ds, lat="lat", lon="lon"):
    if ds[lon].min() < 0:
        ds = ds.assign_coords({lon: ((ds[lon] + 360) % 360)})
    if not np.all(np.diff(ds[lon].values) > 0):
        ds = ds.sortby(lon)
    try:
        if ds[lat][0] > ds[lat][-1]:
            ds = ds.sortby(lat)
    except Exception:
        pass
    return ds

def sel_bbox(ds, la, lo, bbox):
    lat_vals = ds[la].values
    lat_slice = slice(bbox["lat_min"], bbox["lat_max"]) if lat_vals[0] < lat_vals[-1] else slice(bbox["lat_max"], bbox["lat_min"])
    lon_min, lon_max = bbox["lon_min"], bbox["lon_max"]
    try:
        return ds.sel({la: lat_slice, lo: slice(lon_min, lon_max)})
    except Exception:
        return ds

def find_files(patterns: List[str]) -> Tuple[List[str], Optional[str]]:
    for p in patterns:
        files = sorted(glob(p))
        if files:
            return files, p
    return [], None

def open_satellite_safe(patterns: List[str], var_candidates: List[str], bbox: Dict[str, float], label: str):
    files, used_pat = find_files(patterns)
    logging.info(f"  ↳ {label}: found {len(files)} files (pattern used: {used_pat})")
    if not files:
        return None, None, f"no files for pattern(s) {patterns}"
    try:
        ds = xr.open_mfdataset(files, combine="by_coords", chunks="auto",
                               coords="minimal", data_vars="minimal",
                               compat="override", join="override")
    except Exception as e1:
        try:
            ds = xr.open_mfdataset(files, combine="nested", concat_dim="time", chunks="auto",
                                   coords="minimal", data_vars="minimal",
                                   compat="override", join="override")
        except Exception as e2:
            return None, None, f"could not open ({e1} // fallback {e2})"
    try:
        t, la, lo = detect_coords(ds)
        if t != "time" or la != "lat" or lo != "lon":
            ds = ds.rename({t:"time", la:"lat", lo:"lon"})
        ds = ensure_sorted(ds, "lat", "lon")
        ds = sel_bbox(ds, "lat", "lon", bbox)
    except Exception as e:
        return None, None, f"coords/bbox: {e}"
    var = None
    for c in var_candidates:
        if c in ds.data_vars:
            var = c; break
    if var is None:
        return None, None, f"variables {var_candidates} not found in {list(ds.data_vars)}"
    return ds, var, None

def open_modis_chl_l3m(pattern: str, bbox: Dict[str, float]):
    files = sorted(glob(pattern))
    logging.info(f"  ↳ CHL: found {len(files)} files (pattern used: {pattern})")
    if not files:
        return None, None, "no files"
    dsets = []
    for fp in files:
        ds = xr.open_dataset(fp, decode_times=False)
        lat = "lat" if "lat" in ds.dims or "lat" in ds.coords else "latitude"
        lon = "lon" if "lon" in ds.dims or "lon" in ds.coords else "longitude"
        var = "chlor_a" if "chlor_a" in ds.data_vars else ("CHL_chlor_a" if "CHL_chlor_a" in ds.data_vars else None)
        if var is None:
            return None, None, f"{os.path.basename(fp)} without chlor_a"
        if lat != "lat" or lon != "lon":
            ds = ds.rename({lat:"lat", lon:"lon"})
        if float(ds["lon"].min()) < 0:
            ds = ds.assign_coords(lon=((ds["lon"] + 360) % 360))
        ds = ensure_sorted(ds, "lat", "lon")
        try:
            ds = ds.sel(lat=slice(bbox["lat_min"], bbox["lat_max"]),
                        lon=slice(bbox["lon_min"], bbox["lon_max"]))
        except Exception:
            pass
        m = re.search(r"(\d{8})_(\d{8})", fp)
        if m:
            t0 = pd.to_datetime(m.group(1), format="%Y%m%d")
            t1 = pd.to_datetime(m.group(2), format="%Y%m%d")
            tt = t0 + (t1 - t0)/2
        else:
            t0 = pd.to_datetime(ds.attrs.get("time_coverage_start","2000-01-01"))
            t1 = pd.to_datetime(ds.attrs.get("time_coverage_end", str(t0)))
            tt = t0 + (t1 - t0)/2
        ds = ds[[var]].expand_dims(time=[np.datetime64(tt, "ns")]).rename({var:"chlor_a"})
        dsets.append(ds)
    out = xr.concat(dsets, dim="time")
    return out, "chlor_a", None

def grid_res_deg(ds: xr.Dataset) -> Tuple[float, float]:
    dlat = float(np.abs(ds["lat"].diff("lat").median()))
    dlon = float(np.abs(ds["lon"].diff("lon").median()))
    if not np.isfinite(dlat) or dlat <= 0: dlat = 0.25
    if not np.isfinite(dlon) or dlon <= 0: dlon = 0.25
    return dlat, dlon

def _to_writable_float_1d(arr_like) -> np.ndarray:
    arr = np.asarray(arr_like, dtype=float)
    if arr.ndim > 1:
        arr = arr.ravel()
    if not arr.flags.writeable:
        arr = np.array(arr, dtype=float, copy=True)
    else:
        arr = arr.astype(float, copy=True)
    arr.setflags(write=True)
    return arr

def _interp_linear_then_nearest(da: xr.DataArray, times, lats, lons) -> np.ndarray:
    coords = xr.Dataset({
        "time": ("obs", np.asarray(times)),
        "lat":  ("obs", np.asarray(lats)),
        "lon":  ("obs", to_360(np.asarray(lons)))
    })
    da_lin = da.interp(coords, method="linear").compute()
    v = _to_writable_float_1d(da_lin.to_pandas().values)
    mask = ~np.isfinite(v)
    if np.any(mask):
        da_near = da.interp(coords, method="nearest").compute()
        vn = _to_writable_float_1d(da_near.to_pandas().values)
        v[mask] = vn[mask]
    return v

def central_gradient(da: xr.DataArray, times, lats, lons, dlat_deg, dlon_deg):
    f_xp = _interp_linear_then_nearest(da, times, lats, to_360(np.asarray(lons) + dlon_deg))
    f_xm = _interp_linear_then_nearest(da, times, lats, to_360(np.asarray(lons) - dlon_deg))
    f_yp = _interp_linear_then_nearest(da, times, np.asarray(lats) + dlat_deg, to_360(lons))
    f_ym = _interp_linear_then_nearest(da, times, np.asarray(lats) - dlat_deg, to_360(lons))
    dx, dy = earth_dx_dy_m(np.asarray(lats), dlon_deg, dlat_deg)
    dfdx = (f_xp - f_xm)/(2.0*dx); dfdy = (f_yp - f_ym)/(2.0*dy)
    return dfdx, dfdy

def geostrophic_from_ssh(ssh_dfdx, ssh_dfdy, lats_deg):
    g = 9.81; omega = 7.292115e-5
    f = 2 * omega * np.sin(np.radians(lats_deg))
    f = np.where(np.abs(f) < 1e-10, np.sign(f)*1e-10 + 1e-10, f)
    u = -g * ssh_dfdy / f
    v =  g * ssh_dfdx / f
    return u, v

def apply_kalman_filter(df):
    if df.empty or len(df) < 2:
        return pd.DataFrame()
    z = df[['longitude', 'latitude']].values
    kf = KalmanFilter(
        transition_matrices=np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]),
        observation_matrices=np.array([[1,0,0,0],[0,1,0,0]]),
        initial_state_mean=np.array([z[0,0], z[0,1], 0, 0]),
        transition_covariance=0.01*np.eye(4),
        observation_covariance=0.1*np.eye(2),
        n_dim_state=4, n_dim_obs=2
    )
    state, _ = kf.filter(z)
    out = df.copy()
    out["lon_kalman"] = state[:,0]
    out["lat_kalman"] = state[:,1]
    out["vel_lon_kalman"] = state[:,2]
    out["vel_lat_kalman"] = state[:,3]
    return out

def rolling_fft_features(series: pd.Series, window: int, dt_hours: float):
    def _fft_pow(x):
        x = np.asarray(x, float)
        if np.any(~np.isfinite(x)): return np.nan, np.nan, np.nan
        x = x - x.mean()
        spec = np.fft.rfft(x)
        power = (np.abs(spec)**2) / len(x)
        freqs = np.fft.rfftfreq(len(x), d=dt_hours*3600.0)
        ny = freqs.max(); cut = ny / 3.0
        low  = power[freqs <= cut].sum()
        high = power[freqs >  cut].sum()
        return low, high, low/(high + 1e-12)
    lows, highs, ratios = [], [], []
    arr = series.values
    for i in range(len(arr)):
        if i < window - 1:
            lows.append(np.nan); highs.append(np.nan); ratios.append(np.nan)
        else:
            l, h, r = _fft_pow(arr[i - window + 1:i + 1])
            lows.append(l); highs.append(h); ratios.append(r)
    idx = series.index
    return pd.Series(lows, idx), pd.Series(highs, idx), pd.Series(ratios, idx)

def compute_foraging_proxy(df):
    dt_h = df["timestamp"].diff().dt.total_seconds().values / 3600.0
    dist_km = haversine_km(df["lat_kalman"].shift(1), df["lon_kalman"].shift(1),
                             df["lat_kalman"], df["lon_kalman"])
    speed_kmh = dist_km / (dt_h + 1e-6)
    heading = np.degrees(np.arctan2(df["lat_kalman"].diff(), df["lon_kalman"].diff()))
    turn = (heading - heading.shift(1)); turn = (turn + 180) % 360 - 180
    speed_th = np.nanquantile(speed_kmh, CONFIG.speed_quantile_for_foraging)
    turn_th = CONFIG.turn_angle_th_deg
    label = ((speed_kmh < speed_th) & (np.abs(turn) > turn_th)).astype(float)
    return pd.Series(label, index=df.index).fillna(0.0)

def create_features(df, sst, chl, ssh):
    if df.empty:
        return pd.DataFrame()
    out = df.copy()

    out["speed_deg"] = np.sqrt(out["vel_lon_kalman"]**2 + out["vel_lat_kalman"]**2)
    out["accel_deg_h2"] = out["speed_deg"].diff() / (out["timestamp"].diff().dt.total_seconds() / 3600.0 + 1e-6)

    out["hour"] = out["timestamp"].dt.hour
    out["doy"]  = out["timestamp"].dt.dayofyear
    out["hour_sin"] = np.sin(2*np.pi*out["hour"]/24.0)
    out["hour_cos"] = np.cos(2*np.pi*out["hour"]/24.0)
    out["doy_sin"]  = np.sin(2*np.pi*out["doy"]/365.0)
    out["doy_cos"]  = np.cos(2*np.pi*out["doy"]/365.0)

    times = out["timestamp"].values
    lats  = out["lat_kalman"].values
    lons  = out["lon_kalman"].values

    if sst is not None:
        ds, var = sst
        da = ds[var]
        dlat, dlon = grid_res_deg(ds)
        out["sst"] = _interp_linear_then_nearest(da, times, lats, lons)
        dfdx, dfdy = central_gradient(da, times, lats, lons, dlat, dlon)
        out["sst_grad_mag"] = np.sqrt(dfdx**2 + dfdy**2)
        clim = da.groupby("time.month").mean()
        anom = da.groupby("time.month") - clim
        out["sst_anom"] = _interp_linear_then_nearest(anom, times, lats, lons)

    if chl is not None:
        ds, var = chl
        da = ds[var]
        dlat, dlon = grid_res_deg(ds)
        chl_val = _interp_linear_then_nearest(da, times, lats, lons)
        mask = np.isfinite(chl_val) & (chl_val > 0) & (chl_val < 1e3)
        chl_val = np.where(mask, chl_val, np.nan)
        out["chl"] = chl_val
        out["chl_log10"] = np.log10(np.where(np.isfinite(chl_val) & (chl_val > 0), chl_val, np.nan) + 1e-12)
        dfdx, dfdy = central_gradient(da, times, lats, lons, dlat, dlon)
        out["chl_grad_mag"] = np.sqrt(dfdx**2 + dfdy**2)
        clim = da.groupby("time.month").mean()
        anom = da.groupby("time.month") - clim
        out["chl_anom"] = _interp_linear_then_nearest(anom, times, lats, lons)

    have_geo = False
    if ssh is not None:
        ds, var = ssh
        da = ds[var]
        dlat, dlon = grid_res_deg(ds)
        out["ssh"] = _interp_linear_then_nearest(da, times, lats, lons)
        dfdx, dfdy = central_gradient(da, times, lats, lons, dlat, dlon)
        out["ssh_grad_mag"] = np.sqrt(dfdx**2 + dfdy**2)
        ug, vg = geostrophic_from_ssh(dfdx, dfdy, lats)
        out["u_geo"] = ug; out["v_geo"] = vg
        have_geo = True
        ug_s = pd.Series(ug).rolling(60, min_periods=5, center=True).mean()
        vg_s = pd.Series(vg).rolling(60, min_periods=5, center=True).mean()
        ug_an = ug - ug_s.bfill().ffill().values
        vg_an = vg - vg_s.bfill().ffill().values
        out["eke"] = 0.5*(ug_an**2 + vg_an**2)
        clim = da.groupby("time.month").mean()
        anom = da.groupby("time.month") - clim
        out["ssh_anom"] = _interp_linear_then_nearest(anom, times, lats, lons)

    dt_hours = pd.Timedelta(CONFIG.resample_rule).total_seconds()/3600.0
    for col in ["sst_anom", "chl_log10", "ssh_anom"]:
        if col in out.columns:
            low, high, ratio = rolling_fft_features(out[col].astype(float).interpolate().bfill(),
                                                     CONFIG.fft_window, dt_hours)
            out[f"{col}_fft_low"] = low
            out[f"{col}_fft_high"] = high
            out[f"{col}_fft_ratio"] = ratio

    dlon_deg = lon_diff_min_deg(out["lon_kalman"])
    dlat_deg = out["lat_kalman"].diff()
    dt_s = out["timestamp"].diff().dt.total_seconds()
    lat_mid = (out["lat_kalman"] + out["lat_kalman"].shift(1))/2.0
    dx_m, dy_m = earth_dx_dy_m(lat_mid.values, 1.0, 1.0)
    u_abs_mps = (dlon_deg * dx_m) / (dt_s + 1e-6)
    v_abs_mps = (dlat_deg * dy_m) / (dt_s + 1e-6)
    out["u_abs_mps"] = u_abs_mps.fillna(0.0).values
    out["v_abs_mps"] = v_abs_mps.fillna(0.0).values
    out["speed_abs_mps"] = np.sqrt(out["u_abs_mps"]**2 + out["v_abs_mps"]**2)

    if have_geo:
        out["u_rel_mps"] = out["u_abs_mps"] - out["u_geo"]
        out["v_rel_mps"] = out["v_abs_mps"] - out["v_geo"]
        out["speed_rel_mps"] = np.sqrt(out["u_rel_mps"]**2 + out["v_rel_mps"]**2)
        sp = np.sqrt(out["u_geo"]**2 + out["v_geo"]**2) + 1e-9
        sa = out["speed_abs_mps"] + 1e-9
        out["cos_align"] = (out["u_geo"]*out["u_abs_mps"] + out["v_geo"]*out["v_abs_mps"]) / (sp*sa)

    out["foraging_label"] = compute_foraging_proxy(out)
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out

def fit_hotspots(df_train: pd.DataFrame, eps_km: float, min_samples: int):
    if df_train.empty:
        raise ValueError("TRAIN is empty: cannot estimate hotspots.")
    coords_rad = np.radians(np.c_[df_train["lat_kalman"].values, df_train["lon_kalman"].values])
    db = DBSCAN(eps=eps_km/6371.0, min_samples=min_samples, metric="haversine")
    labels = db.fit_predict(coords_rad)
    cluster_ids = sorted([c for c in np.unique(labels) if c != -1])
    if len(cluster_ids) == 0:
        raise RuntimeError("DBSCAN found no hotspots (only noise).")

    centroids = []; radii_km = []; summary_rows = []
    for idx, cid in enumerate(cluster_ids, start=1):
        sub = df_train[labels == cid]
        lat_arr = sub["lat_kalman"].values; lon_arr = sub["lon_kalman"].values
        lat_mean = float(np.mean(lat_arr)); lon_mean = float(np.mean(lon_arr))
        d2mean = haversine_km(lat_arr, lon_arr, lat_mean, lon_mean)
        i_medoid = int(np.nanargmin(d2mean)) if np.isfinite(d2mean).all() else 0
        lat_c = float(lat_arr[i_medoid]); lon_c = float(lon_arr[i_medoid])
        d_km = haversine_km(lat_arr, lon_arr, lat_c, lon_c)
        r_q = float(np.nanpercentile(d_km, CONFIG.radius_quantile))
        rad = max(1.05*eps_km, min(r_q, CONFIG.max_radius_km))
        centroids.append([lat_c, lon_c]); radii_km.append(rad)
        summary_rows.append({"zone_idx": idx, "dbscan_id": cid,
                             "lat_center": lat_c, "lon_center": lon_c,
                             "radius_km": rad, "count": len(sub)})
    centroids = np.array(centroids, dtype=float)
    radii_km  = np.array(radii_km, dtype=float)
    zones_summary = pd.DataFrame(summary_rows)
    CONFIG.results_dir.mkdir(exist_ok=True)
    zones_summary.to_csv(CONFIG.results_dir / "zones_summary.csv", index=False)
    return {"centroids": centroids, "radii_km": radii_km, "zones_summary": zones_summary}

def assign_zones(df: pd.DataFrame, centroids: np.ndarray, radii_km: np.ndarray, unknown_class=0) -> np.ndarray:
    if df.empty:
        return np.array([], dtype=int)
    lat = df["lat_kalman"].values; lon = df["lon_kalman"].values
    K = centroids.shape[0]
    d_all = [haversine_km(lat, lon, centroids[j,0], centroids[j,1]) for j in range(K)]
    D = np.vstack(d_all).T
    jmin = np.argmin(D, axis=1); dmin = D[np.arange(len(lat)), jmin]
    assigned = np.where(dmin <= radii_km[jmin], jmin + 1, unknown_class)
    return assigned.astype(int)

def color_palette_for_ids(ids: List[str], max_colors: int = 60):
    base = plt.get_cmap("tab20").colors
    if max_colors <= len(base):
        cols = base[:max_colors]
    else:
        extra = max_colors - len(base)
        hsv = plt.get_cmap("hsv")(np.linspace(0, 1, extra+1))[:-1, :3]
        cols = np.vstack([base, hsv])
    id_sorted = sorted(list(ids))
    return {iid: plt.matplotlib.colors.to_hex(cols[i % len(cols)]) for i, iid in enumerate(id_sorted)}

def _geodesic_circle_points(lat0, lon0, radius_km, n=200):
    R = 6371.0; phi1 = np.radians(lat0); lam1 = np.radians(lon0); delta = radius_km/R
    bearings = np.linspace(0.0, 2*np.pi, n)
    sin_phi1, cos_phi1 = np.sin(phi1), np.cos(phi1)
    sin_delta, cos_delta = np.sin(delta), np.cos(delta)
    sin_phi2 = sin_phi1*cos_delta + cos_phi1*sin_delta*np.cos(bearings)
    phi2 = np.arcsin(np.clip(sin_phi2, -1, 1))
    y = np.sin(bearings) * sin_delta * cos_phi1
    x = cos_delta - sin_phi1 * sin_phi2
    lam2 = lam1 + np.arctan2(y, x); lam2 = (lam2 + np.pi) % (2*np.pi) - np.pi
    lats = np.degrees(phi2); lons = np.degrees(lam2)
    return lats, lons

def _try_cartopy_ax(extent):
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        proj_data = ccrs.PlateCarree(); proj_map = ccrs.Mercator()
        fig = plt.figure(figsize=(13,10)); ax = plt.axes(projection=proj_map)
        ax.set_extent(extent, crs=proj_data)
        ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor="#DCEAF7")
        ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor="#F5F5F5")
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.7)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.5, linestyle=":")
        ax.add_feature(cfeature.LAKES.with_scale('50m'), alpha=0.3)
        return fig, ax, proj_data
    except Exception:
        return None, None, None

def _extent_from_points(lats, lons):
    lat_min, lat_max = float(np.nanmin(lats)), float(np.nanmax(lats))
    lon_min, lon_max = float(np.nanmin(lons)), float(np.nanmax(lons))
    lat_pad = max(2.0, 0.05*(lat_max - lat_min + 1e-3))
    lon_pad = max(2.0, 0.05*(lon_max - lon_min + 1e-3))
    return [lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad]

def plot_map_hotspots_only(centroids, radii_km, all_points_df, save_path):
    if all_points_df.empty:
        return
    lats_all = all_points_df["lat_kalman"].values
    lons_all = _to_180(all_points_df["lon_kalman"].values)
    extent = _extent_from_points(lats_all, lons_all)

    fig, ax, crs = _try_cartopy_ax(extent)
    if ax is None:
        fig, ax = plt.subplots(figsize=(12,8))
        crs = None

    K = centroids.shape[0]
    for j in range(K):
        latc, lonc = centroids[j,0], _to_180(centroids[j,1])
        if crs is None: ax.plot(lonc, latc, marker="*", ms=12, zorder=5)
        else: ax.plot(lonc, latc, marker="*", ms=12, transform=crs, zorder=5)
        latcirl, loncirl = _geodesic_circle_points(latc, lonc, radii_km[j], n=240)
        if crs is None: ax.plot(loncirl, latcirl, ls="--", lw=1.6, alpha=0.8, zorder=3)
        else: ax.plot(loncirl, latcirl, ls="--", lw=1.6, alpha=0.8, transform=crs, zorder=3)
        ax.text(lonc, latc, f" HS{j+1}", fontsize=11,
                va="center", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.5", alpha=0.8),
                transform=crs if crs else None, zorder=6)

    ax.set_title("Hotspots (centroids and radii only)")
    plt.tight_layout()
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path, dpi=220); plt.close(fig)

def _plot_segments(ax, lons, lats, crs, color, lw=1.1, alpha=0.9):
    if crs is None:
        ax.plot(lons, lats, lw=lw, alpha=alpha, color=color, solid_capstyle="round")
    else:
        ax.plot(lons, lats, lw=lw, alpha=alpha, color=color, transform=crs, solid_capstyle="round")

def plot_map_individuals(all_splits_df: pd.DataFrame, save_path: str, centroids=None, radii_km=None):
    if all_splits_df.empty:
        return
    df = all_splits_df.sort_values(["id","timestamp"]).copy()
    lats_all = df["lat_kalman"].values
    lons_all = _to_180(df["lon_kalman"].values)
    extent = _extent_from_points(lats_all, lons_all)

    fig, ax, crs = _try_cartopy_ax(extent)
    if ax is None:
        fig, ax = plt.subplots(figsize=(13,10))
        crs = None

    ids = df["id"].unique().tolist()
    palette = color_palette_for_ids(ids, max_colors=max(60, len(ids)))
    pd.DataFrame([{"id": iid, "color": palette[iid]} for iid in ids]).to_csv(
        CONFIG.results_dir / "id_color_map.csv", index=False)

    gap = pd.Timedelta(hours=CONFIG.map_gap_hours)
    step = max(1, int(CONFIG.map_downsample_step))
    labels_left = CONFIG.map_max_labeled_ids

    for iid, g in df.groupby("id", sort=False):
        g = g.iloc[::step].copy()
        g["lon180"] = _to_180(g["lon_kalman"].values)
        g["dt"] = g["timestamp"].diff()
        split_idx = np.where((g["dt"] > gap) | (g["dt"].isna()))[0]
        idx = [0] + split_idx.tolist() + [len(g)]
        color = palette[iid]
        for i in range(len(idx)-1):
            seg = g.iloc[idx[i]:idx[i+1]]
            if len(seg) < 2: continue
            _plot_segments(ax, seg["lon180"].values, seg["lat_kalman"].values, crs, color, lw=1.3, alpha=0.85)
        first = g.iloc[0]; last = g.iloc[-1]
        if crs is None:
            ax.plot(first["lon180"], first["lat_kalman"], marker="^", ms=6, color=color)
            ax.plot(last["lon180"],  last["lat_kalman"],  marker="s", ms=5, color=color)
        else:
            ax.plot(first["lon180"], first["lat_kalman"], marker="^", ms=6, color=color, transform=crs)
            ax.plot(last["lon180"],  last["lat_kalman"],  marker="s", ms=5, color=color, transform=crs)
        if labels_left > 0:
            ax.text(first["lon180"], first["lat_kalman"], f" {iid}",
                    fontsize=8, color="black",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.5", alpha=0.7),
                    transform=crs if crs else None)
            labels_left -= 1

    if centroids is not None and radii_km is not None:
        for j in range(centroids.shape[0]):
            latc, lonc = centroids[j,0], _to_180(centroids[j,1])
            if crs is None: ax.plot(lonc, latc, marker="*", ms=11, color="#333333", zorder=5)
            else: ax.plot(lonc, latc, marker="*", ms=11, color="#333333", transform=crs, zorder=5)
            ax.text(lonc, latc, f" HS{j+1}",
                    fontsize=9, va="center", ha="left",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.5", alpha=0.8),
                    transform=crs if crs else None, zorder=6)

    ax.set_title("Trajectories by individual (distinct colors)\n▲ start | ■ end")
    plt.tight_layout()
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path, dpi=220); plt.close(fig)

def plot_map_zones(cleaned_dict, centroids, radii_km, save_path="map_zones.png"):
    frames = []
    for k in ["train","val","test"]:
        if k in cleaned_dict and not cleaned_dict[k].empty:
            frames.append(cleaned_dict[k][["lat_kalman","lon_kalman","timestamp"]].copy())
    if not frames:
        return
    all_pts = pd.concat(frames, axis=0)
    lats_all = all_pts["lat_kalman"].values
    lons_all = _to_180(all_pts["lon_kalman"].values)
    extent = _extent_from_points(lats_all, lons_all)

    fig, ax, crs = _try_cartopy_ax(extent)
    if ax is None:
        fig, ax = plt.subplots(figsize=(12,8))
        crs = None

    if CONFIG.map_plot_tracks:
        if "train" in cleaned_dict and not cleaned_dict["train"].empty:
            pts = cleaned_dict["train"]; lon = _to_180(pts["lon_kalman"].values); lat = pts["lat_kalman"].values
            if crs is None: ax.scatter(lon, lat, s=5, alpha=0.25, label="TRAIN")
            else: ax.scatter(lon, lat, s=5, alpha=0.25, label="TRAIN", transform=crs)
        if "val" in cleaned_dict and not cleaned_dict["val"].empty:
            pts = cleaned_dict["val"]; lon = _to_180(pts["lon_kalman"].values); lat = pts["lat_kalman"].values
            if crs is None: ax.scatter(lon, lat, s=5, alpha=0.25, label="VAL")
            else: ax.scatter(lon, lat, s=5, alpha=0.25, label="VAL", transform=crs)
        if "test" in cleaned_dict and not cleaned_dict["test"].empty:
            pts = cleaned_dict["test"]; lon = _to_180(pts["lon_kalman"].values); lat = pts["lat_kalman"].values
            if crs is None: ax.scatter(lon, lat, s=5, alpha=0.25, label="TEST")
            else: ax.scatter(lon, lat, s=5, alpha=0.25, label="TEST", transform=crs)

    K = centroids.shape[0]
    for j in range(K):
        latc, lonc = centroids[j,0], _to_180(centroids[j,1])
        if crs is None: ax.plot(lonc, latc, marker="*", ms=13, color="#111111", zorder=5)
        else: ax.plot(lonc, latc, marker="*", ms=13, color="#111111", transform=crs, zorder=5)
        latcirl, loncirl = _geodesic_circle_points(latc, lonc, radii_km[j], n=240)
        if crs is None: ax.plot(loncirl, latcirl, ls="--", lw=1.8, alpha=0.9, zorder=4)
        else: ax.plot(loncirl, latcirl, ls="--", lw=1.8, alpha=0.9, transform=crs, zorder=4)
        ax.text(lonc, latc, f" HS{j+1}", fontsize=11,
                va="center", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.5", alpha=0.85),
                transform=crs if crs else None, zorder=6)

    ax.set_title("Hotspot centroids and radii (faded trajectories)")
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    plt.tight_layout()
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path, dpi=220); plt.close(fig)

def choose_id_col(df, prefs):
    for c in prefs:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find any ID column among {prefs}")

def compute_valid_window_starts(df_sorted: pd.DataFrame, id_col: str, window: int) -> np.ndarray:
    starts = []
    if window <= 0 or df_sorted.empty:
        return np.array([], dtype=int)
    for _, g in df_sorted.groupby(id_col, sort=False):
        n = len(g)
        if n <= window:
            continue
        base = g.index.min()
        last_start_rel = n - (window + 1)
        if last_start_rel < 0:
            continue
        starts.extend(range(base, base + last_start_rel + 1))
    return np.array(starts, dtype=int)

class TrajectoryDatasetHybridWindows(Dataset):
    def __init__(self, X, y_pos, y_zone, window, idx_lon, idx_lat, x_min, x_scale, starts, df_ref: pd.DataFrame):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_pos = torch.tensor(y_pos, dtype=torch.float32)
        self.y_zone = torch.tensor(y_zone, dtype=torch.long)
        self.W = window
        self.idx_lon = idx_lon; self.idx_lat = idx_lat
        self.x_min = torch.tensor(x_min, dtype=torch.float32)
        self.x_scale = torch.tensor(x_scale, dtype=torch.float32)
        self.starts = np.asarray(starts, dtype=int)
        self.df_ref = df_ref

    def __len__(self): return len(self.starts)

    def __getitem__(self, k):
        i0 = int(self.starts[k])
        end_t = i0 + self.W
        assert end_t < len(self.y_pos), f"Invalid window: i0={i0}, W={self.W}, len={len(self.y_pos)}"
        seq = self.X[i0:end_t]
        target_pos = self.y_pos[end_t]
        target_zone = self.y_zone[end_t]
        last_scaled = torch.stack([seq[-1, self.idx_lon], seq[-1, self.idx_lat]])
        last_unscaled = (last_scaled - self.x_min) / self.x_scale
        target_index = torch.tensor(end_t, dtype=torch.long)
        return seq, target_pos, target_zone, last_unscaled, target_index

class HybridPredictorGRU(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        cfg = CONFIG
        self.gru = nn.GRU(
            input_size=num_features, hidden_size=cfg.gru_hidden_size,
            num_layers=cfg.gru_num_layers, batch_first=True,
            dropout=cfg.gru_dropout if cfg.gru_num_layers > 1 else 0.0,
            bidirectional=cfg.gru_bidirectional
        )
        out_dim = cfg.gru_hidden_size * (2 if cfg.gru_bidirectional else 1)
        self.post_gru_dropout = nn.Dropout(cfg.gru_dropout)
        self.delta_head = nn.Sequential(nn.Linear(out_dim, 64), nn.ReLU(), nn.Linear(64, 2))
        self.zone_head  = nn.Sequential(nn.Linear(out_dim, 128), nn.ReLU(), nn.Linear(128, num_classes))

    def forward(self, x):
        y, _ = self.gru(x)
        h = y[:, -1, :]
        h = self.post_gru_dropout(h)
        delta  = self.delta_head(h)
        logits = self.zone_head(h)
        return delta, logits

def _infer_fs_from_rule(rule: str) -> float:
    dt_h = pd.Timedelta(rule).total_seconds()/3600.0
    fs_h = 1.0 / max(dt_h, 1e-6)
    return fs_h * 24.0

def _add_bins(df: pd.DataFrame, spatial_res_deg=0.25, temporal_rule="1D"):
    d = df.copy()
    d["lon180"] = _to_180(d["lon_kalman"].values)
    lat_min = np.floor(d["lat_kalman"].min()); lat_max = np.ceil(d["lat_kalman"].max())
    lon_min = np.floor(d["lon180"].min());   lon_max = np.ceil(d["lon180"].max())
    lat_bins = np.arange(lat_min, lat_max + spatial_res_deg, spatial_res_deg)
    lon_bins = np.arange(lon_min, lon_max + spatial_res_deg, spatial_res_deg)
    d["lat_bin"] = pd.cut(d["lat_kalman"], bins=lat_bins, labels=lat_bins[:-1])
    d["lon_bin"] = pd.cut(d["lon180"],     bins=lon_bins, labels=lon_bins[:-1])
    d["time_bin"] = d["timestamp"].dt.floor(temporal_rule)
    d = d.dropna(subset=["lat_bin","lon_bin","time_bin"])
    d["lat_bin"] = d["lat_bin"].astype(float); d["lon_bin"] = d["lon_bin"].astype(float)
    return d

def compute_time_domain_correlation_df(df: pd.DataFrame, x_col: str, y_col: str,
                                       spatial_res_deg=0.25, temporal_rule="1D",
                                       min_observations=5, max_lag_steps=5):
    d = _add_bins(df, spatial_res_deg, temporal_rule)
    g = d.groupby(["time_bin","lat_bin","lon_bin"], observed=False).agg(x=(x_col,"mean"), y=(y_col,"mean")).reset_index()
    rows = []
    for (latb, lonb), sub in g.groupby(["lat_bin","lon_bin"], observed=False):
        sub = sub.sort_values("time_bin").dropna(subset=["x","y"])
        if len(sub) < min_observations: continue
        if sub["x"].std() == 0 or sub["y"].std() == 0: continue
        pr, _ = stats.pearsonr(sub["x"], sub["y"])
        sr, _ = stats.spearmanr(sub["x"], sub["y"])
        lags = range(-max_lag_steps, max_lag_steps + 1)
        lag_corrs = []
        for L in lags:
            if L < 0:  x = sub["x"].values[:L]; y = sub["y"].values[-L:]
            elif L > 0: x = sub["x"].values[L:]; y = sub["y"].values[:-L]
            else:       x = sub["x"].values;   y = sub["y"].values
            if len(x) >= min_observations and np.std(x) > 0 and np.std(y) > 0:
                lag_corrs.append(np.corrcoef(x,y)[0,1])
            else:
                lag_corrs.append(np.nan)
        if np.any(np.isfinite(lag_corrs)):
            i_best = int(np.nanargmax(np.abs(lag_corrs))); best_lag = list(lags)[i_best]; best_lag_corr = lag_corrs[i_best]
        else:
            best_lag, best_lag_corr = 0, np.nan
        rows.append({"lat": latb, "lon": lonb, "n_obs": len(sub),
                     "pearson_r": pr, "spearman_r": sr,
                     "best_lag_steps": best_lag, "best_lag_corr": best_lag_corr})
    return pd.DataFrame(rows)

def compute_frequency_domain_coherence_df(df: pd.DataFrame, x_col: str, y_col: str,
                                          spatial_res_deg=0.25, temporal_rule="1D",
                                          min_observations=12, nperseg=None,
                                          detrend="linear", window="hann"):
    d = _add_bins(df, spatial_res_deg, temporal_rule)
    g = d.groupby(["time_bin","lat_bin","lon_bin"], observed=False).agg(x=(x_col,"mean"), y=(y_col,"mean")).reset_index()
    fs_cpd = _infer_fs_from_rule(temporal_rule)
    rows = []
    for (latb, lonb), sub in g.groupby(["lat_bin","lon_bin"], observed=False):
        sub = sub.sort_values("time_bin").dropna(subset=["x","y"])
        if len(sub) < min_observations: continue
        if sub["x"].std() == 0 or sub["y"].std() == 0: continue
        x = sub["x"].values; y = sub["y"].values
        if detrend:
            x = signal.detrend(x, type=detrend); y = signal.detrend(y, type=detrend)
        nps = max(8, min(len(x)//2, 64)) if nperseg is None else min(nperseg, len(x))
        f, Cxy = signal.coherence(x, y, fs=fs_cpd, nperseg=nps, window=window)
        if len(f) <= 1: continue
        f_no0, coh_no0 = f[1:], Cxy[1:]
        mean_coh = float(np.nanmean(coh_no0)) if len(coh_no0) else 0.0
        dom_idx = int(np.nanargmax(coh_no0)) if np.any(np.isfinite(coh_no0)) else 0
        dom_freq = float(f_no0[dom_idx]) if len(f_no0) else np.nan
        dom_period_days = float(1.0/dom_freq) if (dom_freq and dom_freq > 0) else np.nan
        rows.append({"lat": latb, "lon": lonb, "n_obs": len(sub),
                     "mean_coherence": mean_coh, "dominant_period_days": dom_period_days})
    return pd.DataFrame(rows)

def _quick_corr_plots(time_df: pd.DataFrame, freq_df: pd.DataFrame, out_path: str):
    Path(out_path).parent.mkdir(exist_ok=True, parents=True)
    fig, axes = plt.subplots(2,2, figsize=(13,10))
    ax = axes[0,0]
    if time_df is not None and len(time_df):
        sc = ax.scatter(time_df["lon"], time_df["lat"], c=time_df["pearson_r"], s=60, cmap="RdBu_r", vmin=-1, vmax=1, edgecolors="k")
        ax.set_title("Pearson's r by cell"); ax.set_xlabel("Lon"); ax.set_ylabel("Lat")
        plt.colorbar(sc, ax=ax, label="r")
    else:
        ax.text(0.5,0.5,"No time-domain correlations", ha="center"); ax.axis("off")
    ax = axes[0,1]
    if time_df is not None and len(time_df):
        ax.hist(time_df["pearson_r"].dropna(), bins=20, edgecolor="k", alpha=0.8); ax.axvline(0, color="r", ls="--")
        ax.set_title("Pearson's r Histogram")
    else:
        ax.text(0.5,0.5,"No time-domain correlations", ha="center"); ax.axis("off")
    ax = axes[1,0]
    if freq_df is not None and len(freq_df):
        sc = ax.scatter(freq_df["lon"], freq_df["lat"], c=freq_df["mean_coherence"], s=60, cmap="viridis", vmin=0, vmax=1, edgecolors="k")
        ax.set_title("Mean Coherence"); ax.set_xlabel("Lon"); ax.set_ylabel("Lat")
        plt.colorbar(sc, ax=ax, label="coherence")
    else:
        ax.text(0.5,0.5,"No frequency-domain coherence", ha="center"); ax.axis("off")
    ax = axes[1,1]
    if freq_df is not None and len(freq_df):
        ax.scatter(freq_df["dominant_period_days"], freq_df["mean_coherence"], s=60, alpha=0.85, edgecolors="k")
        ax.set_xlabel("Dominant period (days)"); ax.set_ylabel("Mean Coherence"); ax.grid(True, alpha=0.3)
        ax.set_title("Coherence vs dominant period")
    else:
        ax.text(0.5,0.5,"No frequency-domain coherence", ha="center"); ax.axis("off")
    plt.tight_layout(); fig.savefig(out_path, dpi=180, bbox_inches="tight"); plt.close(fig)

class EcoQnA:
    def __init__(self, df_all: pd.DataFrame, sst, chl, ssh, outdir="results_qna"):
        self.df = df_all.copy(); self.sst = sst; self.chl = chl; self.ssh = ssh
        self.outdir = Path(outdir); self.outdir.mkdir(exist_ok=True)

    def _savefig(self, fig, name):
        fp = self.outdir / name; fig.savefig(fp, dpi=220, bbox_inches="tight"); plt.close(fig); return fp

    def _used_vs_available(self, value_col: str, roi_pad_deg: float = 0.5, samples_per_time: int = 3, max_samples: int = 6000):
        d = self.df.dropna(subset=["lat_kalman","lon_kalman","timestamp",value_col]).copy()
        if d.empty: return d.assign(set_type="used"), d.assign(set_type="available").iloc[:0]
        rng = np.random.default_rng(CONFIG.seed); rows = []
        for _, r in d.iterrows():
            n = samples_per_time
            lat_s = rng.uniform(r["lat_kalman"]-roi_pad_deg, r["lat_kalman"]+roi_pad_deg, size=n)
            lon_s = rng.uniform(r["lon_kalman"]-roi_pad_deg, r["lon_kalman"]+roi_pad_deg, size=n)
            t_s = np.repeat(r["timestamp"], n)
            rows.append(pd.DataFrame({"timestamp": t_s, "lat": lat_s, "lon": lon_s}))
        avail = pd.concat(rows, ignore_index=True)

        if value_col.startswith("sst"):
            ds, var = self.sst
            da = ds[var]
            if "anom" in value_col:
                clim = da.groupby("time.month").mean()
                da = da.groupby("time.month") - clim
            v = _interp_linear_then_nearest(da, avail["timestamp"].values, avail["lat"].values, avail["lon"].values)
            avail[value_col] = v

        elif value_col.startswith("chl"):
            ds, var = self.chl
            da = ds[var]
            v = _interp_linear_then_nearest(da, avail["timestamp"].values, avail["lat"].values, avail["lon"].values)
            v = np.where(np.isfinite(v) & (v > 0) & (v < 1e3), v, np.nan)
            if "log10" in value_col:
                v = np.log10(v + 1e-12)
            avail[value_col] = v

        elif value_col.startswith("ssh"):
            ds, var = self.ssh
            da = ds[var]
            v = _interp_linear_then_nearest(da, avail["timestamp"].values, avail["lat"].values, avail["lon"].values)
            avail[value_col] = v
        else:
            return d.assign(set_type="used"), d.assign(set_type="available").iloc[:0]

        avail = avail.replace([np.inf,-np.inf], np.nan).dropna(subset=[value_col])
        if len(avail) > max_samples:
            avail = avail.sample(max_samples, random_state=CONFIG.seed)

        used = d[["timestamp","lat_kalman","lon_kalman",value_col]].rename(columns={"lat_kalman":"lat","lon_kalman":"lon"})
        used["set_type"] = "used"; avail["set_type"] = "available"
        return used[["timestamp","lat","lon",value_col,"set_type"]], avail[["timestamp","lat","lon",value_col,"set_type"]]

    def Q1_alignment_vs_eke(self):
        if "cos_align" not in self.df.columns or "eke" not in self.df.columns:
            return {"answer":"Q1: u_geo/v_geo/EKE/cos_align not available."}
        d = self.df.dropna(subset=["cos_align","eke"]).copy()
        q = np.nanquantile(d["eke"], [0.25,0.75]); lo, hi = q[0], q[1]
        d["EKE_bin"] = pd.cut(d["eke"], bins=[-np.inf, lo, hi, np.inf], labels=["low","medium","high"])
        g = d.groupby("EKE_bin", observed=False)["cos_align"].describe()[["mean","std","25%","50%","75%","count"]]
        fig, ax = plt.subplots(figsize=(7.2,4.6))
        d.boxplot(column="cos_align", by="EKE_bin", ax=ax)
        ax.set_title("Alignment cos(θ) vs EKE"); ax.set_ylabel("cos(θ)"); ax.figure.suptitle("")
        fp = self._savefig(fig, "Q1_alignment_vs_EKE.png")
        g.to_csv(self.outdir / "Q1_alignment_stats.csv")
        return {"answer": f"Q1: See boxplot; summary:\n{g}", "figures":[fp],
                "tables":[self.outdir / "Q1_alignment_stats.csv"]}

    def Q2_sst_used_available(self):
        if self.sst is None or ("sst" not in self.df.columns and "sst_anom" not in self.df.columns):
            return {"answer":"Q2: SST not available."}
        col = "sst_anom" if "sst_anom" in self.df.columns else "sst"
        used, avail = self._used_vs_available(col)
        if avail.empty or used.empty: return {"answer":"Q2: insufficient data for comparison."}
        u = used[col]; a = avail[col]
        stat, p = stats.ks_2samp(u, a)
        cliffs = (np.mean(u.values.reshape(-1,1) > a.values.reshape(1,-1)) - np.mean(u.values.reshape(-1,1) < a.values.reshape(1,-1)))
        fig, ax = plt.subplots(figsize=(7.2,4.6))
        ax.hist(a, bins=40, alpha=0.5, label="available"); ax.hist(u, bins=40, alpha=0.6, label="used")
        ax.set_title(f"Used vs Available — {'SST (anom)' if col=='sst_anom' else 'SST'}"); ax.legend(); ax.set_xlabel("SST"); ax.grid(alpha=0.3)
        fp = self._savefig(fig, "Q2_sst_used_available.png")
        pd.DataFrame({"stat":[stat], "p_value":[p], "cliffs_delta":[cliffs]}).to_csv(self.outdir / "Q2_sst_stats.csv", index=False)
        return {"answer": f"Q2: KS p={p:.3g}, Cliff’s Δ={cliffs:.2f}.", "figures":[fp], "tables":[self.outdir / "Q2_sst_stats.csv"]}

    def Q3_chl_used_available(self):
        if self.chl is None or "chl_log10" not in self.df.columns:
            return {"answer":"Q3: Chlorophyll not available."}
        used, avail = self._used_vs_available("chl_log10")
        if avail.empty or used.empty: return {"answer":"Q3: insufficient data for comparison."}
        u = used["chl_log10"]; a = avail["chl_log10"]
        stat, p = stats.ks_2samp(u, a)
        cliffs = (np.mean(u.values.reshape(-1,1) > a.values.reshape(1,-1)) - np.mean(u.values.reshape(-1,1) < a.values.reshape(1,-1)))
        fig, ax = plt.subplots(figsize=(7.2,4.6))
        ax.hist(a, bins=40, alpha=0.5, label="available"); ax.hist(u, bins=40, alpha=0.6, label="used")
        ax.set_title("Used vs Available — log10(CHL)"); ax.legend(); ax.set_xlabel("log10(CHL)")
        fp = self._savefig(fig, "Q3_chl_used_available.png")
        pd.DataFrame({"stat":[stat], "p_value":[p], "cliffs_delta":[cliffs]}).to_csv(self.outdir / "Q3_chl_stats.csv", index=False)
        return {"answer": f"Q3: KS p={p:.3g}, Cliff’s Δ={cliffs:.2f}.", "figures":[fp],
                "tables":[self.outdir / "Q3_chl_stats.csv"]}

    def Q4_lag_sst_behavior(self, x_col=None, y_col="sst_anom", max_lag_steps=10):
        x_col = x_col or ("speed_rel_mps" if "speed_rel_mps" in self.df.columns else "speed_abs_mps")
        d = self.df.dropna(subset=[x_col, y_col]).copy()
        if d.empty: return {"answer":"Q4: insufficient data for lag analysis."}
        x = d[x_col].values; y = d[y_col].values
        lags = np.arange(-max_lag_steps, max_lag_steps + 1); cors = []
        for L in lags:
            if L < 0:  xx, yy = x[:L], y[-L:]
            elif L > 0: xx, yy = x[L:], y[:-L]
            else:       xx, yy = x, y
            if len(xx) > 5 and np.std(xx) > 0 and np.std(yy) > 0:
                cors.append(np.corrcoef(xx, yy)[0,1])
            else:
                cors.append(np.nan)
        i = int(np.nanargmax(np.abs(cors))); lag_best = lags[i]; corr_best = cors[i]
        fig, ax = plt.subplots(figsize=(7.2,4.6))
        ax.plot(lags, cors, marker="o"); ax.axhline(0, color="k", lw=0.8); ax.axvline(0, ls="--", color="gray")
        ax.set_title(f"Lag {x_col} vs {y_col} (best={lag_best}, r={corr_best:.2f})")
        ax.set_xlabel(f"lag ({CONFIG.resample_rule} steps)")
        fp = self._savefig(fig, "Q4_lag_sst_behavior.png")
        pd.DataFrame({"lag":lags, "corr":cors}).to_csv(self.outdir / "Q4_lag_curve.csv", index=False)
        return {"answer": f"Q4: best lag={lag_best} (r={corr_best:.2f}).", "figures":[fp],
                "tables":[self.outdir / "Q4_lag_curve.csv"]}

    def Q5_mesoscale_regime(self, thr_m=0.02):
        if "ssh_anom" not in self.df.columns:
            return {"answer":"Q5: ssh_anom not available."}
        d = self.df.dropna(subset=["ssh_anom","speed_abs_mps","foraging_label"]).copy()
        reg = np.where(d["ssh_anom"] > thr_m, "anticyclonic",
              np.where(d["ssh_anom"] < -thr_m, "cyclonic", "neutral"))
        d["regime"] = reg
        g1 = d.groupby("regime", observed=False)["speed_abs_mps"].describe()[["mean","std","count"]]
        g2 = d.groupby("regime", observed=False)["foraging_label"].mean()
        fig, axs = plt.subplots(1,2, figsize=(11,4.6))
        d.boxplot(column="speed_abs_mps", by="regime", ax=axs[0]); axs[0].set_title("Speed by regime"); axs[0].figure.suptitle("")
        axs[1].bar(g2.index, g2.values); axs[1].set_title("ARS rate by regime"); axs[1].set_ylim(0,1)
        fp = self._savefig(fig, "Q5_mesoscale_regime.png")
        g1.to_csv(self.outdir / "Q5_speed_stats.csv"); g2.to_csv(self.outdir / "Q5_ars_rate.csv")
        return {"answer":"Q5: speeds and ARS vary by regime (see figure).",
                "figures":[fp], "tables":[self.outdir / "Q5_speed_stats.csv",
                                          self.outdir / "Q5_ars_rate.csv"]}

    def Q6_seasonal_hotspots(self, eps_km=35.0, min_samples=15):
        out_paths = []
        for q in [1,2,3,4]:
            sub = self.df[self.df["timestamp"].dt.quarter == q][["timestamp","lat_kalman","lon_kalman"]].dropna()
            if len(sub) < min_samples: continue
            coords_rad = np.radians(np.c_[sub["lat_kalman"].values, sub["lon_kalman"].values])
            db = DBSCAN(eps=eps_km/6371.0, min_samples=min_samples, metric="haversine")
            labels = db.fit_predict(coords_rad); sub = sub.assign(lbl=labels)
            cls = sorted([c for c in np.unique(labels) if c != -1])
            if not cls: continue
            cs = []; rs = []
            for cid in cls:
                s = sub[sub["lbl"] == cid]
                latm, lonm = s["lat_kalman"].mean(), s["lon_kalman"].mean()
                d = haversine_km(s["lat_kalman"], s["lon_kalman"], latm, lonm)
                r = np.nanpercentile(d, 80)
                cs.append([latm, lonm]); rs.append(r)
            cs = np.array(cs); rs = np.array(rs)
            fig, ax = plt.subplots(figsize=(7.8,6.2))
            ax.scatter(_to_180(sub["lon_kalman"]), sub["lat_kalman"], s=5, alpha=0.25, label=f"Q{q} points")
            for j in range(len(cs)):
                ax.plot(_to_180(cs[j,1]), cs[j,0], marker="*", ms=12)
                latc, lonc = _geodesic_circle_points(cs[j,0], _to_180(cs[j,1]), rs[j], n=240)
                ax.plot(lonc, latc, ls="--", alpha=0.8); ax.text(_to_180(cs[j,1]), cs[j,0], f" HS{j+1}")
            ax.set_xlabel("Lon"); ax.set_ylabel("Lat"); ax.set_title(f"Seasonal Hotspots Q{q}")
            fp = self._savefig(fig, f"Q6_hotspots_Q{q}.png"); out_paths.append(fp)
            pd.DataFrame({"lat": cs[:,0], "lon": cs[:,1], "radius_km": rs}).to_csv(
                self.outdir / f"Q6_hotspots_Q{q}.csv", index=False)
        if not out_paths: return {"answer":"Q6: No seasonal hotspots were detected."}
        return {"answer":"Q6: seasonal hotspots generated by quarter.", "figures": out_paths}

    def build_report(self, answers: Dict[str, Dict]):
        md = ["# Automatic Report (Ecological Q&A)\n"]
        for k in ["Q1","Q2","Q3","Q4","Q5","Q6"]:
            if k in answers:
                md.append(f"## {k}\n"); md.append(str(answers[k].get("answer","(no text)")) + "\n")
                figs = answers[k].get("figures", []); tabs = answers[k].get("tables", [])
                if figs:
                    for f in figs: md.append(f"![{k}]({Path(f).name})\n")
                if tabs:
                    md.append("Files:\n")
                    for t in tabs: md.append(f"- {Path(t).name}\n")
        txt = "\n".join(md); fp = self.outdir / "report.md"
        with open(fp,"w",encoding="utf-8") as f: f.write(txt)
        logging.info(f"📝 Q&A Report saved to: {fp}")
        return fp

def main():
    usecols = ["timestamp","location-long","location-lat",
               "individual-local-identifier","tag-local-identifier"]
    try:
        raw0 = pd.read_csv(
            CONFIG.movebank_csv,
            usecols=lambda c: (c in usecols),
            parse_dates=["timestamp"],
            dtype={"individual-local-identifier":"string", "tag-local-identifier":"string"},
            engine="python",
            on_bad_lines="skip"
        )
    except FileNotFoundError:
        sys.exit(f"❌ '{CONFIG.movebank_csv}' not found")

    raw0 = raw0.rename(columns={"location-long":"longitude", "location-lat":"latitude"})
    raw0["longitude"] = pd.to_numeric(raw0["longitude"], errors="coerce")
    raw0["latitude"]  = pd.to_numeric(raw0["latitude"],  errors="coerce")

    id_col = None
    for c in CONFIG.id_col_preference:
        if c in raw0.columns:
            id_col = c; break
    if id_col is None:
        sys.exit(f"❌ No ID column found among {CONFIG.id_col_preference}")

    raw0["id"] = raw0[id_col].astype("string").str.strip()
    raw0 = raw0.dropna(subset=["timestamp","longitude","latitude","id"]).copy()
    raw0 = raw0.sort_values(["id","timestamp"]).drop_duplicates(subset=["id","timestamp","longitude","latitude"]).reset_index(drop=True)

    counts = raw0["id"].value_counts()
    keep_ids = counts[counts >= CONFIG.min_points_per_individual].index
    raw0 = raw0[raw0["id"].isin(keep_ids)].reset_index(drop=True)
    logging.info(f"🐋 Individuals detected: {raw0['id'].nunique()} → {sorted(raw0['id'].unique().tolist())}")

    bbox = {
        "lat_min": raw0.latitude.min() - CONFIG.region_margin_deg,
        "lat_max": raw0.latitude.max() + CONFIG.region_margin_deg,
        "lon_min": float(to_360(raw0.longitude.min() - CONFIG.region_margin_deg)),
        "lon_max": float(to_360(raw0.longitude.max() + CONFIG.region_margin_deg))
    }

    logging.info("🌊 Loading satellite datasets (Gulf clip + margin)…")
    sst_ds = sst_var = ssh_ds = ssh_var = chl_ds = chl_var = None

    sst_ds, sst_var, sst_err = open_satellite_safe(
        [str(Path(CONFIG.data_dir) / p) for p in CONFIG.sst_patterns], 
        ["sst","SST","analysed_sst","sea_surface_temperature"], bbox, label="SST")
    if sst_err: logging.warning(f"  ⚠️ SST not available ({sst_err})")
    else:       logging.info(f"  ✓ SST: var='{sst_var}'")

    chl_ds, chl_var, chl_err = open_modis_chl_l3m(str(Path(CONFIG.data_dir) / CONFIG.chl_pattern), bbox)
    if chl_err: logging.warning(f"  ⚠️ Chlorophyll not available ({chl_err})")
    else:
        logging.info(f"  ✓ Chlorophyll: var='{chl_var}', "
              f"{pd.to_datetime(chl_ds.time.values[0]).date()}…{pd.to_datetime(chl_ds.time.values[-1]).date()}")

    ssh_ds, ssh_var, ssh_err = open_satellite_safe(
        [str(Path(CONFIG.data_dir) / p) for p in CONFIG.ssh_patterns], 
        ["adt","ssh","sla","sea_surface_height","absolute_dynamic_topography"], bbox, label="Altimetry")
    if ssh_err: logging.warning(f"  ⚠️ Altimetry not available ({ssh_err})")
    else:       logging.info(f"  ✓ Altimetry: var='{ssh_var}'")

    train_ratio = CONFIG.split_ratios["train"]; val_ratio = CONFIG.split_ratios["val"]
    def split_by_individual(df, train_ratio=0.7, val_ratio=0.15, seed=CONFIG.seed):
        groups = df["id"].values
        test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)
        gss1 = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
        idx_all = np.arange(len(df))
        trval_idx, te_idx = next(gss1.split(idx_all, groups=groups))
        df_trval, df_test = df.iloc[trval_idx].copy(), df.iloc[te_idx].copy()
        rel_val = val_ratio / max(1e-9, (train_ratio + val_ratio))
        gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed)
        tr_idx, va_idx = next(gss2.split(np.arange(len(df_trval)), groups=df_trval["id"].values))
        df_train, df_val = df_trval.iloc[tr_idx].copy(), df_trval.iloc[va_idx].copy()
        logging.info(f"🔪 Split by individual → {df_train['id'].nunique()} train | {df_val['id'].nunique()} val | {df_test['id'].nunique()} test")
        return {"train": df_train, "val": df_val, "test": df_test}
    split_raw = split_by_individual(raw0, train_ratio, val_ratio)

    def safe_resample_lonlat(df_id: pd.DataFrame, rule: str) -> pd.DataFrame:
        if df_id.empty: return pd.DataFrame()
        df_id = df_id.sort_values("timestamp").copy()
        work = df_id[["timestamp", "longitude", "latitude"]].copy().set_index("timestamp")
        try:
            r = work.resample(rule).mean(numeric_only=True)
        except TypeError:
            r = work.select_dtypes(include=[np.number]).resample(rule).mean()
        r = r.interpolate(method="time").reset_index()
        return r

    def preprocess_one_individual(df_id):
        if df_id.empty: return pd.DataFrame()
        df6 = safe_resample_lonlat(df_id, CONFIG.resample_rule)
        dfk = apply_kalman_filter(df6)
        feat = create_features(
            dfk,
            (sst_ds, sst_var) if sst_ds is not None else None,
            (chl_ds, chl_var) if chl_ds is not None else None,
            (ssh_ds, ssh_var) if ssh_ds is not None else None
        )
        feat["id"] = df_id["id"].iloc[0]
        return feat

    processed = {}
    for name, df_part in split_raw.items():
        logging.info(f"\n— Processing {name.upper()} (by individual) —")
        parts = []
        for _, g in tqdm(df_part.groupby("id", sort=False), desc=f"Processing {name}"):
            g = g[["timestamp","longitude","latitude","id"]].sort_values("timestamp")
            out = preprocess_one_individual(g)
            if not out.empty:
                parts.append(out)
        processed[name] = pd.concat(parts, axis=0, ignore_index=True) if parts else pd.DataFrame()

    BASE = [
        "lon_kalman","lat_kalman",
        "vel_lon_kalman","vel_lat_kalman",
        "speed_deg","accel_deg_h2",
        "hour_sin","hour_cos","doy_sin","doy_cos",
        "u_abs_mps","v_abs_mps","speed_abs_mps",
        "u_rel_mps","v_rel_mps","speed_rel_mps",
        "cos_align","eke"
    ]
    ENV_ALL = ["sst","sst_anom","sst_grad_mag",
               "chl","chl_log10","chl_anom","chl_grad_mag",
               "ssh","ssh_anom","ssh_grad_mag","u_geo","v_geo"]
    SPEC_ALL = [
        "sst_anom_fft_low","sst_anom_fft_high","sst_anom_fft_ratio",
        "chl_log10_fft_low","chl_log10_fft_high","chl_log10_fft_ratio",
        "ssh_anom_fft_low","ssh_anom_fft_high","ssh_anom_fft_ratio"
    ]

    if processed["train"].empty:
        sys.exit("❌ No training data after preprocessing.")

    present_cols = [c for c in BASE if c in processed["train"].columns]
    present_cols += [c for c in ENV_ALL if c in processed["train"].columns]
    present_cols += [c for c in SPEC_ALL if c in processed["train"].columns]
    FEATURES = present_cols
    TARGET_POS = ["lon_kalman","lat_kalman"]

    def clean_for_model(df):
        need = list(set([c for c in FEATURES if c in df.columns] + TARGET_POS))
        return df.dropna(subset=need).copy()

    cleaned = {k: clean_for_model(v) for k, v in processed.items()}
    if cleaned["train"].empty:
        logging.error("\n🧪 DEBUG: NaN count per column in TRAIN")
        logging.error(processed["train"][list(set(FEATURES + TARGET_POS))].isna().sum().sort_values(ascending=False))
        sys.exit("❌ No training data after cleaning.")

    logging.info("\n📍 Fitting hotspots (DBSCAN) with TRAIN…")
    hs = fit_hotspots(cleaned["train"], CONFIG.dbscan_eps_km, CONFIG.dbscan_min_samples)
    centroids = hs["centroids"]; radii_km = hs["radii_km"]
    K = centroids.shape[0]
    logging.info(f"  ✓ Hotspots detected: K={K}")
    print(hs["zones_summary"])

    for name in ["train","val","test"]:
        if name in cleaned and not cleaned[name].empty:
            z = assign_zones(cleaned[name], centroids, radii_km, unknown_class=CONFIG.unknown_class_index)
            cleaned[name] = cleaned[name].copy(); cleaned[name]["zone_class"] = z

    if CONFIG.enable_undersampling and not cleaned["train"].empty:
        logging.info("\n⚖️ Balancing TRAIN with undersampling…")
        df_train = cleaned['train']
        counts = df_train['zone_class'].value_counts().sort_values(ascending=False)
        if len(counts) > 1:
            excl = set(CONFIG.undersample_exclude_classes)
            counts_eligible = counts[~counts.index.isin(excl)]
            if not counts_eligible.empty:
                majority_class = counts_eligible.idxmax()
                second_size = counts_eligible.iloc[1] if len(counts_eligible) > 1 else counts_eligible.iloc[0]
                threshold = int(max(1, CONFIG.undersample_factor * second_size))
                df_majority = df_train[df_train['zone_class'] == majority_class]
                df_others   = df_train[df_train['zone_class'] != majority_class]
                n_take = min(len(df_majority), threshold)
                if n_take < len(df_majority):
                    df_majority_down = df_majority.sample(n=n_take, random_state=CONFIG.seed)
                    df_bal = pd.concat([df_majority_down, df_others], axis=0)
                    if 'timestamp' in df_bal.columns:
                        df_bal = df_bal.sort_values(['id','timestamp'])
                    cleaned['train'] = df_bal.reset_index(drop=True)
                    logging.info(f"✔️ Undersampling applied to class {majority_class}.")
                else:
                    logging.info("ⓘ Majority class is already below threshold; skipping.")
        logging.info(f"TRAIN counts: {cleaned['train']['zone_class'].value_counts().to_dict()}")

    if CONFIG.enable_corr_analysis:
        if CONFIG.corr_compute_on == "all":
            df_corr_base = pd.concat([cleaned.get("train", pd.DataFrame()),
                                      cleaned.get("val",   pd.DataFrame()),
                                      cleaned.get("test",  pd.DataFrame())], axis=0, ignore_index=True)
        else:
            df_corr_base = cleaned["train"].copy()
        x_col = "speed_rel_mps" if "speed_rel_mps" in df_corr_base.columns else \
                ("foraging_label" if "foraging_label" in df_corr_base.columns else "speed_abs_mps")
        y_col = "sst_anom" if "sst_anom" in df_corr_base.columns else ("sst" if "sst" in df_corr_base.columns else None)
        time_df = freq_df = None
        if y_col is not None:
            logging.info(f"\n📈 Correlations — x={x_col} vs y={y_col}")
            time_df = compute_time_domain_correlation_df(
                df_corr_base, x_col=x_col, y_col=y_col,
                spatial_res_deg=CONFIG.corr_spatial_res_deg,
                temporal_rule=CONFIG.corr_temporal_rule,
                min_observations=CONFIG.corr_min_obs_time,
                max_lag_steps=CONFIG.corr_max_lag_steps
            )
            freq_df = compute_frequency_domain_coherence_df(
                df_corr_base, x_col=x_col, y_col=y_col,
                spatial_res_deg=CONFIG.corr_spatial_res_deg,
                temporal_rule=CONFIG.corr_temporal_rule,
                min_observations=CONFIG.corr_min_obs_freq
            )
            if time_df is not None: time_df.to_csv(CONFIG.corr_output_dir / "time_domain_correlations.csv", index=False)
            if freq_df is not None: freq_df.to_csv(CONFIG.corr_output_dir / "frequency_domain_correlations.csv", index=False)
            _quick_corr_plots(time_df, freq_df, str(CONFIG.corr_output_dir / "corr_coh_overview.png"))
            logging.info(f"  ✓ Correlation summary saved in '{CONFIG.corr_output_dir}'")

    scaler_X = MinMaxScaler().fit(cleaned["train"][FEATURES])
    scaler_y = MinMaxScaler().fit(cleaned["train"][TARGET_POS])
    try:
        idx_lon = FEATURES.index("lon_kalman"); idx_lat = FEATURES.index("lat_kalman")
    except ValueError:
        sys.exit("❌ 'lon_kalman' or 'lat_kalman' not in FEATURES.")

    x_min   = np.array([scaler_X.min_[idx_lon],  scaler_X.min_[idx_lat]],  dtype=np.float32)
    x_scale = np.array([scaler_X.scale_[idx_lon],scaler_X.scale_[idx_lat]], dtype=np.float32)
    y_min   = torch.tensor(scaler_y.min_,   dtype=torch.float32, device=device)
    y_scale = torch.tensor(scaler_y.scale_, dtype=torch.float32, device=device)

    def to_loader(df, shuffle):
        if df.empty: return None
        df = df.sort_values(["id","timestamp"]).reset_index(drop=True)
        X = scaler_X.transform(df[FEATURES])
        y_pos = df[TARGET_POS].values
        y_zone = df["zone_class"].astype(int).values
        W = CONFIG.window_size
        starts = (compute_valid_window_starts(df, id_col="id", window=W)
                  if CONFIG.enable_undersampling
                  else np.arange(0, max(0, len(df) - W)))
        starts = starts[(starts + W) < len(df)]
        if len(starts) == 0:
            logging.warning("ⓘ No valid windows (tracks too short for window size)."); return None
        ds = TrajectoryDatasetHybridWindows(X, y_pos, y_zone, W,
                                            idx_lon, idx_lat, x_min, x_scale, starts, df_ref=df)
        return DataLoader(ds, batch_size=CONFIG.batch_size, shuffle=shuffle)

    loaders = {
        "train": to_loader(cleaned["train"], True),
        "val":   to_loader(cleaned["val"], False) if "val" in cleaned and not cleaned["val"].empty else None,
        "test":  to_loader(cleaned["test"], False) if "test" in cleaned and not cleaned["test"].empty else None,
    }

    num_classes = K + 1
    model = HybridPredictorGRU(num_features=len(FEATURES), num_classes=num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG.learning_rate)
    loss_loc = nn.MSELoss()
    cls_counts  = np.bincount(cleaned["train"]["zone_class"].values, minlength=num_classes).astype(float)
    cls_weights = 1.0/(cls_counts + 1e-6); cls_weights = cls_weights/cls_weights.mean()
    loss_zone = nn.CrossEntropyLoss(weight=torch.tensor(cls_weights, dtype=torch.float32, device=device))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=CONFIG.early_stopping_patience//2)

    class Stopper:
        def __init__(self, patience=CONFIG.early_stopping_patience,
                     path=CONFIG.results_dir / "best_hybrid_model.pt"):
            self.patience = patience; self.path = path; self.best = None; self.counter = 0
        def step(self, val, model):
            score = -val
            if self.best is None or score > self.best:
                self.best = score; self.counter = 0; torch.save(model.state_dict(), self.path)
            else:
                self.counter += 1
            return self.counter >= self.patience
    stopper = Stopper()

    logging.info("\n🧠 Training Bi-GRU (∆position + t+1 hotspot)…")
    tr_losses = []; va_losses = []
    for epoch in range(CONFIG.epochs):
        if loaders["train"] is None:
            sys.exit("❌ No training windows. Check window_size or track lengths.")
        model.train(); tr_loss = 0.0; n_batches_tr = 0
        for seq, y_pos, y_zone, last_unscaled, _target_idx in tqdm(loaders["train"], desc=f"Epoch {epoch+1}/{CONFIG.epochs}"):
            seq = seq.to(device); y_pos = y_pos.to(device); y_zone = y_zone.to(device); last_unscaled = last_unscaled.to(device)
            delta, logits = model(seq)
            pred_unscaled = last_unscaled + delta
            pred_scaled = pred_unscaled * y_scale + y_min
            true_scaled = y_pos * y_scale + y_min
            loss = loss_loc(pred_scaled, true_scaled) + CONFIG.lambda_zone * loss_zone(logits, y_zone)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG.gradient_clip_value)
            opt.step()
            tr_loss += loss.item(); n_batches_tr += 1
        tr_losses.append(tr_loss / max(1, n_batches_tr))

        if loaders["val"] is not None:
            model.eval(); val_loss = 0.0; n_batches_val = 0; Yv = []; Pv = []
            with torch.no_grad():
                for seq, y_pos, y_zone, last_unscaled, _target_idx in loaders["val"]:
                    seq = seq.to(device); y_pos = y_pos.to(device); y_zone = y_zone.to(device); last_unscaled = last_unscaled.to(device)
                    delta, logits = model(seq)
                    pred_unscaled = last_unscaled + delta
                    pred_scaled = pred_unscaled * y_scale + y_min
                    true_scaled = y_pos * y_scale + y_min
                    l = loss_loc(pred_scaled, true_scaled) + CONFIG.lambda_zone * loss_zone(logits, y_zone)
                    val_loss += l.item(); n_batches_val += 1
                    Pv.append(torch.argmax(logits, dim=1).cpu().numpy()); Yv.append(y_zone.cpu().numpy())
            val_loss /= max(1, n_batches_val); va_losses.append(val_loss)
            scheduler.step(val_loss)
            stop = stopper.step(val_loss, model)
            Yv = np.concatenate(Yv) if Yv else np.array([], int)
            Pv = np.concatenate(Pv) if Pv else np.array([], int)
            acc = accuracy_score(Yv, Pv) if len(Yv) else float("nan")
            f1m = f1_score(Yv, Pv, average="macro") if len(Yv) else float("nan")
            logging.info(f"  TrainLoss {tr_losses[-1]:.5f} | ValLoss {val_loss:.5f} | Acc {acc:.3f} | F1(macro) {f1m:.3f}")
            if stop:
                logging.info("🛑 Early stopping triggered."); break
        else:
            logging.info(f"  TrainLoss {tr_losses[-1]:.5f} (no validation set)")

    if va_losses:
        fig, ax = plt.subplots(figsize=(8.4,4.8))
        ax.plot(range(1, len(tr_losses)+1), tr_losses, label="Train Loss")
        ax.plot(range(1, len(va_losses)+1), va_losses, label="Val Loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend()
        fp = CONFIG.results_dir / "loss_curve.png"
        plt.tight_layout(); plt.savefig(fp, dpi=220); plt.close(fig)

    if loaders["test"] is not None:
        logging.info("\n📊 Evaluating on TEST …")
        try:
            model.load_state_dict(torch.load(CONFIG.results_dir / "best_hybrid_model.pt", map_location=device))
        except Exception:
            logging.warning("⚠️ Could not load 'best_hybrid_model.pt'; using current weights.")

        model.eval()
        P_pos = []; T_pos = []; P_cls = []; T_cls = []; Idxs = []
        with torch.no_grad():
            for seq, y_pos, y_zone, last_unscaled, target_idx in loaders["test"]:
                seq = seq.to(device); last_unscaled = last_unscaled.to(device)
                delta, logits = model(seq)
                P_pos.append((last_unscaled + delta).cpu().numpy())
                P_cls.append(torch.argmax(logits, dim=1).cpu().numpy())
                T_pos.append(y_pos.numpy()); T_cls.append(y_zone.numpy())
                Idxs.append(target_idx.numpy())
        P_pos = np.concatenate(P_pos) if P_pos else np.zeros((0,2))
        T_pos = np.concatenate(T_pos) if T_pos else np.zeros((0,2))
        P_cls = np.concatenate(P_cls) if P_cls else np.zeros((0,), dtype=int)
        T_cls = np.concatenate(T_cls) if T_cls else np.zeros((0,), dtype=int)

        if len(T_pos):
            err = haversine_km(T_pos[:,1], T_pos[:,0], P_pos[:,1], P_pos[:,0])
            logging.info(f"    • Mean error (km):    {np.mean(err):.2f}")
            logging.info(f"    • Median error (km): {np.median(err):.2f}")

        if len(T_cls):
            acc = accuracy_score(T_cls, P_cls); f1m = f1_score(T_cls, P_cls, average="macro"); f1w = f1_score(T_cls, P_cls, average="weighted")
            logging.info(f"    • Zone accuracy:      {acc:.3f}")
            logging.info(f"    • F1 (macro):         {f1m:.3f}")
            logging.info(f"    • F1 (weighted):      {f1w:.3f}")
            print("\nClassification Report (TEST):")
            target_names = ["HS0_unknown"] + [f"HS{j}" for j in range(1, K + 1)]
            print(classification_report(T_cls, P_cls, labels=list(range(K + 1)), target_names=target_names, zero_division=0))

        try:
            plot_map_zones(cleaned, centroids, radii_km,
                           save_path=str(CONFIG.results_dir / "map_zones.png"))
            all_pts = pd.concat([cleaned.get("train", pd.DataFrame()),
                                 cleaned.get("val",   pd.DataFrame()),
                                 cleaned.get("test",  pd.DataFrame())], axis=0, ignore_index=True)
            plot_map_hotspots_only(centroids, radii_km, all_pts,
                                   save_path=str(CONFIG.results_dir / "map_hotspots_only.png"))
            plot_map_individuals(all_pts[["id","timestamp","lat_kalman","lon_kalman"]].dropna().copy(),
                                 save_path=str(CONFIG.results_dir / "map_individuals.png"),
                                 centroids=centroids, radii_km=radii_km)
        except Exception as e:
            logging.error(f"⚠️ Could not generate one of the maps: {e}")
    else:
        logging.warning("\n⚠️ No TEST set with enough rows/windows.")

    if CONFIG.enable_qna:
        logging.info("\n🔎 Running Q&A module (Q1..Q6)…")
        df_all = pd.concat([cleaned.get("train", pd.DataFrame()),
                            cleaned.get("val",   pd.DataFrame()),
                            cleaned.get("test",  pd.DataFrame())], axis=0, ignore_index=True)
        qna = EcoQnA(df_all,
                     (sst_ds, sst_var) if sst_ds is not None else None,
                     (chl_ds, chl_var) if chl_ds is not None else None,
                     (ssh_ds, ssh_var) if ssh_ds is not None else None,
                     outdir=str(CONFIG.qna_output_dir))
        answers = {}
        answers["Q1"] = qna.Q1_alignment_vs_eke()
        answers["Q2"] = qna.Q2_sst_used_available()
        answers["Q3"] = qna.Q3_chl_used_available()
        answers["Q4"] = qna.Q4_lag_sst_behavior()
        answers["Q5"] = qna.Q5_mesoscale_regime()
        answers["Q6"] = qna.Q6_seasonal_hotspots(eps_km=35.0, min_samples=15)
        qna.build_report(answers)

    try:
        if sst_ds is not None: sst_ds.close()
        if chl_ds is not None: chl_ds.close()
        if ssh_ds is not None: ssh_ds.close()
    except Exception:
        pass

    logging.info("\n✅ Full pipeline finished.")

if __name__ == "__main__":
    main()