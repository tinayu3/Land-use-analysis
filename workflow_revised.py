from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
import geopandas as gpd

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

try:
    from libpysal.weights import lat2W
    from esda.moran import Moran, Moran_Local
    HAS_PYSAL = True
except ImportError:
    HAS_PYSAL = False


@dataclass
class LULCConfig:
    base_dir: Path = Path("data")
    output_dir: Path = Path("outputs")

    modis_dir: Path = field(init=False)
    dem_path: Path = field(init=False)
    basin_shp: Path = field(init=False)
    station_obs_csv: Path = field(init=False)
    station_reanalysis_csv: Path = field(init=False)

    years: Tuple[int, ...] = (2000, 2005, 2010, 2015, 2020)

    # Class mapping: numeric labels for SVM / rasters
    lulc_classes: Dict[int, str] = field(default_factory=lambda: {
        1: "cropland",
        2: "forest",
        3: "grassland",
        4: "water",
        5: "built_up",
        6: "unused"
    })

    eeq_index: Dict[str, float] = field(default_factory=lambda: {
        "cropland": 0.57,
        "forest": 2.48,
        "grassland": 1.32,
        "water": 3.79,
        "built_up": 0.54,
        "unused": 0.79
    })

    def __post_init__(self):
        self.modis_dir = self.base_dir / "raw" / "modis"
        self.dem_path = self.base_dir / "ancillary" / "dem_yangtze.tif"
        self.basin_shp = self.base_dir / "ancillary" / "yangtze_basin_boundary.shp"
        self.station_obs_csv = self.base_dir / "climate" / "stations_observed_temp.csv"
        self.station_reanalysis_csv = self.base_dir / "climate" / "stations_reanalysis_temp.csv"

        self.output_dir.mkdir(parents=True, exist_ok=True)


def load_raster(path: Path) -> Tuple[np.ndarray, rasterio.Affine, dict]:
    with rasterio.open(path) as src:
        data = src.read()
        transform = src.transform
        meta = src.meta
    return data, transform, meta


def save_raster(path: Path, data: np.ndarray, meta: dict):
    meta = meta.copy()
    
    if data.ndim == 2:
        meta.update(count=1)
    else:
        meta.update(count=data.shape[0])
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(data.astype(meta.get("dtype", "int16")))


def mask_to_basin(raster_path: Path, basin_shp: Path, out_path: Path):
    with rasterio.open(raster_path) as src:
        gdf = gpd.read_file(basin_shp).to_crs(src.crs)
        geometry = [geom for geom in gdf.geometry]
        out_image, out_transform = rasterio.mask.mask(src, geometry, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
    save_raster(out_path, out_image, out_meta)


def preprocess_modis_year(config: LULCConfig, year: int) -> Path:
    mosaic_path = config.modis_dir / f"MOD09A1_{year}_mosaic.tif"

    if not mosaic_path.exists():
        raise FileNotFoundError(f"Missing MODIS mosaic for {year}: {mosaic_path}")

    out_path = config.output_dir / f"MOD09A1_{year}_yangtze_clipped.tif"

    print(f"[Preprocess] Clipping MODIS {year} to basin...")
    mask_to_basin(mosaic_path, config.basin_shp, out_path)

    return out_path


# LULC

def extract_training_data(
    lulc_config: LULCConfig,
    preprocessed_raster: Path,
    training_shp: Path
) -> Tuple[np.ndarray, np.ndarray]:
    
    with rasterio.open(preprocessed_raster) as src:
        bands = src.read()  # (bands, h, w)
        transform = src.transform
        crs = src.crs

    gdf = gpd.read_file(training_shp).to_crs(crs)

    samples = []
    labels = []

    for _, row in gdf.iterrows():
        geom = [row.geometry]
        class_id = int(row["class_id"])

        mask = rasterize(
            geom,
            out_shape=bands.shape[1:],
            transform=transform,
            fill=0,
            all_touched=True
        )


        idx = np.where(mask == 1)
        
        pixel_values = bands[:, idx[0], idx[1]].T  

        samples.append(pixel_values)
        labels.append(np.full(pixel_values.shape[0], class_id, dtype=int))

    X = np.vstack(samples)
    y = np.concatenate(labels)

    print(f"[Training data] Extracted {X.shape[0]} samples with {X.shape[1]} bands.")
    return X, y


def train_svm_classifier(X: np.ndarray, y: np.ndarray) -> SVC:
    
    clf = SVC(kernel="rbf", C=100.0, gamma="scale")
    print("[SVM] Training...")
    clf.fit(X, y)
    print("[SVM] Training done.")
    return clf


def classify_raster_svm(
    preprocessed_raster: Path,
    clf: SVC,
    out_path: Path
) -> Path:
    
    with rasterio.open(preprocessed_raster) as src:
        bands = src.read()  
        meta = src.meta

    n_bands, h, w = bands.shape
    X_pred = bands.reshape(n_bands, -1).T  

    print(f"[SVM] Classifying raster of shape {h}x{w} ...")
    y_pred = clf.predict(X_pred).astype(np.int16)
    lulc_map = y_pred.reshape(h, w)

    meta_out = meta.copy()
    meta_out.update(count=1, dtype="int16")

    save_raster(out_path, lulc_map, meta_out)
    print(f"[SVM] Saved classified LULC map: {out_path}")
    return out_path


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: Dict[int, str]
):
    
    cm = confusion_matrix(y_true, y_pred, labels=list(class_labels.keys()))
    OA = np.sum(np.diag(cm)) / np.sum(cm)

    # Kappa
    total = np.sum(cm)
    row_marginals = np.sum(cm, axis=1)
    col_marginals = np.sum(cm, axis=0)
    expected = np.outer(row_marginals, col_marginals) / total
    kappa = (np.sum(np.diag(cm)) - np.sum(expected.diagonal())) / (total - np.sum(expected.diagonal()))

    print("[Evaluation] Overall Accuracy (OA):", OA)
    print("[Evaluation] Kappa:", kappa)
    print("[Evaluation] Confusion Matrix:\n", cm)
    print("[Evaluation] Classification report:\n",
          classification_report(y_true, y_pred, target_names=list(class_labels.values())))


def compute_single_dynamic_index(
    area_start: float,
    area_end: float,
    T_years: float
) -> float:
    #land-use dynamic index
    return ((area_end - area_start) / area_start) * (1.0 / T_years) * 100.0


def compute_integrated_dynamic_index(
    areas_start: Dict[int, float],
    areas_end: Dict[int, float],
    T_years: float
) -> float:
    
    num = sum(abs(areas_end[i] - areas_start[i]) for i in areas_start)
    den = 2.0 * sum(areas_start.values())
    return (num / den) * (1.0 / T_years) * 100.0


def lulc_area_from_raster(lulc_raster: Path, class_ids: List[int]) -> Dict[int, float]:
    
    with rasterio.open(lulc_raster) as src:
        arr = src.read(1)
    areas = {}
    for cid in class_ids:
        areas[cid] = float(np.sum(arr == cid))
    return areas


def transfer_matrix(
    lulc_start: Path,
    lulc_end: Path,
    class_ids: List[int]
) -> np.ndarray:
    
    with rasterio.open(lulc_start) as src1, rasterio.open(lulc_end) as src2:
        a1 = src1.read(1)
        a2 = src2.read(1)

    assert a1.shape == a2.shape, "Rasters must be same shape"
    n = len(class_ids)
    cid_to_idx = {cid: i for i, cid in enumerate(class_ids)}
    mat = np.zeros((n, n), dtype=np.float64)

    for cid_i in class_ids:
        mask = (a1 == cid_i)
        for cid_j in class_ids:
            mat[cid_to_idx[cid_i], cid_to_idx[cid_j]] = np.sum(mask & (a2 == cid_j))

    return mat


# slope

def compute_slope_from_dem(dem_path: Path) -> Tuple[np.ndarray, rasterio.Affine, dict]:
    
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(float)
        transform = src.transform
        meta = src.meta


    dx = transform.a
    dy = -transform.e

    
    dzdx = (np.roll(dem, -1, axis=1) - np.roll(dem, 1, axis=1)) / (2 * dx)
    dzdy = (np.roll(dem, -1, axis=0) - np.roll(dem, 1, axis=0)) / (2 * dy)

    slope_rad = np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2))
    slope_deg = np.degrees(slope_rad)

    meta_out = meta.copy()
    meta_out.update(dtype="float32", count=1)

    return slope_deg.astype(np.float32), transform, meta_out


def categorize_slope(slope: np.ndarray) -> np.ndarray:
    
    bins = [0, 2, 6, 15, 25, 90]
    
    return np.digitize(slope, bins) - 1


def slope_transition_stats(
    lulc_start: Path,
    lulc_end: Path,
    slope_cat_raster: np.ndarray,
    class_ids: List[int]
) -> pd.DataFrame:
    
    with rasterio.open(lulc_start) as s1, rasterio.open(lulc_end) as s2:
        a1 = s1.read(1)
        a2 = s2.read(1)

    assert a1.shape == a2.shape == slope_cat_raster.shape

    records = []
    for cid_i in class_ids:
        for cid_j in class_ids:
            mask_trans = (a1 == cid_i) & (a2 == cid_j)
            for slope_bin in range(5):
                area = np.sum(mask_trans & (slope_cat_raster == slope_bin))
                if area > 0:
                    records.append({
                        "from": cid_i,
                        "to": cid_j,
                        "slope_bin": slope_bin,
                        "area_pixels": int(area)
                    })
    return pd.DataFrame.from_records(records)



def compute_morans_I_for_lulc(lulc_raster: Path) -> Tuple[float, float]:
    # Moran's I 
    if not HAS_PYSAL:
        raise ImportError("libpysal/esda not installed. Install them to compute Moran's I.")

    with rasterio.open(lulc_raster) as src:
        arr = src.read(1)

    
    factor = 10
    arr_small = arr[::factor, ::factor].astype(float)
    nrows, ncols = arr_small.shape

    x = arr_small.flatten()
    
    w = lat2W(nrows, ncols)
    moran = Moran(x, w)
    return moran.I, moran.p_sim


def compute_local_moran(lulc_raster: Path) -> Moran_Local:
    
    if not HAS_PYSAL:
        raise ImportError("libpysal/esda not installed. Install them to compute local Moran.")

    with rasterio.open(lulc_raster) as src:
        arr = src.read(1)

    factor = 10
    arr_small = arr[::factor, ::factor].astype(float)
    nrows, ncols = arr_small.shape
    x = arr_small.flatten()
    w = lat2W(nrows, ncols)
    return Moran_Local(x, w)



def compute_omr(config: LULCConfig) -> pd.DataFrame:
    
    obs = pd.read_csv(config.station_obs_csv)
    rean = pd.read_csv(config.station_reanalysis_csv)

    merged = pd.merge(obs, rean, on=["station_id", "year"], suffixes=("_obs", "_rean"))

    results = []

    for metric in ["t_mean", "t_min", "t_max"]:
        col_obs = f"{metric}_obs"
        col_rean = f"{metric}_rean"

        for sid, group in merged.groupby("station_id"):
            
            years = group["year"].values
            y_obs = group[col_obs].values
            y_rean = group[col_rean].values

            if len(years) < 3:
                continue

            slope_obs, _ = np.polyfit(years, y_obs, 1)
            slope_rean, _ = np.polyfit(years, y_rean, 1)

            omr = slope_obs - slope_rean
            Eu = abs(omr / slope_obs) * 100 if slope_obs != 0 else np.nan

            results.append({
                "station_id": sid,
                "metric": metric,
                "trend_obs": slope_obs,
                "trend_rean": slope_rean,
                "OMR": omr,
                "Eu_percent": Eu
            })

    return pd.DataFrame(results)



def compute_eeq_for_lulc(
    lulc_raster: Path,
    config: LULCConfig
) -> float:
    
    with rasterio.open(lulc_raster) as src:
        arr = src.read(1)

    total_pixels = arr.size
    total_score = 0.0

    for cid, cname in config.lulc_classes.items():
        eeq_val = config.eeq_index[cname]
        count = np.sum(arr == cid)
        total_score += eeq_val * count

    return total_score / total_pixels


def eeq_contribution_of_transition(
    lulc_start: Path,
    lulc_end: Path,
    config: LULCConfig
) -> pd.DataFrame:
    
    with rasterio.open(lulc_start) as s1, rasterio.open(lulc_end) as s2:
        a1 = s1.read(1)
        a2 = s2.read(1)

    assert a1.shape == a2.shape

    records = []
    for cid_i, name_i in config.lulc_classes.items():
        for cid_j, name_j in config.lulc_classes.items():
            mask = (a1 == cid_i) & (a2 == cid_j)
            area = np.sum(mask)
            if area == 0:
                continue

            eeq_i = config.eeq_index[name_i]
            eeq_j = config.eeq_index[name_j]
            delta = (eeq_j - eeq_i) * area

            records.append({
                "from": cid_i,
                "to": cid_j,
                "from_name": name_i,
                "to_name": name_j,
                "area_pixels": int(area),
                "eeq_change": delta
            })

    df = pd.DataFrame.from_records(records)
    return df



def main():
    config = LULCConfig()

    preprocessed_paths = {}
    for year in config.years:
        preprocessed_paths[year] = preprocess_modis_year(config, year)

    
    training_shp = config.base_dir / "training" / "training_samples_2000.shp"
    X_train, y_train = extract_training_data(config, preprocessed_paths[2000], training_shp)
    clf = train_svm_classifier(X_train, y_train)

    lulc_maps = {}
    for year in config.years:
        out_lulc = config.output_dir / f"lulc_{year}.tif"
        lulc_maps[year] = classify_raster_svm(preprocessed_paths[year], clf, out_lulc)

   
    print("\n[Evaluation] (Demo using training data as validation)")
    y_pred_demo = clf.predict(X_train)
    evaluate_classification(y_train, y_pred_demo, config.lulc_classes)

   
    years_pairs = list(zip(config.years[:-1], config.years[1:]))
    class_ids = list(config.lulc_classes.keys())

    for y0, y1 in years_pairs:
        areas0 = lulc_area_from_raster(lulc_maps[y0], class_ids)
        areas1 = lulc_area_from_raster(lulc_maps[y1], class_ids)
        T_years = y1 - y0

        print(f"\n[Land-use dynamics] {y0}-{y1}")
        for cid in class_ids:
            lc_i = compute_single_dynamic_index(areas0[cid], areas1[cid], T_years)
            cname = config.lulc_classes[cid]
            print(f"  {cname}: {lc_i:.5f} %/year")

        lc = compute_integrated_dynamic_index(areas0, areas1, T_years)
        print(f"  Integrated land-use dynamic index: {lc:.5f} %/year")

        # Transfer matrix
        mat = transfer_matrix(lulc_maps[y0], lulc_maps[y1], class_ids)
        print("  Transfer matrix (rows from, cols to):")
        print(mat)

   
    slope, slope_transform, slope_meta = compute_slope_from_dem(config.dem_path)
    slope_cat = categorize_slope(slope)

    
    df_slope_stats = slope_transition_stats(
        lulc_maps[2015],
        lulc_maps[2020],
        slope_cat,
        class_ids
    )
    df_slope_stats.to_csv(config.output_dir / "slope_transition_2015_2020.csv", index=False)
    print("\n[Slope transitions] Saved slope-dependent transitions to CSV.")

    
    if HAS_PYSAL:
        for year in config.years:
            I, p = compute_morans_I_for_lulc(lulc_maps[year])
            print(f"[Moran's I] Year {year}: I = {I:.4f}, p = {p:.4f}")
    else:
        print("\n[Warning] libpysal/esda not installed, Moran's I skipped.")

    
    if config.station_obs_csv.exists() and config.station_reanalysis_csv.exists():
        df_omr = compute_omr(config)
        df_omr.to_csv(config.output_dir / "omr_results.csv", index=False)
        print("[OMR] Saved OMR results to CSV.")
    else:
        print("[OMR] Station or reanalysis CSV not found, skipping OMR.")

    
    eeq_values = {}
    for year in config.years:
        eeq = compute_eeq_for_lulc(lulc_maps[year], config)
        eeq_values[year] = eeq
        print(f"[EEQ] Year {year}: mean EEQ = {eeq:.4f}")

    
    df_eeq_trans = eeq_contribution_of_transition(lulc_maps[2010], lulc_maps[2020], config)
    df_eeq_trans.to_csv(config.output_dir / "eeq_contributions_2010_2020.csv", index=False)
    print("[EEQ] Saved EEQ transition contributions to CSV.")

    print("\n[Done] Full pipeline executed. Check the 'outputs' directory for results.")


if __name__ == "__main__":
    main()