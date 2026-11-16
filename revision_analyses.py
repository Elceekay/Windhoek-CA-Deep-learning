"""revision_analyses.py - analyses requested by the reviewers (Windhoek CA-deep learning manuscript)."""
import os
import time
import resource
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import numpy as np
import pandas as pd
from scipy import ndimage, optimize

PX_KM2 = 900 / 1e6
NAMES = ["Urban", "Vegetation", "Bare Ground", "Water"]
D_DEFAULT = "/content/drive/MyDrive/GEE_Windhoek/"
PAPER = {1990: "Windhoek_LULC_1990_corrected_v3.tif", 1995: "Windhoek_LULC_1995_corrected_v2.tif",
         2000: "Windhoek_LULC_2000_corrected.tif", 2005: "Windhoek_LULC_2005_corrected.tif",
         2010: "Windhoek_LULC_2010_Final.tif", 2015: "Windhoek_LULC_2015_Final.tif",
         2020: "Windhoek_LULC_2020_Final.tif", 2025: "Windhoek_LULC_2025_Final_FIXED.tif"}
RAW = {y: f"Windhoek_LULC_{y}.tif" for y in PAPER}
K, HK = 9, 4


# =====================================================================================================
# 1. ACCURACY AND AREA
# =====================================================================================================
def error_matrix_summary(cm, names=NAMES):
    """cm: counts, rows = reference, columns = map."""
    cm = np.asarray(cm, float)
    n = cm.sum()
    diag = np.diag(cm)
    ref_tot, map_tot = cm.sum(1), cm.sum(0)
    pa, ua = diag / np.maximum(ref_tot, 1e-12), diag / np.maximum(map_tot, 1e-12)
    f1 = 2 * pa * ua / np.maximum(pa + ua, 1e-12)
    po = diag.sum() / max(n, 1e-12)
    pe = (ref_tot / max(n, 1e-12) * map_tot / max(n, 1e-12)).sum()
    df = pd.DataFrame({"reference_total": ref_tot, "map_total": map_tot, "PA_%": 100 * pa, "UA_%": 100 * ua,
                       "F1_%": 100 * f1}, index=names)
    return df.round(1), {"OA_%": round(100 * po, 2), "kappa": round((po - pe) / max(1 - pe, 1e-12), 3),
                         "expected_agreement_pe": round(pe, 4)}


def balanced_oa_ci(pa_percent, n_per_class=50):
    """95% CI of the class-balanced OA (mean of PAs) under allocation by reference class (Table 3)."""
    p = np.asarray(pa_percent, float) / 100
    se = np.sqrt((p * (1 - p) / n_per_class).sum()) / len(p)
    return round(100 * p.mean(), 1), round(196 * se, 1)


def _calibrate(cm, m):
    """Classical calibration: m = C pi, C[k, j] = P(map k | ref j); solve with pi >= 0, sum(pi) = 1."""
    C = (cm / cm.sum(1, keepdims=True)).T
    try:
        pi = np.linalg.solve(C, m)
    except np.linalg.LinAlgError:
        pi = np.full(len(m), -1.0)
    if (pi < 0).any():            # inadmissible solution: constrained least squares on the simplex
        k = len(m)
        r = optimize.minimize(lambda p: ((C @ p - m) ** 2).sum(), m, method="SLSQP", bounds=[(0, 1)] * k,
                              constraints=[{"type": "eq", "fun": lambda p: p.sum() - 1}],
                              options={"ftol": 1e-14, "maxiter": 500})
        pi = np.clip(r.x, 0, None)
    pi = pi / pi.sum()
    oa = float((pi * np.diag(C)).sum())
    ua = np.clip(pi * np.diag(C) / np.maximum(m, 1e-12), 0, 1)
    return pi, oa, ua


def _calibrate_fixed(cm, m, fixed):
    """Calibrate with some classes held at their mapped proportion (e.g. Water)."""
    free = [k for k in range(len(m)) if k not in fixed]
    if not fixed:
        return _calibrate(cm, m)
    sub = cm[np.ix_(free, free)]
    mf = m[free] / m[free].sum()
    pf, _, _ = _calibrate(sub, mf)
    pi = np.zeros(len(m)); pi[fixed] = m[fixed]; pi[free] = pf * m[free].sum()
    C = (cm / cm.sum(1, keepdims=True)).T
    oa = float((pi * np.diag(C)).sum())
    ua = np.clip(pi * np.diag(C) / np.maximum(m, 1e-12), 0, 1)
    return pi, oa, ua


def calibration_estimator(cm, mapped_km2, n_boot=2000, seed=0, names=NAMES, fix=("Water",)):
    """Error-adjusted areas, area-weighted OA and UA for a sample allocated by REFERENCE class
    (Czaplewski & Catts, 1992). cm rows = reference, columns = map. Fills Tables 2, 3 and S4."""
    cm = np.asarray(cm, float)
    if (cm.sum(1) == 0).any():
        raise ValueError("a reference class has no validation points")
    mapped = np.asarray(mapped_km2, float)
    A_tot, m = mapped.sum(), mapped / mapped.sum()
    fixed = [names.index(f) for f in fix]
    pi, oa, ua = _calibrate_fixed(cm, m, fixed)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        cb = np.vstack([rng.multinomial(int(r.sum()), r / r.sum()) for r in cm]).astype(float)
        pb, ob, ub_ = _calibrate_fixed(cb, m, fixed)
        boots.append(np.concatenate([pb * A_tot, [ob], ub_]))
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    k = len(names)
    out = pd.DataFrame({"mapped_km2": mapped, "adjusted_km2": pi * A_tot, "adj_lo": lo[:k], "adj_hi": hi[:k],
                        "UA_adjusted_%": 100 * ua, "UA_lo": 100 * lo[k + 1:], "UA_hi": 100 * hi[k + 1:]},
                       index=names).round(2)
    oa_txt = f"{100 * oa:.1f} ({100 * lo[k]:.1f}–{100 * hi[k]:.1f})"
    return out, {"area_weighted_OA_%": oa_txt}


def olofsson_estimator(cm_map_rows, mapped_km2, names=NAMES):
    """Stratified estimators of Olofsson et al. (2014) for the NEW independent sample.
    cm_map_rows: counts, rows = MAP class (strata), columns = reference. Fills Table S8."""
    n = np.asarray(cm_map_rows, float)
    Amap = np.asarray(mapped_km2, float)
    W = Amap / Amap.sum()
    ni = n.sum(1)
    p = W[:, None] * n / ni[:, None]
    pj = p.sum(0)
    area = pj * Amap.sum()
    se_pj = np.sqrt(((W[:, None] ** 2) * (n / ni[:, None]) * (1 - n / ni[:, None]) / (ni[:, None] - 1)).sum(0))
    U = np.diag(n) / ni
    oa = np.trace(p)
    se_oa = np.sqrt((W ** 2 * U * (1 - U) / (ni - 1)).sum())
    se_ua = np.sqrt(U * (1 - U) / (ni - 1))
    PA = np.diag(p) / pj
    Nhat_j = (Amap[:, None] / ni[:, None] * n).sum(0)
    se_pa = []
    for j in range(len(names)):
        t1 = Amap[j] ** 2 * (1 - PA[j]) ** 2 * U[j] * (1 - U[j]) / (ni[j] - 1)
        t2 = PA[j] ** 2 * sum(Amap[i] ** 2 * (n[i, j] / ni[i]) * (1 - n[i, j] / ni[i]) / (ni[i] - 1)
                              for i in range(len(names)) if i != j)
        se_pa.append(np.sqrt(t1 + t2) / Nhat_j[j])
    df = pd.DataFrame({"area_km2": area, "area_ci95_km2": 1.96 * se_pj * Amap.sum(), "UA_%": 100 * U,
                       "UA_ci95": 196 * se_ua, "PA_%": 100 * PA, "PA_ci95": 196 * np.array(se_pa)},
                      index=names).round(2)
    return df, {"OA_%": round(100 * oa, 2), "OA_ci95": round(196 * se_oa, 2)}


# =====================================================================================================
# 2. MAPS, CORRECTION STAGES, REVERSALS
# =====================================================================================================
def read(path):
    import rasterio
    with rasterio.open(path) as s:
        return s.read(1), s.profile


def correction_stage_stats(raw, auto, final, valid):
    """Share of pixels changed and urban-area change between P1 (raw), P2 (automated) and P3 (final)."""
    v = valid
    u = lambda a: float(((a == 1) & v).sum() * PX_KM2)
    ch = lambda a, b: float(100 * (a[v] != b[v]).mean())
    return {"urban_P1": u(raw), "urban_P2": u(auto), "urban_P3": u(final),
            "pix_changed_P1_P2_%": ch(raw, auto), "pix_changed_P2_P3_%": ch(auto, final),
            "pix_changed_P1_P3_%": ch(raw, final),
            "urban_change_P1_P2": u(auto) - u(raw), "urban_change_P2_P3": u(final) - u(auto)}


def reversal_stats(maps, valid):
    """Urban -> non-urban per interval and 'unstable' cells (urban, later non-urban, later urban again)."""
    yrs = sorted(maps)
    rows = []
    for a, b in zip(yrs, yrs[1:]):
        ua, ub = (maps[a] == 1) & valid, (maps[b] == 1) & valid
        lost = ua & ~ub
        rows.append({"interval": f"{a}-{b}", "urban_start_km2": ua.sum() * PX_KM2, "lost_km2": lost.sum() * PX_KM2,
                     "lost_%": 100 * lost.sum() / max(ua.sum(), 1)})
    U = np.stack([(maps[y] == 1) & valid for y in yrs])
    seen_urban = np.maximum.accumulate(U, axis=0)
    left_after_urban = np.zeros_like(valid)
    unstable = np.zeros_like(valid)
    for k in range(1, len(yrs)):
        left_after_urban |= seen_urban[k - 1] & ~U[k]
        unstable |= left_after_urban & U[k]
    df = pd.DataFrame(rows).round(2)
    return df, unstable, float(unstable.sum() * PX_KM2)


# =====================================================================================================
# 2b. VALIDATION POINTS -> ERROR MATRICES; AUTOMATED (P2) MAPS
# =====================================================================================================
def class_areas(L):
    """Mapped area (km²) of classes 1-4 over all valid pixels of a map (municipal extent)."""
    return [round(float((L == k).sum() * PX_KM2), 2) for k in (1, 2, 3, 4)]


def _find(cols, *cands):
    low = {str(c).lower(): c for c in cols}
    for c in cands:
        if c in low:
            return low[c]
    return None


def read_points(D, year, ref_col=None):
    """Validation points: from the .shp (true label 'ref'), else from the .csv; duplicate spots counted once."""
    import json
    import geopandas as gpd
    from shapely.geometry import shape
    shp, csv = f"{D}ValidationPoints_{year}.shp", f"{D}ValidationPoints_{year}.csv"
    g, src = None, None
    if os.path.exists(shp):
        g = gpd.read_file(shp)
        if g.crs is None:
            g = g.set_crs(4326)
        g = g[g.geometry.notna() & ~g.geometry.is_empty]
        src = "shp"
        if len(g) == 0:
            g = None
    if g is None and os.path.exists(csv):
        df = pd.read_csv(csv)
        geom = [shape(json.loads(t)) if isinstance(t, str) else None for t in df[".geo"]]
        g = gpd.GeoDataFrame(df, geometry=geom, crs=4326)
        g = g[g.geometry.notna() & ~g.geometry.is_empty]
        src = "csv"
    if g is None or len(g) == 0:
        raise FileNotFoundError(f"No validation points with locations for {year} in {D}")
    rc = _find(g.columns, "ref", "landcover", "reference", "lc")
    g = g.explode(index_parts=False)
    xy = g.geometry.apply(lambda p: (round(p.x, 7), round(p.y, 7)))
    dup = xy.duplicated().values
    g = g[~dup]
    g.attrs["source"], g.attrs["duplicates_removed"] = src, int(dup.sum())
    return g, rc


def points_matrix(D, year, raster_path, ref_col=None):
    """Error matrix (rows = reference 1-4, columns = map 1-4) of a raster at the validation points."""
    import rasterio
    from rasterio.transform import rowcol
    g, ref_col = read_points(D, year)
    if len(g) == 0:
        return np.zeros((4, 4), int), 0
    with rasterio.open(raster_path) as src:
        L = src.read(1); g = g.to_crs(src.crs)
        pts = g.geometry if (g.geom_type == "Point").all() else g.geometry.representative_point()
        r, c = rowcol(src.transform, pts.x.values, pts.y.values)
    r, c = np.asarray(r), np.asarray(c)
    ok = (r >= 0) & (r < L.shape[0]) & (c >= 0) & (c < L.shape[1])
    ref = pd.to_numeric(g[ref_col], errors="coerce").fillna(-1).values.astype(int)[ok]
    mp = L[r[ok], c[ok]].astype(int)
    keep = np.isin(ref, [1, 2, 3, 4]) & np.isin(mp, [1, 2, 3, 4])
    cm = np.zeros((4, 4), int)
    np.add.at(cm, (ref[keep] - 1, mp[keep] - 1), 1)
    return cm, int((~ok).sum() + (~keep).sum())


def csv_matrix(D, year):
    """Published (before-correction) error matrix: 'ref' = true label, 'landcover' = GEE classification."""
    f = f"{D}ValidationPoints_{year}.csv"
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f)
    if not {"ref", "landcover"} <= set(df.columns):
        return None
    cm = np.zeros((4, 4), int)
    for a, b in zip(df["ref"], df["landcover"]):
        if a in (1, 2, 3, 4) and b in (1, 2, 3, 4):
            cm[int(a) - 1, int(b) - 1] += 1
    return cm


def mode_filter3(L, valid=None):
    """3 x 3 majority filter on classes 1-4 (ties keep the original class)."""
    valid = (L > 0) if valid is None else valid
    counts = np.stack([ndimage.uniform_filter(((L == k) & valid).astype(np.float32), 3, mode="constant")
                       for k in (1, 2, 3, 4)])
    best = counts.argmax(0) + 1
    top = counts.max(0)
    tie = (counts == top).sum(0) > 1
    out = np.where(tie, L, best).astype(L.dtype)
    return np.where(valid, out, L)


def spectral_from_stack(stack_path, like_profile):
    """NDVI and SWIR1 reflectance from a GEE ClassStack export, resampled to the LULC grid."""
    import rasterio
    from rasterio.warp import reproject, Resampling
    with rasterio.open(stack_path) as s:
        names = [str(d).lower() if d else "" for d in s.descriptions]
        print("bands in", os.path.basename(stack_path), names)

        def band(*cands):
            for c in cands:
                if c in names:
                    a = s.read(names.index(c) + 1).astype(np.float32)
                    dst = np.full((like_profile["height"], like_profile["width"]), np.nan, np.float32)
                    reproject(a, dst, src_transform=s.transform, src_crs=s.crs, dst_transform=like_profile["transform"],
                              dst_crs=like_profile["crs"], resampling=Resampling.nearest)
                    return dst
            return None
        ndvi = band("ndvi")
        if ndvi is None:
            red, nir = band("red", "sr_b4", "b4", "sr_b3", "b3"), band("nir", "sr_b5", "b5", "sr_b4")
            ndvi = (nir - red) / (nir + red + 1e-9) if red is not None and nir is not None else None
        swir1 = band("swir1", "sr_b6", "b6", "sr_b5", "b5")
    if swir1 is not None and np.nanmedian(swir1) > 1.5:          # raw C2 L2 digital numbers -> reflectance
        swir1 = swir1 * 2.75e-05 - 0.2
    return ndvi, swir1


def make_automated(raw, ndvi, swir1, valid=None, min_veg_patch=50):
    """P2 = majority filter + the two spectral-contextual rules described in the manuscript."""
    valid = (raw > 0) if valid is None else valid
    P = mode_filter3(raw, valid)
    lab, n = ndimage.label(P == 2, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel())
    coherent = (sizes[lab] >= min_veg_patch) & (lab > 0)
    r1 = (P == 2) & (ndvi < 0.20) & ~coherent & valid
    P[r1] = 3
    near_urban = ndimage.maximum_filter((P == 1).astype(np.uint8), size=3) > 0
    r2 = (P == 3) & (ndvi < 0.10) & (swir1 > 0.18) & near_urban & valid
    P[r2] = 1
    return P, {"rule1_veg_to_bare_px": int(r1.sum()), "rule2_bare_to_urban_px": int(r2.sum())}


# =====================================================================================================
# 3. MODELLING SET-UP
# =====================================================================================================
class Study:
    def __init__(self, D=D_DEFAULT, files=PAPER, alt_maps=None):
        self.D = D
        self.P = {y: read(D + f)[0] for y, f in files.items()}
        self.profile = read(D + files[2020])[1]
        self.mask = np.logical_and.reduce([np.isin(self.P[y], [1, 2, 3, 4]) for y in self.P])
        self.alt = alt_maps or {}
        self.FEAT = {y: self.feats(self.P[y]) for y in self.P}
        self.WIN = {y: self.window(self.FEAT[y]) for y in self.P}
        print(f"Modelling mask: {self.mask.sum() * PX_KM2:.2f} km²")

    def feats(self, L):
        mask = self.mask

        def dens(b, s):
            b = (b & mask).astype(np.float32)
            return (ndimage.uniform_filter(b, s, mode="constant") * s * s - b) / (s * s - 1)

        def dist(c):
            return np.minimum(ndimage.distance_transform_edt(~((L == c) & mask)) * 0.03, 10) / 10
        return np.stack([dens(L == 1, 3), dens(L == 1, 9), dens(L == 2, 9), dens(L == 3, 9),
                         dist(1), dist(4)], -1).astype(np.float32)

    @staticmethod
    def window(F):
        from numpy.lib.stride_tricks import sliding_window_view
        return sliding_window_view(np.pad(F, ((HK, HK), (HK, HK), (0, 0))), (K, K), axis=(0, 1))

    def X_cnn(self, t0, idx):
        r, c = np.unravel_index(idx, self.mask.shape)
        return np.ascontiguousarray(self.WIN[t0][r, c].transpose(0, 2, 3, 1))

    def X_seq(self, t0, idx):
        return np.stack([self.X_cnn(y, idx) for y in (t0 - 10, t0 - 5, t0)], 1)

    def X_centre(self, t0, idx):
        return self.FEAT[t0].reshape(-1, self.FEAT[t0].shape[-1])[idx]

    def blocks(self, block=67, frac=0.2, seed=42):
        H, W_ = self.mask.shape
        blk = (np.arange(H)[:, None] // block) * 10000 + (np.arange(W_)[None, :] // block)
        ids = np.unique(blk[self.mask])
        is_val = np.isin(blk, np.random.default_rng(seed).choice(ids, int(frac * len(ids)), replace=False))
        return blk, is_val

    def buffer_regions(self, is_val):
        """Cells whose 9x9 window contains no cell of the other subset (four-cell buffer)."""
        near_val = ndimage.maximum_filter(is_val & self.mask, size=K)
        near_train = ndimage.maximum_filter(~is_val & self.mask, size=K)
        return ~is_val & ~near_val, is_val & ~near_train

    def boundary_share(self, is_val, pairs):
        """Share of training candidate cells (non-urban at t0) within four cells of a tuning block."""
        near_val = ndimage.maximum_filter(is_val & self.mask, size=K)
        tot = aff = 0
        for a, _ in pairs:
            c = (self.P[a] != 1) & self.mask & ~is_val
            tot += c.sum(); aff += (c & near_val).sum()
        return 100 * aff / tot

    def sample_idx(self, t0, t1, region, rng, max_pos=15000, neg_ratio=1, exclude=None):
        c_ = (self.P[t0] != 1) & self.mask & region
        if exclude is not None:
            c_ &= ~exclude
        pos = np.flatnonzero(c_ & (self.P[t1] == 1))
        neg = np.flatnonzero(c_ & (self.P[t1] != 1))
        n = min(len(pos), max_pos)
        nn = min(len(neg), int(round(n * neg_ratio)))
        idx = np.concatenate([rng.choice(pos, n, replace=False), rng.choice(neg, nn, replace=False)])
        return idx, np.r_[np.ones(n), np.zeros(nn)].astype(np.float32)

    def build(self, pairs, region, xfun, rng, **kw):
        X, y, I = [], [], []
        for a, b in pairs:
            i, t = self.sample_idx(a, b, region, rng, **kw)
            X.append(xfun(a, i)); y.append(t); I.append(i)
        return np.concatenate(X), np.concatenate(y), np.concatenate(I)

    def base_rate(self, pairs):
        """Urban gain as % of candidate cells per transition."""
        return {f"{a}-{b}": round(100 * (((self.P[a] != 1) & (self.P[b] == 1) & self.mask).sum() /
                                         ((self.P[a] != 1) & self.mask).sum()), 2) for a, b in pairs}


# =====================================================================================================
# 4. NETWORKS (identical architectures to the notebook)
# =====================================================================================================
def _keras():
    import tensorflow as tf
    return tf, tf.keras.layers


def _compile(m):
    tf, _ = _keras()
    m.compile(tf.keras.optimizers.Adam(1e-3), "binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="auc")])
    return m


def _cbr(x, f):
    _, L = _keras()
    return L.ReLU()(L.BatchNormalization()(L.Conv2D(f, 3, padding="same")(x)))


def make_cnn(C=6):
    tf, L = _keras()
    i = tf.keras.Input((K, K, C)); x = i
    for f in (32, 64, 128):
        x = _cbr(x, f)
    x = L.Dropout(0.3)(L.Dense(64, activation="relu")(L.GlobalAveragePooling2D()(x)))
    return _compile(tf.keras.Model(i, L.Dense(1, activation="sigmoid")(x)))


def make_unet(C=6):
    tf, L = _keras()

    def gate(skip, g, f):
        a = L.ReLU()(L.Add()([L.Conv2D(f, 1)(skip), L.Conv2D(f, 1)(g)]))
        return L.Multiply()([skip, L.Conv2D(1, 1, activation="sigmoid")(a)])
    i = tf.keras.Input((K, K, C))
    x = L.ZeroPadding2D(((3, 4), (3, 4)))(i)
    s1 = _cbr(_cbr(x, 32), 32); x = L.MaxPooling2D()(s1)
    s2 = _cbr(_cbr(x, 64), 64); x = L.MaxPooling2D()(s2)
    s3 = _cbr(_cbr(x, 128), 128); x = L.MaxPooling2D()(s3)
    x = L.Dropout(0.3)(_cbr(_cbr(x, 256), 256))
    for s, f in ((s3, 128), (s2, 64), (s1, 32)):
        x = L.Conv2DTranspose(f, 2, strides=2, padding="same")(x)
        x = _cbr(_cbr(L.Concatenate()([x, gate(s, x, f // 2)]), f), f)
    o = L.Conv2D(1, 1, activation="sigmoid")(x)
    o = L.Flatten()(L.Cropping2D(((7, 8), (7, 8)))(o))
    return _compile(tf.keras.Model(i, o))


def make_lstm(C=6, T=3):
    tf, L = _keras()
    ei = tf.keras.Input((K, K, C)); e = _cbr(_cbr(ei, 32), 64)
    enc = tf.keras.Model(ei, L.GlobalAveragePooling2D()(e))
    i = tf.keras.Input((T, K, K, C))
    seq = L.TimeDistributed(enc)(i)
    x = L.Bidirectional(L.LSTM(128))(seq)
    x = L.Concatenate()([x, L.Lambda(lambda t: t[:, -1])(seq)])
    x = L.Dropout(0.3)(L.Dense(64, activation="relu")(x))
    return _compile(tf.keras.Model(i, L.Dense(1, activation="sigmoid")(x)))


MAKERS = {"CA-CNN": make_cnn, "CA-U-Net": make_unet, "CA-LSTM-CNN": make_lstm}


def count_params(model):
    return int(sum(np.prod(w.shape) for w in model.trainable_weights))


# =====================================================================================================
# 5. EVALUATION
# =====================================================================================================
def ug_scores(u_prev, u_obs, pm, mask):
    og, pg = (~u_prev & u_obs)[mask], (~u_prev & pm)[mask]
    H, M, F = (og & pg).sum(), (og & ~pg).sum(), (~og & pg).sum()
    return dict(FoM=round(H / max(H + M + F, 1), 3), hits_km2=round(H * PX_KM2, 2), misses_km2=round(M * PX_KM2, 2),
                false_alarms_km2=round(F * PX_KM2, 2))


def boot_fom(u_prev, u_obs, pm, mask, blk, n=2000, seed=0):
    og, pg = (~u_prev & u_obs)[mask], (~u_prev & pm)[mask]
    ub, bi = np.unique(blk[mask], return_inverse=True)
    c = np.stack([np.bincount(bi, (og & pg).astype(float), len(ub)), np.bincount(bi, (og & ~pg).astype(float), len(ub)),
                  np.bincount(bi, (~og & pg).astype(float), len(ub))], 1)
    draw = np.random.default_rng(seed).integers(0, len(ub), (n, len(ub)))
    s = c[draw].sum(1)
    return s[:, 0] / s.sum(1)


def _test_maps(S, test_pair, test_ref=None):
    t0, t1 = test_pair
    u0 = (S.P[t0] == 1) & S.mask
    u1 = ((test_ref if test_ref is not None else S.P[t1]) == 1) & S.mask
    cand = np.flatnonzero(~u0 & S.mask)
    return u0, u1, cand, int((~u0 & u1).sum())


def _allocate(u0, cand, prob, n_gain):
    pm = u0.copy(); pm.flat[cand[np.argsort(prob)[::-1][:n_gain]]] = True
    return pm


def predict_all(model, xfun, t0, idx, chunk=50000):
    return np.concatenate([model.predict(xfun(t0, idx[i:i + chunk]), batch_size=2048, verbose=0).ravel()
                           for i in range(0, len(idx), chunk)])


def lstm_ok(S, pair):
    return all(y in S.P for y in (pair[0] - 10, pair[0] - 5, pair[0]))


def run_network(S, name, seed=42, train=None, tune=((2015, 2020),), test=(2020, 2025), block=67, buffer=False,
                neg_ratio=1, exclude=None, test_ref=None, max_epochs=60, patience=6, max_pos=15000, verbose=0):
    """Tune on blocks -> best epoch -> refit on train+tune over the full mask -> allocate the observed test gain."""
    tf, _ = _keras()
    train = list(train or [(1990, 1995), (1995, 2000), (2000, 2005), (2005, 2010), (2010, 2015)])
    tune = list(tune)
    xfun = S.X_seq if name == "CA-LSTM-CNN" else S.X_cnn
    if name == "CA-LSTM-CNN":
        train = [p for p in train if lstm_ok(S, p)]
    blk, is_val = S.blocks(block)
    r_tr, r_va = (S.buffer_regions(is_val) if buffer else (~is_val, is_val))
    kw = dict(max_pos=max_pos, neg_ratio=neg_ratio, exclude=exclude)
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed); rng = np.random.default_rng(seed)
    Xtr, ytr, _ = S.build(train, r_tr, xfun, rng, **kw)
    Xva, yva, _ = S.build(tune, r_va, xfun, rng, **kw)
    if len(yva) < 100:
        raise ValueError(f"only {len(yva)} tuning samples (block={block}); use smaller blocks or more tuning blocks")
    m = MAKERS[name](Xtr.shape[-1])
    es = tf.keras.callbacks.EarlyStopping("val_auc", mode="max", patience=patience, restore_best_weights=True)
    t_start = time.time()
    h = m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=max_epochs, batch_size=256, callbacks=[es], verbose=verbose)
    t_tune = time.time() - t_start
    best = int(np.argmax(h.history["val_auc"])) + 1
    del Xtr, Xva
    Xall, yall, _ = S.build(train + tune, S.mask, xfun, rng, **kw)
    tf.keras.utils.set_random_seed(seed); mf = MAKERS[name](Xall.shape[-1])
    mf.fit(Xall, yall, epochs=best, batch_size=256, verbose=verbose); del Xall
    u0, u1, cand, n_gain = _test_maps(S, test, test_ref)
    t0 = time.time(); prob = predict_all(mf, xfun, test[0], cand); t_inf = time.time() - t0
    pm = _allocate(u0, cand, prob, n_gain)
    sc = ug_scores(u0, u1, pm, S.mask)
    sc.update(model=name, seed=seed, best_epoch=best, epochs_run=len(h.history["loss"]),
              s_per_epoch=round(t_tune / len(h.history["loss"]), 1), params=count_params(mf),
              inference_s_per_100k=round(100000 * t_inf / max(len(cand), 1), 2), peak_ram_gb=peak_ram_gb())
    return sc, pm, prob


def run_logistic(S, train=None, tune=((2015, 2020),), test=(2020, 2025), neg_ratio=1, exclude=None,
                 test_ref=None, block=67, buffer=False, penalty="l2", seed=42, max_pos=15000):
    """Logistic-regression CA as in the notebook (centre-cell values, train+tune transitions, full mask)."""
    from sklearn.linear_model import LogisticRegression
    train = list(train or [(1990, 1995), (1995, 2000), (2000, 2005), (2005, 2010), (2010, 2015)])
    region = S.mask
    if buffer:
        _, is_val = S.blocks(block)
        a, b = S.buffer_regions(is_val); region = a | b
    X, y, _ = S.build(train + list(tune), region, S.X_centre, np.random.default_rng(seed),
                      max_pos=max_pos, neg_ratio=neg_ratio, exclude=exclude)
    lr = LogisticRegression(max_iter=2000, **({} if penalty == "l2" else {"penalty": None})).fit(X, y)
    u0, u1, cand, n_gain = _test_maps(S, test, test_ref)
    pm = _allocate(u0, cand, lr.predict_proba(S.X_centre(test[0], cand))[:, 1], n_gain)
    sc = ug_scores(u0, u1, pm, S.mask); sc.update(model="Logistic-CA")
    return sc, pm, lr


def run_neighbourhood(S, test=(2020, 2025), test_ref=None):
    u0, u1, cand, n_gain = _test_maps(S, test, test_ref)
    dens9 = ndimage.uniform_filter(u0.astype(float), 9, mode="constant") * 81 - u0
    pm = _allocate(u0, cand, dens9.flat[cand] + 1e-6 * np.random.default_rng(0).random(len(cand)), n_gain)
    sc = ug_scores(u0, u1, pm, S.mask); sc.update(model="Neighbourhood")
    return sc, pm


def peak_ram_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)


# =====================================================================================================
# 6. ROBUSTNESS EXPERIMENTS -> Table 7
# =====================================================================================================
LOG_COLS = ["experiment", "model", "seed", "FoM", "hits_km2", "misses_km2", "false_alarms_km2", "best_epoch",
            "epochs_run", "s_per_epoch", "params", "inference_s_per_100k", "peak_ram_gb"]

EXPERIMENTS = {
    "hindcast_2015_2020": dict(train=[(1990, 1995), (1995, 2000), (2000, 2005), (2005, 2010)],
                               tune=[(2010, 2015)], test=(2015, 2020)),
    "test_ref_automated_2025": dict(test_ref="P2_2025"),
    "equal_transitions_3": dict(train=[(2000, 2005), (2005, 2010), (2010, 2015)]),
    "blocks_1km": dict(block=33),
    "blocks_4km": dict(block=133),
    "buffer_4cells": dict(buffer=True),
    "ratio_1_3": dict(neg_ratio=3),
    "ratio_1_5": dict(neg_ratio=5),
    "unstable_excluded": dict(exclude="unstable"),
}


def run_robustness(S, models=("CA-CNN", "Logistic-CA", "Neighbourhood"), experiments=None, seeds=(42,),
                   out_csv=None, p2_2025_file="Windhoek_LULC_2025_automated.tif", quick=False):
    """Runs the Table 7 experiments and appends rows to a CSV (safe to resume after a disconnect)."""
    out_csv = out_csv or S.D + "revision_runs/Table7_robustness.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    done = set()
    if os.path.exists(out_csv):
        d = pd.read_csv(out_csv); done = set(zip(d.experiment, d.model, d.seed))
    unstable = None
    for exp in (experiments or EXPERIMENTS):
        cfg = dict(EXPERIMENTS[exp])
        if cfg.get("test_ref") == "P2_2025":
            if not os.path.exists(S.D + p2_2025_file):
                print(f"skipped {exp}: {p2_2025_file} not found (run Cell 8 first)")
                continue
            cfg["test_ref"] = read(S.D + p2_2025_file)[0]
        if cfg.get("exclude") == "unstable":
            if unstable is None:
                _, unstable, _ = reversal_stats({y: S.P[y] for y in S.P if y <= 2020}, S.mask)
            cfg["exclude"] = unstable
        for name in models:
            for seed in seeds:
                if (exp, name, seed) in done:
                    continue
                if name == "Neighbourhood":
                    if exp not in ("hindcast_2015_2020", "test_ref_automated_2025"):
                        continue
                    sc, _ = run_neighbourhood(S, cfg.get("test", (2020, 2025)), cfg.get("test_ref"))
                elif name == "Logistic-CA":
                    if exp in ("blocks_1km", "blocks_4km"):
                        continue
                    kw = {k: v for k, v in cfg.items() if k in ("train", "tune", "test", "neg_ratio", "exclude",
                                                                 "test_ref", "block", "buffer")}
                    sc, _, _ = run_logistic(S, **kw)
                else:
                    kw = dict(cfg)
                    if quick:
                        kw.update(max_epochs=2, max_pos=500)
                    if name == "CA-LSTM-CNN" and exp == "equal_transitions_3":
                        continue
                    if name == "CA-LSTM-CNN" and kw.get("train") and not any(lstm_ok(S, p) for p in kw["train"]):
                        continue
                    try:
                        sc, _, _ = run_network(S, name, seed=seed, **kw)
                    except ValueError as err:
                        print(f"skipped {exp} {name}: {err}")
                        continue
                sc.update(experiment=exp, seed=seed)
                row = pd.DataFrame([sc]).reindex(columns=LOG_COLS)
                row.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
                print(exp, name, seed, sc["FoM"])
    return pd.read_csv(out_csv)


# =====================================================================================================
# 7. LOGISTIC-REGRESSION DIAGNOSTICS and CORRELOGRAM
# =====================================================================================================
FEATURE_NAMES = ["Urban density 3x3", "Urban density 9x9", "Vegetation density 9x9", "Bare density 9x9",
                 "Distance to urban", "Distance to water"]


def lr_diagnostics(S, pairs=None, n_boot=1000, block=67, seed=42):
    """Unpenalised logistic regression, 2 km block-bootstrap 95% CIs, VIF and correlation matrix."""
    from sklearn.linear_model import LogisticRegression, LinearRegression
    pairs = pairs or [(y, y + 5) for y in range(1990, 2020, 5)]
    X, y, I = S.build(pairs, S.mask, S.X_centre, np.random.default_rng(seed))
    fit = LogisticRegression(penalty=None, max_iter=5000).fit(X, y)
    vif = []
    for j in range(X.shape[1]):
        others = np.delete(X, j, axis=1)
        r2 = LinearRegression().fit(others, X[:, j]).score(others, X[:, j])
        vif.append(1 / max(1 - r2, 1e-12))
    blk, _ = S.blocks(block)
    b_of = blk.ravel()[I]
    ub, inv = np.unique(b_of, return_inverse=True)
    groups = [np.flatnonzero(inv == k) for k in range(len(ub))]
    rng = np.random.default_rng(seed)
    coefs = []
    for _ in range(n_boot):
        pick = np.concatenate([groups[k] for k in rng.integers(0, len(ub), len(ub))])
        coefs.append(LogisticRegression(penalty=None, max_iter=5000).fit(X[pick], y[pick]).coef_[0])
    lo, hi = np.percentile(np.array(coefs), [2.5, 97.5], axis=0)
    tab = pd.DataFrame({"coef_unpenalised": fit.coef_[0], "ci_lo": lo, "ci_hi": hi, "VIF": vif},
                       index=FEATURE_NAMES).round(3)
    tab["odds_ratio_typical_change"] = [np.exp(0.1 * c) if "density" in n else np.exp(-0.1 * c)
                                        for n, c in zip(FEATURE_NAMES, fit.coef_[0])]
    corr = pd.DataFrame(np.corrcoef(X.T), index=FEATURE_NAMES, columns=FEATURE_NAMES).round(2)
    return tab.round(3), corr


def correlogram(S, lr_model, t0=2015, t1=2020, dists_km=(0.09, 0.15, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6),
                n_pairs=200000, seed=0):
    """Lagged correlation of logistic-CA residuals by distance (Figure S3)."""
    rng = np.random.default_rng(seed)
    cand = (S.P[t0] != 1) & S.mask
    prob = np.full(S.mask.shape, np.nan, np.float32)
    idx = np.flatnonzero(cand)
    prob.flat[idx] = lr_model.predict_proba(S.X_centre(t0, idx))[:, 1]
    res = np.where(cand, (S.P[t1] == 1).astype(np.float32) - prob, np.nan)
    H, W_ = res.shape
    rows = []
    for d in dists_km:
        i = rng.choice(idx, n_pairs)
        r, c = np.unravel_index(i, res.shape)
        th = rng.uniform(0, 2 * np.pi, n_pairs)
        r2 = np.round(r + d / 0.03 * np.sin(th)).astype(int); c2 = np.round(c + d / 0.03 * np.cos(th)).astype(int)
        ok = (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W_)
        a, b = res[r[ok], c[ok]], res[r2[ok], c2[ok]]
        ok2 = ~np.isnan(a) & ~np.isnan(b)
        rows.append({"distance_km": d, "correlation": np.corrcoef(a[ok2], b[ok2])[0, 1], "n_pairs": int(ok2.sum())})
    return pd.DataFrame(rows).round(3)


# =====================================================================================================
# 8. DEMAND SCENARIOS AND SOCIO-SPATIAL INDICATORS
# =====================================================================================================
def demand_rates(S, pop_2025=519617, pop_2050=None):
    """Trend (low/central/high), recent-rate (2020-2025 net) and population-driven rates (km² per year)."""
    from scipy.stats import t as tdist
    Y = np.array(sorted(S.P)); A = np.array([((S.P[y] == 1) & S.mask).sum() * PX_KM2 for y in Y])
    b, a0 = np.polyfit(Y, A, 1)
    se = np.sqrt(((A - (a0 + b * Y)) ** 2).sum() / (len(Y) - 2) / ((Y - Y.mean()) ** 2).sum())
    tq = tdist.ppf(0.975, len(Y) - 2)
    rates = {"trend_low": b - tq * se, "trend_central": b, "trend_high": b + tq * se,
             "recent_2020_2025": (A[-1] - A[-2]) / 5}
    if pop_2050:
        per_cap = A[-1] / pop_2025
        rates["population_driven"] = per_cap * (pop_2050 - pop_2025) / 25
    r2 = 1 - ((A - (a0 + b * Y)) ** 2).sum() / ((A - A.mean()) ** 2).sum()
    return {k: round(v, 3) for k, v in rates.items()}, round(r2, 3)


def train_final(S, name, epochs, seed=42):
    """Refit on all transitions to 2025 for scenario simulation."""
    tf, _ = _keras()
    xfun = S.X_seq if name == "CA-LSTM-CNN" else S.X_cnn
    pairs = [(y, y + 5) for y in range(1990, 2025, 5)]
    if name == "CA-LSTM-CNN":
        pairs = [p for p in pairs if lstm_ok(S, p)]
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
    X, y, _ = S.build(pairs, S.mask, xfun, np.random.default_rng(seed))
    m = MAKERS[name](X.shape[-1]); m.fit(X, y, epochs=epochs, batch_size=256, verbose=0)
    return m, xfun


def simulate(S, model, xfun, rate_km2_yr, start=2025, steps=(2030, 2035, 2040, 2045, 2050), save_prefix=None):
    """Recursive demand-constrained allocation; predictors recomputed after each step."""
    need = int(round(rate_km2_yr * 5 / PX_KM2))
    cur, out = S.P[start].copy(), {}
    FEAT0, WIN0 = dict(S.FEAT), dict(S.WIN)
    for t in steps:
        c_idx = np.flatnonzero((cur != 1) & (cur != 4) & S.mask)
        prob = predict_all(model, xfun, t - 5, c_idx)
        new = cur.copy(); new.flat[c_idx[np.argsort(prob)[::-1][:need]]] = 1
        S.FEAT[t] = S.feats(new); S.WIN[t] = S.window(S.FEAT[t]); S.P[t] = new
        out[t] = new; cur = new
        if save_prefix:
            import rasterio
            prof = S.profile.copy(); prof.update(dtype="uint8", nodata=0, count=1, compress="lzw")
            with rasterio.open(f"{save_prefix}_{t}.tif", "w", **prof) as o:
                o.write(np.where(S.mask, new, 0).astype("uint8"), 1)
    for t in steps:
        S.P.pop(t, None)
    S.FEAT, S.WIN = FEAT0, WIN0
    return out


def distance_to_cbd(S, sim_2050, cbd_xy=(712500, 7503000)):
    """Mean distance (km) of simulated new urban land and of 2025 urban land from the CBD."""
    t = S.profile["transform"]
    rr, cc = np.indices(S.mask.shape)
    x = t.c + (cc + 0.5) * t.a; y = t.f + (rr + 0.5) * t.e
    d = np.hypot(x - cbd_xy[0], y - cbd_xy[1]) / 1000
    u25 = (S.P[2025] == 1) & S.mask
    new = (sim_2050 == 1) & S.mask & ~u25
    return {"new_urban_mean_km": round(float(d[new].mean()), 2), "urban_2025_mean_km": round(float(d[u25].mean()), 2)}


def scenario_overlap(sim_a, sim_b, u25, mask):
    """Share of scenario A's new urban land also simulated in scenario B (%)."""
    a = (sim_a == 1) & mask & ~u25; b = (sim_b == 1) & mask & ~u25
    return round(100 * (a & b).sum() / max(a.sum(), 1), 1)
