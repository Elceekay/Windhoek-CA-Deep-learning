"""make_figures.py - map figures to Reviewer 2's specifications (no north arrow, 5 km scale bar, 600 dpi)."""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.5,
                            "xtick.labelsize": 6, "ytick.labelsize": 6, "xtick.major.width": 0.5,
                            "ytick.major.width": 0.5})
W_MM = 174
LULC_COLS = ["#d7301f", "#33a02c", "#f4a259", "#1f78b4"]
LULC_NAMES = ["Urban", "Vegetation", "Bare Ground", "Water"]
PLACES = {"Wanaheda": (709800, 7508400), "Katutura": (712000, 7507400), "Goreangab Dam": (707000, 7505500),
          "CBD": (713100, 7503000), "Klein Windhoek": (716800, 7502000), "Academia": (711500, 7497900)}


def _out(S, name):
    d = os.path.join(S.D, "revision_figures"); os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def _crop(S, pad=10):
    rr, cc = np.where(S.mask)
    r0, r1 = max(rr.min() - pad, 0), min(rr.max() + pad, S.mask.shape[0] - 1)
    c0, c1 = max(cc.min() - pad, 0), min(cc.max() + pad, S.mask.shape[1] - 1)
    t = S.profile["transform"]
    return (slice(r0, r1 + 1), slice(c0, c1 + 1)), [t.c + c0 * t.a, t.c + (c1 + 1) * t.a,
                                                    t.f + (r1 + 1) * t.e, t.f + r0 * t.e]


def _style_axes(ax, ext, left=True, bottom=True):
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    step = 10000
    xs = np.arange(np.ceil(ext[0] / step) * step, ext[1], step)
    ys = np.arange(np.ceil(ext[2] / step) * step, ext[3], step)
    ax.set_xticks(xs); ax.set_yticks(ys)
    ax.set_xticklabels([f"{x / 1000:.0f}" for x in xs] if bottom else [])
    ax.set_yticklabels([f"{y / 1000:.0f}" for y in ys] if left else [], rotation=90, va="center")
    ax.tick_params(length=2, pad=1)
    ax.set_aspect("equal")


def _scalebar(ax, ext, km=5):
    x0 = ext[1] - (ext[1] - ext[0]) * 0.05 - km * 1000
    y0 = ext[2] + (ext[3] - ext[2]) * 0.06
    ax.plot([x0, x0 + km * 1000], [y0, y0], color="black", lw=1.5, solid_capstyle="butt")
    ax.text(x0 + km * 500, y0 + (ext[3] - ext[2]) * 0.02, f"{km} km", ha="center", va="bottom", fontsize=6)


def _save(fig, path):
    fig.savefig(path + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(path + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", path + ".png / .pdf")


def _lulc_img(S, L, crop):
    img = np.where(S.mask, L, 0).astype(float)
    img[~S.mask] = np.nan
    return img[crop]


def fig3_lulc_grid(S, years=(1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025), labels=True):
    crop, ext = _crop(S)
    fig, axes = plt.subplots(3, 3, figsize=(W_MM / 25.4, W_MM / 25.4 * 1.02))
    cmap = ListedColormap(LULC_COLS)
    for k, ax in enumerate(axes.flat):
        if k == 8:
            ax.axis("off")
            ax.legend(handles=[Patch(color=c, label=n) for c, n in zip(LULC_COLS, LULC_NAMES)], loc="center",
                      frameon=False, fontsize=7, title="Land use/land cover", title_fontsize=7)
            ax.text(0.5, 0.12, "WGS 84 / UTM zone 33S\ncoordinates in km", transform=ax.transAxes,
                    ha="center", fontsize=6)
            continue
        y = years[k]
        ax.imshow(_lulc_img(S, S.P[y], crop), cmap=cmap, vmin=0.5, vmax=4.5, extent=ext, interpolation="nearest")
        _style_axes(ax, ext, left=(k % 3 == 0), bottom=(k >= 5))
        ax.set_title(f"({'abcdefgh'[k]}) {y}", fontsize=7, loc="left", pad=2)
        if k == 7:
            _scalebar(ax, ext)
        if labels and k == 0:
            for name, (x, yy) in PLACES.items():
                ax.text(x, yy, name, fontsize=4.5, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
    fig.subplots_adjust(wspace=0.05, hspace=0.12)
    _save(fig, _out(S, "Figure3_LULC_1990_2025"))


def fig4_urban_1990_2025(S):
    crop, ext = _crop(S)
    fig, axes = plt.subplots(1, 2, figsize=(W_MM / 25.4, W_MM / 25.4 * 0.5))
    for k, (ax, y) in enumerate(zip(axes, (1990, 2025))):
        img = np.where(S.mask, np.where(S.P[y] == 1, 2, 1), np.nan)[crop]
        ax.imshow(img, cmap=ListedColormap(["#eeeeee", "#d7301f"]), vmin=0.5, vmax=2.5, extent=ext,
                  interpolation="nearest")
        a = ((S.P[y] == 1) & S.mask).sum() * 0.0009
        ax.set_title(f"({'ab'[k]}) {y}: {a:.1f} km² urban", fontsize=7, loc="left", pad=2)
        _style_axes(ax, ext, left=(k == 0))
    _scalebar(axes[1], ext)
    fig.legend(handles=[Patch(color="#d7301f", label="Urban"), Patch(color="#eeeeee", label="Non-urban")],
               loc="lower center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.04))
    _save(fig, _out(S, "Figure4_urban_1990_2025"))


def fig5_hits(S, pred_file="revision_runs/Pred_urban_2025_CA-CNN_20101_s42.tif"):
    import revision_analyses as ra
    pred = ra.read(S.D + pred_file)[0] == 2
    crop, ext = _crop(S)
    u20, u25 = (S.P[2020] == 1) & S.mask, (S.P[2025] == 1) & S.mask
    og, pg = ~u20 & u25, ~u20 & pred & S.mask
    img = np.full(S.mask.shape, np.nan); img[S.mask] = 0; img[u20] = 1
    img[og & pg] = 2; img[og & ~pg] = 3; img[~og & pg] = 4
    cols = ["#f2f2f2", "#9e9e9e", "#1a9850", "#d73027", "#4575b4"]
    labs = ["Non-urban", "Urban 2020", "Hit", "Miss", "False alarm"]
    fig, ax = plt.subplots(figsize=(W_MM / 25.4 * 0.62, W_MM / 25.4 * 0.62))
    ax.imshow(img[crop], cmap=ListedColormap(cols), vmin=-0.5, vmax=4.5, extent=ext, interpolation="nearest")
    _style_axes(ax, ext); _scalebar(ax, ext)
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(cols, labs)], loc="upper left", frameon=False,
              fontsize=6)
    _save(fig, _out(S, "Figure5_hits_misses"))


def fig6_simulations(S, models=("CA-CNN", "CA-U-Net", "CA-LSTM-CNN"), years=(2030, 2040, 2050),
                     pattern="revision_runs/Proj_{m}_{y}_central.tif", scenario="central trend scenario"):
    import revision_analyses as ra
    crop, ext = _crop(S)
    u25 = (S.P[2025] == 1) & S.mask
    fig, axes = plt.subplots(3, 3, figsize=(W_MM / 25.4, W_MM / 25.4 * 0.98))
    cols = ["#f2f2f2", "#7f7f7f", "#b30000"]
    for i, m in enumerate(models):
        for j, y in enumerate(years):
            ax = axes[i, j]
            sim = ra.read(S.D + pattern.format(m=m, y=y))[0] == 1
            img = np.full(S.mask.shape, np.nan); img[S.mask] = 0; img[u25] = 1; img[sim & S.mask & ~u25] = 2
            ax.imshow(img[crop], cmap=ListedColormap(cols), vmin=-0.5, vmax=2.5, extent=ext, interpolation="nearest")
            _style_axes(ax, ext, left=(j == 0), bottom=(i == 2))
            ax.set_title(f"({'abcdefghi'[3 * i + j]}) {y}", fontsize=7, loc="left", pad=2)
            if j == 0:
                ax.set_ylabel(m, fontsize=7, fontweight="bold")
    _scalebar(axes[2, 2], ext)
    fig.legend(handles=[Patch(color=cols[1], label="Urban 2025"),
                        Patch(color=cols[2], label=f"Simulated new urban land ({scenario})")],
               loc="lower center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(wspace=0.05, hspace=0.12, bottom=0.07)
    _save(fig, _out(S, "Figure6_simulations"))


def fig7_agreement(S, models=("CA-CNN", "CA-U-Net", "CA-LSTM-CNN"),
                   pattern="revision_runs/Proj_{m}_2050_central.tif", rates=None):
    import revision_analyses as ra
    crop, ext = _crop(S)
    u25 = (S.P[2025] == 1) & S.mask
    gain = [(ra.read(S.D + pattern.format(m=m))[0] == 1) & S.mask & ~u25 for m in models]
    cons = np.sum(gain, 0)
    img = np.full(S.mask.shape, np.nan); img[S.mask] = 0; img[u25] = 1
    for k in (1, 2, 3):
        img[cons == k] = 1 + k
    cols = ["#f2f2f2", "#7f7f7f", "#fdd49e", "#fc8d59", "#b30000"]
    labs = ["Non-urban", "Urban 2025", "New urban: 1 model", "New urban: 2 models", "New urban: 3 models"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W_MM / 25.4, W_MM / 25.4 * 0.47),
                               gridspec_kw={"width_ratios": [1.15, 1]})
    a.imshow(img[crop], cmap=ListedColormap(cols), vmin=-0.5, vmax=4.5, extent=ext, interpolation="nearest")
    _style_axes(a, ext); _scalebar(a, ext)
    a.set_title("(a) Agreement on simulated gain, 2025–2050", fontsize=7, loc="left", pad=2)
    a.legend(handles=[Patch(color=c, label=l) for c, l in zip(cols, labs)], loc="upper left", fontsize=5.5,
             frameon=False)
    Y = np.array(sorted(y for y in S.P if y <= 2025))
    A = np.array([((S.P[y] == 1) & S.mask).sum() * 0.0009 for y in Y])
    rates = rates or ra.demand_rates(S)[0]
    bb, a0 = np.polyfit(Y, A, 1)
    b.plot(Y, A, "o", color="black", ms=3, label="Observed (corrected maps)")
    yy = np.arange(1990, 2051)
    b.plot(yy, a0 + bb * yy, ":", color="grey", lw=0.8, label="Linear trend 1990–2025")
    fut = np.arange(2025, 2051, 5)
    for k, st, c in (("trend_low", "--", "#4575b4"), ("trend_central", "-", "#b30000"), ("trend_high", "--", "#d73027"),
                     ("recent_2020_2025", "-.", "#7b3294")):
        if k in rates:
            b.plot(fut, A[-1] + rates[k] * (fut - 2025), st, color=c, lw=1, label=k.replace("_", " "))
    b.set_xlabel("Year"); b.set_ylabel("Urban area (km²)")
    b.legend(fontsize=5.5, frameon=False); b.set_title("(b) Observed and scenario urban area", fontsize=7, loc="left")
    _save(fig, _out(S, "Figure7_agreement_scenarios"))


def fig1_study_area(S, namibia_regions=None):
    crop, ext = _crop(S)
    fig = plt.figure(figsize=(W_MM / 25.4, W_MM / 25.4 * 0.62))
    ax = fig.add_axes([0.05, 0.08, 0.62, 0.88])
    img = np.where(S.mask, np.where(S.P[2025] == 1, 2, 1), np.nan)[crop]
    ax.imshow(img, cmap=ListedColormap(["#e8e3d3", "#d7301f"]), vmin=0.5, vmax=2.5, extent=ext,
              interpolation="nearest")
    for name, (x, y) in PLACES.items():
        ax.text(x, y, name, fontsize=6, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    _style_axes(ax, ext); _scalebar(ax, ext)
    ax.set_xlabel("Easting (km)"); ax.set_ylabel("Northing (km)")
    ax.legend(handles=[Patch(color="#d7301f", label="Urban land, 2025"),
                       Patch(color="#e8e3d3", label="Municipal area (study area)")], loc="upper left",
              fontsize=6, frameon=False)
    if namibia_regions:
        import geopandas as gpd
        inset = fig.add_axes([0.70, 0.30, 0.28, 0.60])
        g = gpd.read_file(namibia_regions).to_crs(4326)
        g.plot(ax=inset, color="#f0f0f0", edgecolor="#555555", linewidth=0.3)
        inset.plot(17.08, -22.57, "s", color="#d7301f", ms=4)
        inset.text(17.5, -22.57, "Windhoek", fontsize=6, va="center")
        inset.set_title("Namibia (regions)", fontsize=7); inset.tick_params(labelsize=5)
    _save(fig, _out(S, "Figure1_study_area"))


def figS3_correlogram(S, df):
    fig, ax = plt.subplots(figsize=(W_MM / 25.4 * 0.55, W_MM / 25.4 * 0.38))
    ax.plot(df.distance_km, df.correlation, "o-", color="black", ms=3, lw=0.8)
    ax.axhline(0, color="grey", lw=0.5); ax.axvline(2, color="#b30000", ls="--", lw=0.8)
    ax.text(2.05, ax.get_ylim()[1] * 0.9, "2 km block", color="#b30000", fontsize=6)
    ax.set_xlabel("Distance (km)"); ax.set_ylabel("Residual correlation")
    _save(fig, _out(S, "FigureS3_correlogram"))
