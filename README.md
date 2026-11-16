# Windhoek urban growth: cellular automata and deep learning (version v3)

Code, data and results for the manuscript (archived on Zenodo; the DOI is given in the paper's Data Availability statement):

> Ncube, E. C., & Ajayi, O. G. *Integrated Cellular Automata–Deep Learning Modelling of Urban Growth in Windhoek, Namibia* (under review).

Version v3 contains every analysis reported in the revised manuscript and its Supplementary Material, including the robustness analyses (Tables S12–S14).

## Contents

| File / folder | What it holds |
|---|---|
| `revision_analyses.py` | All analyses: map statistics, accuracy assessment, CA–deep learning models, benchmarks, evaluation, sensitivity tests and scenarios |
| `make_figures.py` | Figures of the manuscript and Supplementary Material |
| `results/notebook.ipynb` | The Google Colab notebook with all cells, in run order |
| `results/Windhoek_LULC_<year>_….tif` | Final land use/land cover maps (P3), 1990–2025: `1990_corrected_v3`, `1995_corrected_v2`, `2000_corrected`, `2005_corrected`, `2010_Final`, `2015_Final`, `2020_Final`, `2025_Final_FIXED` |
| `maps_automated/Windhoek_LULC_<year>_automated.tif` | Automated maps before manual editing (P2), 1990–2025 |
| `manual_edits_P2_to_P3.gpkg` | Polygons of every cell whose class differs between the automated (P2) and final (P3) map, with year, original class, corrected class and area |
| `validation_points.zip` | Original reference points of each epoch (`ValidationPoints_<year>`, as shapefile and CSV) |
| `Independent2025_KEY.csv`, `Independent2025_labels.xlsx`, `Independent2025_NUMBERED.kml` | The independently drawn 2025 sample: point key with map classes, reference labels, and the points for viewing in Google Earth |
| `results/tables/` | Every table of the manuscript (Table1–Table8) and the Supplementary Material (TableS1–TableS14) as CSV; the first line of each file is the table caption |
| `results/figures/` | Every figure of the manuscript (Figure1–Figure8) and the Supplementary Material (FigureS1–FigureS3) as PNG; `figure_captions.txt` gives each caption |
| `requirements.txt` | Package versions of the Google Colab environment at release (see the notes at the top of the file) |

**Maps:** GeoTIFF, 30 m, WGS 84 / UTM zone 33S. Class codes: 1 Urban, 2 Vegetation, 3 Bare Ground, 4 Water, 0 no data.

## Reproducing the analyses

1. Create a Google Drive folder named `GEE_Windhoek` and place in it, side by side: the two `.py` files, all 16 maps (final and automated), the unzipped contents of `validation_points.zip` and the three `Independent2025` files. The code expects all of these in that one folder.
2. Open `results/notebook.ipynb` in Google Colab and run the cells in order. Each session starts with the cell that mounts Drive and the cell that loads the code and maps.
3. All networks use fixed random seeds (42, 1, 2 and 3). Trained model weights are not archived; retraining with the released code regenerates them.

Models were trained on Google Colab standard CPU runtimes, except the three CA-U-Net runs of the main-test robustness experiments (Table S12), which used a T4 GPU runtime.

## Licence

See the licence stated on the Zenodo record.
