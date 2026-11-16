# Windhoek urban growth: cellular automata and deep learning (release v2.0.0)

Code, data and results for the manuscript:

> Ncube, E. C., & Ajayi, O. G. *Integrated Cellular Automata–Deep Learning Modelling of Urban Growth in Windhoek, Namibia* (under review).

Version 2.0.0 contains every analysis reported in the revised manuscript and its Supplementary Material, including the second-round robustness analyses (Tables S12–S14).

## Contents

| Folder / file | What it holds |
|---|---|
| `code/revision_analyses.py` | All analyses: map statistics, accuracy assessment, CA–deep learning models, benchmarks, evaluation, sensitivity tests and scenarios |
| `code/make_figures.py` | Figures of the manuscript and supplement |
| `code/notebook.ipynb` | The Google Colab notebook with all cells, in run order |
| `data/maps_final/` | Final land use/land cover maps (P3), 1990–2025 |
| `data/maps_automated/` | Automated maps before manual editing (P2), 1990–2025 |
| `data/manual_edits/` | `manual_edits_P2_to_P3.gpkg`: polygons of every cell whose class differs between the automated (P2) and final (P3) map, with year, original class, corrected class and area; `manual_edits_summary.csv`: changed area by year and class pair |
| `data/reference/` | Original reference points (`ValidationPoints_<year>`) and the independent 2025 sample (`Independent2025_KEY.csv`, `Independent2025_labels.xlsx`, and `Independent2025_NUMBERED.kml` for viewing the points in Google Earth) |
| `results/tables/` | Every table of the manuscript (Table1–Table8) and the Supplementary Material (TableS1–TableS14) as CSV; the first line of each file is the table caption |
| `results/figures/` | Every figure of the manuscript (Figure1–Figure8) and the Supplementary Material (FigureS1–FigureS3) as PNG; `figure_captions.txt` gives each caption |
| `requirements.txt` | Package versions of the Google Colab environment at release (see the notes at the top of the file) |

**Maps:** GeoTIFF, 30 m, WGS 84 / UTM zone 33S. Class codes: 1 Urban, 2 Vegetation, 3 Bare Ground, 4 Water, 0 no data.

## Reproducing the analyses

1. Copy `code/` and `data/` into a Google Drive folder named `GEE_Windhoek`.
2. Open `code/notebook.ipynb` in Google Colab and run the cells in order. Each session starts with the cell that mounts Drive and the cell that loads the code and maps.
3. All networks use fixed random seeds (42, 1, 2 and 3). Trained model weights are not archived; retraining with the released code regenerates them.

Models were trained on Google Colab standard CPU runtimes, except the three CA-U-Net runs of the main-test robustness experiments (Table S12), which used a T4 GPU runtime.

## Licence

TODO: choose a licence (for example MIT for the code and CC BY 4.0 for the data) and add a `LICENSE` file.
