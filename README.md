# Appliance Energy Forecasting — Time Series Case Study

7PAM2033 coursework: a time-series case study forecasting household
appliance energy use, comparing five benchmark methods, SARIMAX, a
feature-based gradient-boosting model, and a zero-shot time-series
foundation model.

## Project aim

Forecast short-term household appliance energy use and evaluate whether
increasingly complex models improve on simple benchmark methods. The
project addresses six questions, answered in the report:

1. Which benchmark model is strongest, and what does this reveal about
   appliance energy use?
2. Does SARIMAX improve on the strongest benchmark?
3. Does the feature-based model improve when lag, rolling, time, sensor,
   and weather features are added, and which feature groups help most?
4. Does the foundation model outperform the simpler models?
5. Which covariates are genuinely known at the forecast origin?
6. Which model would be most suitable for practical smart-home energy
   forecasting?

## Dataset

[UCI Appliances Energy Prediction dataset](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)
(Candanedo, Feldheim & Deramaix, 2017) — appliance energy use, indoor
temperature/humidity sensors, outdoor weather, sampled every 10 minutes
over ~4.5 months. Fetched automatically at runtime (see below); no need
to download it manually.

## Forecasting setup

- **Target:** `Appliances` — total energy use in Wh, resampled to hourly
  (summed across each hour's six 10-minute readings; sensor/weather
  columns are averaged instead — see Part 1).
- **Horizon:** 24 hours.
- **Train/test split:** chronological hold-out, final 14 days as test.
- **Metrics:** MAE, RMSE, MASE (scaled against a 24h seasonal-naive
  baseline), and Bias.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── part1_data_prep_eda.py        # data loading, cleaning, EDA
├── part2_problem_definition.py   # shared constants + evaluation metrics
├── part3_benchmark_models.py     # mean / naive / seasonal-naive / drift
├── part4_sarimax.py              # SARIMAX grid search, fit, forecast
├── part5_feature_engineering.py  # time/lag/rolling feature table
├── part6_feature_model.py        # XGBoost, tuning, ablation
├── part7_foundation_model.py     # Chronos-Bolt, zero-shot
├── notebooks/                    # Colab-ready notebook per part, 01-08
├── tests/                        # pytest suite (see below)
├── data/                         # generated at runtime, not committed
└── outputs/
    ├── figures/                  # committed — small, are the results
    ├── metrics/                  # committed
    ├── forecasts/                # committed
    └── model_objects/            # not committed — see .gitignore
```

**Deviation from the supplementary brief's suggested layout, noted
deliberately rather than left unexplained:** the brief's example
structure nests everything inside `src/appliance_energy/` with a further
`models/` subpackage. This repo uses one consolidated, fully-documented
script per assignment part at the repo root instead. Two reasons: the
supplementary demo script provided alongside the brief was itself a
single flat file, not a nested package, and a flat structure keeps each
part's full pipeline (data → model → evaluation → save) readable in one
place rather than spread across multiple imports. `part2_problem_definition.py`
is the one shared module every later part imports from (constants and
metric functions), so logic isn't duplicated across parts despite the
flat layout.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Running the pipeline

Each part is provided as both a `.py` script (`python part1_data_prep_eda.py`,
etc.) and a notebook in `notebooks/`, with an "Open in Colab" badge — the
notebooks were the primary way this was developed and run.

**Run in order, Parts 1 → 7**, since later parts depend on earlier ones'
outputs (either regenerated automatically if missing, or read directly):

| Part | Runtime | Notes |
|---|---|---|
| 1–3, 5 | seconds | Self-contained; fetch/rebuild data automatically |
| 4 (SARIMAX) | **15–25 minutes** | Full non-seasonal AIC grid search (147 combinations) as required by the brief, plus a seasonal refinement grid |
| 6 (XGBoost) | ~7 minutes | `TimeSeriesSplit` hyperparameter search |
| 7 (foundation model) | a few minutes | Needs internet access to `huggingface.co` to download Chronos-Bolt's pretrained weights on first run |
| 8 (comparison) | seconds | **Does not regenerate** Parts 3/4/6/7 — reads their saved `outputs/metrics/*.csv` and `outputs/forecasts/*.csv`. Run those parts first (same Colab session, or upload their output files into a fresh one). Any model whose files aren't found is skipped with a warning rather than failing. |

Each part is independently reproducible from a fresh clone: the `.py`
scripts and notebooks fetch the raw dataset directly from a public
GitHub mirror of the UCI data on first run, so no manual download step
is required.

## Outputs

- `outputs/metrics/model_comparison.csv` — MAE/RMSE/MASE/Bias for every
  model, produced by Part 8.
- `outputs/forecasts/all_forecasts.csv` — actual values plus every
  model's forecast over the test period.
- `outputs/figures/` — all plots (EDA, diagnostics, forecast comparisons).

## Data leakage and evaluation caveats

Different models in this project are evaluated under genuinely different
conditions — not a level playing field, and stated explicitly rather than
glossed over:

- **Benchmarks (Part 3) and SARIMAX (Part 4):** blind forecasts — no
  information from the test period is used at all.
- **XGBoost (Part 6):** a *conditional* forecast — evaluated using real
  historical lag values and real future weather for the test rows (see
  Part 5's feature-engineering notes for the full reasoning).
- **Chronos-Bolt (Part 7):** zero-shot and target-only — no fine-tuning
  on this dataset, no covariates at all.

## Tests

```bash
pytest
```

23 tests across four files, covering:
- `test_data.py` — missingness detection, hourly resampling (sum vs.
  mean, matching Part 1's documented design choice)
- `test_features.py` — lag/rolling features never leak the current or
  future target value; the feature table has no missing target values
- `test_evaluation.py` — MASE is exactly zero for a perfect forecast;
  the train/test split is chronological; forecast length matches the
  test period
- `test_benchmarks.py` — each benchmark forecast function produces the
  correct values and length

## Acknowledgements

Dataset: Candanedo, L.M., Feldheim, V., Deramaix, D. (2017). *Data driven
prediction models of energy use of appliances in a low energy house*.
Energy and Buildings, 140, 81-97.
