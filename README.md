# EPL 2026/27 Match Outcome Prediction

A machine-learning project for predicting English Premier League match outcomes using historical EPL records, current match statistics, and referee tendencies.

## Project objective

Predict the final result:

- `H` — Home Win
- `D` — Draw
- `A` — Away Win

The project deliberately separates **pre-match forecasting** from **in-match forecasting**.

### Pre-match mode

Designed for upcoming 2026/27 fixtures. It uses information that can be known before kickoff:

- team identity
- Elo strength
- recent form
- goals for / against
- home/away performance
- historical shots and shots on target
- historical corners
- historical discipline
- referee tendencies
- calendar features

### In-match mode

Designed around the feature set requested for matches such as Arsenal vs Chelsea. It can use:

- Arsenal/home shots
- Chelsea/away shots
- shots on target
- half-time score
- corners
- fouls
- yellow cards
- red cards
- current xG when available
- historical team/referee features

**Final score is not a feature.** `FullTimeHomeGoals` and `FullTimeAwayGoals` are the ground truth. Supplying them to a model that predicts `FullTimeResult` would be direct target leakage.

---

## Data used

Three user-supplied datasets were integrated:

1. `epl_final.csv`
2. `epl-allseasons-matchstats.csv`
3. `Book1.xlsx`

The sources are reconciled with:

`Season + MatchDate + normalized HomeTeam + normalized AwayTeam`

### Integration result

- **9,828 unique match/fixture rows**
- **9,617 completed matches**
- **211 unplayed 2025/26 fixture placeholders**
- **2024/25 expanded from 350 to 380 matches**
- **2025/26 contains 380 scheduled rows, 169 completed results**
- **67 unique referees**
- **46 referees with at least 20 historical matches**

The 211 unplayed rows are retained for completeness but are never treated as draws and never used as training targets.

---

## Referee impact analysis

The referee module measures **historical tendencies/associations**, not causal effects.

For referees with enough observations it calculates:

- average total fouls
- average total yellow cards
- average total red cards
- average home and away yellow cards
- home-yellow-card bias
- historical home-win rate
- historical draw rate
- historical away-win rate

The referee history used for a match is calculated from information available before that match.

Example high-card historical referees in this dataset include John Brooks, Tim Robinson, David Coote, Darren Bond and Darren England. These are descriptive results, not evidence that officiating alone causes outcomes.

---

## Machine-learning models

### Linear Regression

Predicts:

`Goal Difference = Home Goals - Away Goals`

The predicted goal difference is mapped to H/D/A and used as an interpretable baseline.

### XGBoost

Directly predicts H/D/A and returns:

`P(Home), P(Draw), P(Away)`

### Model settings

Pre-match XGBoost:
```text
n_estimators = 250
max_depth = 2
learning_rate = 0.05
min_child_weight = 1
subsample = 0.90
colsample_bytree = 0.90
reg_alpha = 0.0
reg_lambda = 1.0
```

In-match XGBoost:
```text
n_estimators = 350
max_depth = 4
learning_rate = 0.035
min_child_weight = 3
subsample = 0.85
colsample_bytree = 0.90
reg_alpha = 0.2
reg_lambda = 1.5
```

---

## Chronological evaluation

The held-out test season is **2024/25**.

| Model | Accuracy | Macro F1 | MAE | RMSE | R² |
|---|---:|---:|---:|---:|---:|
| Linear Regression — Pre-match | 0.487 | 0.440 | 1.422 | 1.781 | 0.087 |
| XGBoost — Pre-match | 0.524 | 0.390 | — | — | — |
| Linear Regression — In-match | 0.651 | 0.581 | 0.922 | 1.164 | 0.619 |
| XGBoost — In-match | 0.651 | 0.590 | — | — | — |

For this benchmark, the in-match models have a major advantage because the model sees how the game is actually unfolding.

---

## 2026/27 prediction output

The repository contains predictions for the upcoming fixture window represented in:

`data/2026_27_upcoming_fixtures_mw6_mw10.csv`

and:

`results/2026_27_upcoming_predictions_mw6_mw10.csv`

The official Premier League schedule is subject to fixture amendments.

### Important current-data limitation

The supplied datasets do not include completed 2026/27 match results/statistics. Therefore the included 2026/27 pre-match predictions are based on supplied history through the **completed portion of 2025/26**.

The local snapshot should be refreshed with current 2026/27 results before being used as a live-season forecasting system.

---

## Repository structure

```text
epl-2026-27-match-prediction/
│
├── data/
│   ├── raw/
│   │   ├── epl_final.csv
│   │   ├── epl-allseasons-matchstats.csv
│   │   └── Book1.xlsx
│   ├── epl_combined.csv
│   ├── engineered_features.csv
│   ├── referee_impact_profile.csv
│   ├── 2026_27_upcoming_fixtures_mw6_mw10.csv
│   └── source_notes.md
│
├── notebooks/
│   └── 01_EPL_2026_27_Match_Prediction.ipynb
│
├── models/
│   ├── prematch_xgboost.joblib
│   ├── prematch_linear_regression.joblib
│   ├── inmatch_xgboost.joblib
│   └── inmatch_linear_regression.joblib
│
├── results/
│   ├── model_metrics_2024_25.csv
│   ├── classification_reports.txt
│   ├── data_audit_summary.csv
│   ├── feature_dictionary.csv
│   ├── referee_impact_eligible_20_matches.csv
│   ├── 2026_27_upcoming_predictions_mw6_mw10.csv
│   ├── example_inmatch_arsenal_vs_chelsea.csv
│   └── model plots
│
├── src/
│   └── epl_pipeline.py
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Installation

```bash
python -m venv .venv
```

Windows:

```bash
.venv\\Scripts\\activate
```

Install:

```bash
pip install -r requirements.txt
```

Launch:

```bash
jupyter notebook
```

Run:

```text
notebooks/01_EPL_2026_27_Match_Prediction.ipynb
```

---

## Resume-ready description

**English Premier League Match Outcome Prediction | Python, XGBoost, Scikit-learn**

- Integrated **9,800+ EPL match/fixture records** from multiple historical sources and engineered Elo, recent-form, home/away, shooting, discipline and referee-tendency features.
- Built Linear Regression and XGBoost models for H/D/A outcome prediction with chronological validation, probability outputs and confusion-matrix analysis.
- Developed separate pre-match and in-match pipelines, incorporating half-time score, shots, corners, fouls and cards for live prediction while preventing final-score target leakage.

---

## Data sources

Official Premier League fixtures:
https://www.premierleague.com/en/fixtures

A public 2026/27 fixture-data reference used during research:
https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2026-27/fixtures.csv
