# Cross-Domain Generalization of Lightweight IDS Models for IoT in 5G Networks

CS427 - Mobile Communications | Samuela Robin (S11199961) & Katea Yavunitu (S1114901)

## What this project does

Trains a lightweight machine learning model (Random Forest / XGBoost) on one
IoT/5G intrusion detection dataset, then measures how much its accuracy drops
when tested on a second, structurally different dataset without retraining.

## Folder structure

```
cs427-ids-project/
├── data/
│   ├── raw/              <- put downloaded datasets here (NOT committed to git)
│   └── processed/        <- cleaned/preprocessed versions (also gitignored)
├── notebooks/
│   └── 01_exploration.ipynb   <- for interactive exploration
├── src/
│   ├── data_loader.py    <- loads and cleans both datasets
│   ├── train.py          <- trains the model on domain A
│   ├── evaluate.py       <- runs in-domain and cross-domain evaluation
│   └── visualize.py      <- generates comparison charts
├── outputs/
│   ├── figures/          <- saved charts (confusion matrices, comparisons)
│   └── models/           <- saved trained model files
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Datasets

- **5G-NIDD** (training domain): https://dx.doi.org/10.21227/xtep-hv36
- **UNSW-NB15** (cross-domain test): https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15

Download both, extract, and place the CSV files in `data/raw/`.

## Running the pipeline

**Important: run these commands from the project's root folder (not from inside `src/`), or the file paths won't resolve correctly.**

```bash
python src/data_loader.py     # sanity-check: confirms both datasets load
python src/train.py           # train the model on domain A only
python src/evaluate.py        # run in-domain and cross-domain evaluation
python src/visualize.py       # generate comparison charts
```

Right now `data_loader.py` uses synthetic placeholder data (see the TODO
comments inside it) so you can confirm the whole pipeline runs correctly
before plugging in the real datasets. Once you've downloaded 5G-NIDD and
UNSW-NB15 into `data/raw/`, edit the two functions in `data_loader.py` to
read the real CSV files instead.

Or work through `notebooks/01_exploration.ipynb` interactively instead.
