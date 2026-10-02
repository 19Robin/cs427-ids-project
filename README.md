# Cross-Domain Evaluation of a Lightweight ML Intrusion Detection System for 5G / Mobile Network Traffic

CS427 - Mobile Communications | Samuela Robin (S11199961) & Katea Yavunitu (S1114901)

## Research question

How well does a lightweight machine-learning intrusion detection system (IDS)
trained on traffic from a **5G mobile network** perform when it is applied,
**without retraining**, to traffic from a **different IoT network environment**?

```
5G mobile network (5G-NIDD)  ->  flow features  ->  ML IDS (trained on 80% of 5G-NIDD)
                                                     |-> in-domain test: 20% held-out 5G-NIDD
CICIoT2023 (non-5G IoT lab)  ->  same 7 features ----> cross-domain test: no retraining
```

## Why this is a mobile-communications project

- The training data, **5G-NIDD**, is network traffic generated on a real 5G
  test network, with benign users and attackers connected through 5G base
  stations (DoS/DDoS floods and port scans).
- 5G connects very large numbers of IoT devices (mMTC); compromised devices
  can attack the mobile network, so operators need network-level intrusion
  detection.
- An IDS deployed near the traffic (e.g. at the network edge / MEC) must be
  **lightweight**, so model size and inference time are reported.
- Mobile network environments change (cells, operators, slices, IoT
  verticals). The project measures how a 5G-trained IDS behaves when the
  environment changes.

Scope note: the seven common features are generic IP-flow features, not
5G-specific radio/core signalling features. **CICIoT2023 is not a 5G
dataset**; it is used as the "different environment".

## Datasets

| Role | Dataset | Processed rows | Malicious share |
|---|---|---|---|
| Source (train + in-domain test) | 5G-NIDD — https://dx.doi.org/10.21227/xtep-hv36 | 1,215,676 | 60.7% |
| Target (cross-domain test only) | CICIoT2023 (Merged01–05 CSVs) | 3,579,528 | 97.7% |

Place the raw files in `data/raw/5g_nidd.csv` and `data/raw/cic_iot/Merged*.csv`
(both are git-ignored).

## Common features

| Feature | 5G-NIDD source | CICIoT2023 source |
|---|---|---|
| Rate | `Rate` | `Rate` |
| Packet_Count | `TotPkts` | `Number` |
| Mean_Packet_Size | `TotBytes / TotPkts` | `AVG` |
| TTL | `sTtl` | `Time_To_Live` |
| TCP / UDP / ICMP | one-hot from `Proto` | `TCP` / `UDP` / `ICMP` |

Mapping caveats visible in `results/feature_distribution_shift.csv`: the
CICIoT2023 protocol columns are fractional (not 0/1) and `Number` never
exceeds 100, so the CICIoT2023 fields are aggregated differently from
5G-NIDD's per-flow records.

## Key results (from `results/`)

| Model | 5G-NIDD F1 | CICIoT2023 F1 | Drop (pp) |
|---|---|---|---|
| Random Forest (10 trees) | 80.84% | 2.25% | 78.59 |
| Decision Tree | 80.34% | 0.71% | 79.63 |
| Logistic Regression | 83.15% | 1.87% | 81.28 |
| XGBoost | 82.83% | 0.23% | 82.60 |

How to read these numbers:

- CICIoT2023 is 97.7% malicious, so cross-domain **accuracy** mostly reflects
  the benign records, and cross-domain **precision** (61–95%) is *below* the
  97.7% base rate — it is not a strength. Recall and F1 are the informative
  metrics.
- On 5G-NIDD a trivial "always malicious" rule scores 75.6% F1, so the
  in-domain results are moderate rather than strong.
- The 100-tree baseline labels 72.4% of 5G-NIDD test records as malicious but
  only 1.2% of CICIoT2023 records.
- Feature and protocol distribution differences provide evidence of domain
  shift that may contribute to the degradation; they do not prove a single
  cause.

## Pipeline

Run from the project root (paths are resolved relative to the repository):

```bash
pip install -r requirements.txt

python src/preprocess_data.py               # build data/processed/*.csv (7 features + Target)
python src/train_model.py                   # 100-tree RF baseline, in-domain results
python src/test_cross_domain.py             # baseline on CICIoT2023
python src/test_lightweight_models.py       # 10/25/50-tree RFs (in-domain)
python src/test_lightweight_cross_domain.py # 10/25/50/100-tree RFs on CICIoT2023
python src/test_other_models.py             # Decision Tree, Logistic Regression, XGBoost
python src/test_attack_types.py             # per-attack-type detection (100-tree baseline)
python src/analyse_feature_shift.py         # feature / protocol distribution shift
python src/extract_demo_samples.py          # real 5G-NIDD test records for the demo (optional)

streamlit run app.py
```

Model files (`models/*.joblib`) are git-ignored because of their size, so the
Prediction Demo only works after running the training scripts locally. All
other dashboard pages read only the files in `results/`.

## Repository layout

```
app.py                     Streamlit dashboard
src/                       preprocessing, training and evaluation scripts (above)
results/                   all result files used by the dashboard
models/                    trained models (git-ignored)
data/raw, data/processed   datasets (git-ignored)
*.py (root)                one-off dataset inspection / mapping-check scripts
```

Legacy files from an earlier UNSW-NB15 prototype (`src/data_loader.py`,
`src/evaluate.py`, `src/diagnose_collapse.py`, `src/visualize.py`,
`notebooks/01_exploration.ipynb`, `outputs/`) are not part of the current
5G-NIDD → CICIoT2023 pipeline and their results should not be reported.

## Limitations

- One random 80/20 split and one seed; no check yet for duplicate feature
  vectors shared between train and test.
- Logistic Regression is trained on unscaled features.
- The common features are coarse and not identically defined across datasets.
- CICIoT2023 contains many attack types that do not exist in 5G-NIDD.
- Not a production IDS.
