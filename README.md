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

## Live IDS monitoring (proof-of-concept)

The **Live IDS Monitoring** page applies the existing 10-tree Random Forest
(`models/random_forest_10_trees.joblib`) to live traffic:

```
live packets -> flows per time window -> same 7 features -> existing model -> BENIGN / MALICIOUS -> dashboard
```

**The model is only loaded and used for `predict()` / `predict_proba()`; it
is never retrained.** On load, `src/live_predictor.py` checks that the model's
`feature_names_in_` equals the live feature order and that `classes_` is
`[0, 1]` (0 = benign, 1 = malicious, as set in `preprocess_data.py`). The
malicious probability is the class-1 column of `predict_proba` — a model
score (the trees' average leaf class share), not a calibrated probability.

### How the seven features are extracted

5G-NIDD records are Argus *bidirectional flow records*, emitted every ~5 s for
active flows. The live extractor (`src/live_features.py`) reproduces them: one
flow (protocol + the two endpoints, both directions) inside one window is one
record and one prediction. The formulas were checked against the raw
5G-NIDD file:

| Feature | 5G-NIDD (training) | Live computation |
|---|---|---|
| Rate | Argus `Rate` (verified = `(TotPkts−1)/Dur` on 100% of rows) | `(packets − 1) / (last − first packet time)`, 0 for one packet |
| Packet_Count | `TotPkts` | packets of the flow in the window |
| Mean_Packet_Size | `TotBytes / TotPkts` (bytes include the 14-byte Ethernet header) | mean of IP length + 14 |
| TTL | `sTtl` (source TTL) | TTL / IPv6 hop limit of the flow originator's packets |
| TCP / UDP / ICMP | one-hot of `Proto` (ICMPv6 counts as ICMP) | identical one-hot; all 0 for other IP protocols |

The default 5-second window matches the Argus status interval. Non-IP frames
are ignored (5G-NIDD rows without a TTL were dropped in preprocessing).
Only a rolling history (last 500 flows / 240 windows) is kept; packets are
never stored.

### Installation (Windows 11)

1. `pip install -r requirements.txt` (adds `scapy` and `altair`).
2. For Live Capture Mode install **Npcap** from https://npcap.com/#download
   and tick *"Install Npcap in WinPcap API-compatible Mode"*. If you tick
   *"Restrict Npcap driver's access to Administrators only"*, start the
   terminal with **Run as administrator** before `streamlit run app.py`.
   Demo Mode needs neither Npcap nor admin rights.

### Find the network interface

```bash
python -m src.live_capture --list-interfaces
```

Use the adapter that carries the test traffic: usually `WiFi`, or the
*Microsoft Wi-Fi Direct Virtual Adapter* (IP `192.168.137.1`) when the phone
is connected to the laptop's Mobile hotspot. A quick command-line check
without Streamlit:

```bash
python -m src.live_capture --test-capture "WiFi" --seconds 15
python -m src.live_capture --test-demo --seconds 15
```

### Run and use

```bash
streamlit run app.py
```

Open **Live IDS Monitoring** in the sidebar.

- **Live Capture Mode:** choose the interface, optionally enter one device IP
  (only its traffic is analysed), choose the window length, press
  **Start Monitoring**. Capture is passive; nothing is transmitted.
- **Demo Mode:** labelled *SIMULATION — NOT REAL NETWORK TRAFFIC*. Synthetic
  packets are created in memory (never sent) and pass through the same
  feature extraction and model. "Auto" shows normal traffic with a simulated
  suspicious burst every 10 windows; the burst profiles imitate 5G-NIDD attack
  flow statistics, and the model makes the decision.
- **Stop Monitoring** stops capture; **Clear History** empties the tables and
  graphs. While monitoring runs, a status line appears in the sidebar on
  every page.

### Controlled demonstration with an Android phone

The phone only generates **normal traffic** inside your own test network.

1. Laptop: *Settings → Network & internet → Mobile hotspot → On*.
2. Connect the phone to the hotspot; note its IP (usually `192.168.137.x`).
3. Select the Wi-Fi Direct Virtual Adapter (`192.168.137.1`) and optionally
   enter the phone's IP.
4. Start monitoring and use the phone normally: browse, stream video, or
   open the dashboard on the phone at `http://192.168.137.1:8501`.

On a shared Wi-Fi network (no hotspot) the laptop only sees its own traffic
and traffic addressed to it, so use the hotspot for phone traffic. Only use
devices and networks you own or control.

### Limitations of the live proof-of-concept

- **Cross-domain:** the model was validated only on 5G-NIDD and dropped to
  ~2% F1 on CICIoT2023. Home Wi-Fi / phone traffic is another domain, so
  live verdicts demonstrate the pipeline, not reliable detection. Many 5G-NIDD
  attack flows have testbed-specific values (e.g. TTL 63) that ordinary
  laptops (TTL 64/128) rarely produce, while some benign live flows can still
  be flagged.
- Flows are approximated per window from packets; Argus' exact flow timeout
  and state logic is not reproduced, and flows split across window edges.
- Outgoing packets captured on Windows may be larger than the MTU because of
  NIC segmentation offload, which inflates Mean_Packet_Size.
- Scapy dissects packets in Python: suitable for a laptop demo (hundreds to a
  few thousand packets per second), not for high-speed links. At most 5,000
  flows per window are analysed.
- Monitoring state belongs to one browser session; if the browser is closed,
  the monitor stops itself after about 3 minutes.

## Repository layout

```
app.py                     Streamlit dashboard
src/                       preprocessing, training and evaluation scripts (above)
src/live_*.py              live IDS: capture, feature extraction, prediction
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
- Not a production IDS (this includes the live monitoring proof-of-concept).
