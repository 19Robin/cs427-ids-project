# Cross-Domain Evaluation of a Lightweight ML Intrusion Detection System for 5G / Mobile Network Traffic

CS427 - Mobile Communications | Samuela Robin (S11199961) & Katea Yavunitu (S11149012)

## Research question

How well does a lightweight machine-learning intrusion detection system (IDS)
trained on traffic from a **5G mobile network** perform when it is applied,
**without retraining**, to traffic from a **different IoT network environment**?

```
5G mobile network (5G-NIDD)  ->  flow features  ->  ML IDS (trained on 80% of 5G-NIDD)
                                                     |-> in-domain test: 20% held-out 5G-NIDD
CICIoT2023 (non-5G IoT lab)  ->  same 7 features ----> cross-domain test: no retraining
```

## How to run

### 1. Requirements

- Python 3.10 or newer
- Git
- For live packet capture only: [Npcap](https://npcap.com/#download) (Windows)
- Node.js is **not** needed to run the dashboard; the built website is included

### 2. Set up (first time only)

```bat
git clone https://github.com/19Robin/cs427-ids-project.git
cd cs427-ids-project
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

On macOS / Linux, activate with `source .venv/bin/activate` instead.

### 3. Start the dashboard

```bat
.venv\Scripts\activate.bat
uvicorn api.main:app --port 8000
```

Open **http://localhost:8000** in a browser. Press `Ctrl+C` in the terminal to stop.

- For **live packet capture**, open the terminal with **Run as administrator**.
- To open the dashboard on a phone connected to the laptop's hotspot, start it
  with `uvicorn api.main:app --host 0.0.0.0 --port 8000` and browse to
  `http://192.168.137.1:8000` on the phone.

The research pages work straight away because the results are in `results/`.
The Prediction Demo, Edge IDS Monitor and Live IDS Monitoring pages also need
the trained models in `models/`, which are not stored in Git (see step 4).

### 4. Re-create the results and models (optional)

Download the datasets and place them in:

```
data/raw/5g_nidd.csv
data/raw/cic_iot/Merged01.csv ... Merged05.csv
```

Then run, in this order:

```bat
python src\preprocess_data.py
python src\train_model.py
python src\test_cross_domain.py
python src\test_lightweight_models.py
python src\test_lightweight_cross_domain.py
python src\test_other_models.py
python src\test_attack_types.py
python src\analyse_feature_shift.py
python src\extract_demo_samples.py
python src\run_improved_experiments.py
```

`run_improved_experiments.py` takes the longest. Restart the dashboard afterwards.

### 5. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Activate `.venv`, then `pip install -r requirements.txt` |
| A page says "Model file not available" | Run step 4 (at least `test_lightweight_models.py` and `run_improved_experiments.py`) |
| Live capture says Npcap is missing | Install Npcap with *WinPcap API-compatible Mode* ticked, then restart the terminal |
| Live capture permission error | Start the terminal with **Run as administrator** |
| Port 8000 already in use | Use another port, e.g. `--port 8001` |
| Dashboard looks out of date after `git pull` | Restart the server and press `Ctrl+F5` in the browser |

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
python src/run_improved_experiments.py      # dedup + balancing, diagnosis experiments A-D,
                                            # edge model + flows for the Edge IDS Monitor
```

## Dashboard

### React dashboard (recommended)

A React interface (`frontend/`) on top of a FastAPI server (`api/`) that wraps
the existing Python code: result files, trained models, the Edge IDS replay and
the live capture. The built site (`frontend/dist/`) is committed, so a demo
needs **only Python and one command**, run from the project root:

```bash
uvicorn api.main:app --port 8000
```

Open http://localhost:8000. To open it from a phone on the laptop's hotspot,
start it with `--host 0.0.0.0` and browse to `http://192.168.137.1:8000`. For
live packet capture on Windows, start the terminal as administrator (same
Npcap requirements as below).

Pages: Overview · Model Size & Trees · Algorithm Comparison · Cross-Domain
Evaluation · Attack Analysis · Feature & Protocol Shift · Improved Experiments ·
5G Edge IDS Monitor · Live IDS Monitoring · Prediction Demo. Pages whose
results or models are missing show the command that creates them.

Changing the frontend needs Node.js 20+:

```bash
cd frontend
npm install
npm run dev      # development server on http://localhost:5173 (uses the API on port 8000)
npm run build    # rebuild frontend/dist after changes, then commit it
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

1. `pip install -r requirements.txt` (includes `scapy`).
2. For Live Capture Mode install **Npcap** from https://npcap.com/#download
   and tick *"Install Npcap in WinPcap API-compatible Mode"*. If you tick
   *"Restrict Npcap driver's access to Administrators only"*, start the
   terminal with **Run as administrator** before `uvicorn api.main:app --port 8000`.
   Demo Mode needs neither Npcap nor admin rights.

### Find the network interface

```bash
python -m src.live_capture --list-interfaces
```

Use the adapter that carries the test traffic: usually `WiFi`, or the
*Microsoft Wi-Fi Direct Virtual Adapter* (IP `192.168.137.1`) when the phone
is connected to the laptop's Mobile hotspot. A quick command-line check
without the dashboard:

```bash
python -m src.live_capture --test-capture "WiFi" --seconds 15
python -m src.live_capture --test-demo --seconds 15
```

### Run and use

```bash
uvicorn api.main:app --port 8000
```

Open http://localhost:8000 and choose **Live IDS Monitoring** in the sidebar.

- **Live Capture Mode:** choose the interface, optionally enter one device IP
  (only its traffic is analysed), choose the window length, press
  **Start Monitoring**. Capture is passive; nothing is transmitted.
- **Demo Mode:** labelled *SIMULATION — NOT REAL NETWORK TRAFFIC*. Synthetic
  packets are created in memory (never sent) and pass through the same
  feature extraction and model. "Auto" shows normal traffic with a simulated
  suspicious burst every 10 windows; the burst profiles imitate 5G-NIDD attack
  flow statistics, and the model makes the decision.
- **Stop** stops capture; **Clear** empties the tables and graphs.
- **Graphs:** show a 1, 2 or 5 minute window. Use ◀ ▶, drag the threat chart
  or move the slider under the overview strip to look back through the
  session (up to 2,000 windows); **Live** jumps back to the newest data.
- **Export:** downloads the session as CSV (one file per window, one per
  classified flow). The 5G Edge IDS Monitor can export its replay as CSV, and
  **Download all results** in the sidebar gives a ZIP of the `results/` folder.

### Controlled demonstration with an Android phone

The phone only generates **normal traffic** inside your own test network.

1. Laptop: *Settings → Network & internet → Mobile hotspot → On*.
2. Connect the phone to the hotspot; note its IP (usually `192.168.137.x`).
3. Select the Wi-Fi Direct Virtual Adapter (`192.168.137.1`) and optionally
   enter the phone's IP.
4. Start monitoring and use the phone normally: browse, stream video, or
   open the dashboard on the phone at `http://192.168.137.1:8000` (start the
   server with `--host 0.0.0.0`).

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
- The monitor runs in the API server and stops itself about 3 minutes after the
  Live IDS Monitoring page was last open.

## Repository layout

```
api/                       FastAPI server for the React dashboard
frontend/                  React dashboard (Vite + Tailwind + Recharts); dist/ is the built site
src/                       preprocessing, training and evaluation scripts (above)
src/live_*.py              live IDS: capture, feature extraction, prediction
results/                   all result files used by the dashboard
models/                    trained models (git-ignored)
data/raw, data/processed   datasets (git-ignored)
final_feature_validation.py  check of the 7-feature mapping on the raw datasets
```

Files from an earlier UNSW-NB15 prototype and one-off inspection scripts were
removed; they remain available in the Git history.

## Improved experiments (`src/run_improved_experiments.py`)

Writes to `results/improved/` and `models/edge_ids_model.joblib`; shown on the
dashboard's **Improved Experiments** and **Edge IDS Monitor** pages.

- Removes duplicate records and measures train/test overlap in the original split
- Retrains all models with class balancing (and scaled Logistic Regression),
  plus a depth-limited 10-tree Random Forest; reports size in KB (compared with
  the 108.42 KB DL-IID model of Khan et al., 2025) and per-flow latency
- Per-attack-type detection for 5G-NIDD and CICIoT2023
- Diagnosis: A1 protocol flags rounded to 0/1, A2 + Packet_Count removed,
  B comparable attack types only, C per-domain standardisation,
  D 1% labelled CICIoT2023 (retraining, reported separately)

Re-run `src/preprocess_data.py` first so the processed files include `Attack_Type`.

## Limitations

- One random 80/20 split and one seed; no check yet for duplicate feature
  vectors shared between train and test.
- Logistic Regression is trained on unscaled features.
- The common features are coarse and not identically defined across datasets.
- CICIoT2023 contains many attack types that do not exist in 5G-NIDD.
- Not a production IDS (this includes the live monitoring proof-of-concept).
