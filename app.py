"""
Streamlit dashboard for the CS427 project:

    Cross-Domain Evaluation of a Lightweight Machine-Learning Intrusion
    Detection System for 5G / Mobile Network Traffic

Every number shown here is read from the files in results/ (produced by the
scripts in src/). Values that are *derived* from those files (for example
class balance, which is taken from the saved confusion matrices) are labelled
as derived. Nothing is hard-coded except dataset descriptions and labels.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import altair as alt

# Live IDS proof-of-concept (src/live_*.py). Imported defensively so that a
# problem in the live component can never break the research pages.
try:
    from src.live_capture import (
        DEMO_SCENARIOS, CaptureError, LiveMonitor, capture_status, list_interfaces,
    )
    from src.live_predictor import LivePredictor, ModelCompatibilityError
    LIVE_IMPORT_ERROR = None
except Exception as live_error:
    LIVE_IMPORT_ERROR = live_error


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="5G IDS Cross-Domain Evaluation",
    page_icon="📡",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

BASELINE_RESULTS = RESULTS_DIR / "baseline_5g_results.json"
CROSS_DOMAIN_JSON = RESULTS_DIR / "cross_domain_results.json"
LIGHTWEIGHT_RESULTS = RESULTS_DIR / "lightweight_model_comparison.csv"
CROSS_DOMAIN_RESULTS = RESULTS_DIR / "lightweight_cross_domain_comparison.csv"
OTHER_MODELS_RESULTS = RESULTS_DIR / "other_models_comparison.csv"
FEATURE_IMPORTANCE = RESULTS_DIR / "baseline_feature_importance.csv"
ATTACK_RESULTS = RESULTS_DIR / "attack_type_results.csv"
FEATURE_SHIFT = RESULTS_DIR / "feature_distribution_shift.csv"
PROTOCOL_SHIFT = RESULTS_DIR / "protocol_distribution_shift.csv"
DEMO_SAMPLES = RESULTS_DIR / "demo_samples_5g_nidd.csv"

LIGHTWEIGHT_MODEL = MODELS_DIR / "random_forest_10_trees.joblib"

FEATURES = [
    "Rate",
    "Packet_Count",
    "Mean_Packet_Size",
    "TTL",
    "TCP",
    "UDP",
    "ICMP"
]

CONTINUOUS_FEATURES = ["Rate", "Packet_Count", "Mean_Packet_Size", "TTL"]
PROTOCOL_FEATURES = ["TCP", "UDP", "ICMP"]


# ============================================================
# CHART STYLE
# ============================================================

# One colour per dataset, used consistently on every page:
#   blue   = 5G-NIDD (in-domain, 5G mobile network)
#   orange = CICIoT2023 (cross-domain, non-5G IoT network)
COLOR_5G = "#2a78d6"
COLOR_CIC = "#eb6834"
COLOR_REFERENCE = "#7a7974"

# Metric colours (line charts also use distinct markers so the
# series never depend on colour alone).
METRIC_STYLE = {
    "Accuracy": ("#4a3aa7", "o"),
    "Precision": ("#1baf7a", "s"),
    "Recall": ("#e87ba4", "^"),
    "F1": ("#eda100", "D"),
}

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#b5b4ae",
    "axes.labelcolor": "#2b2b2b",
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "grid.color": "#e4e3de",
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "font.size": 10.5,
    "figure.dpi": 110,
})


def show(fig):
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def pct(value, digits=2):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value * 100:.{digits}f}%"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_csv(path):
    if path.exists():
        df = pd.read_csv(path)
        # Guard against duplicated column names, which would make
        # df[column] return a 2-D frame instead of a 1-D series.
        return df.loc[:, ~df.columns.duplicated()].copy()
    return None


@st.cache_data
def load_json(path):
    if path.exists():
        with open(path, "r") as file:
            return json.load(file)
    return None


@st.cache_resource
def load_model(path):
    return joblib.load(path)


def column(df, name):
    """
    Return one numeric 1-D column. Fails loudly instead of silently
    plotting a 2-D object (the cause of the earlier
    'shape mismatch ... (7,) and (7, 2)' error).
    """
    values = df[name]
    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)


def has_columns(df, names):
    return df is not None and all(name in df.columns for name in names)


baseline = load_json(BASELINE_RESULTS)
cross_json = load_json(CROSS_DOMAIN_JSON)
lightweight = load_csv(LIGHTWEIGHT_RESULTS)
cross_domain = load_csv(CROSS_DOMAIN_RESULTS)
other_models = load_csv(OTHER_MODELS_RESULTS)
feature_importance = load_csv(FEATURE_IMPORTANCE)
attack_results = load_csv(ATTACK_RESULTS)
feature_shift = load_csv(FEATURE_SHIFT)
protocol_shift = load_csv(PROTOCOL_SHIFT)
demo_samples = load_csv(DEMO_SAMPLES)


# ============================================================
# DERIVED VALUES (all computed from the result files)
# ============================================================

def confusion_summary(cm):
    """
    Summarise a 2x2 confusion matrix [[TN, FP], [FN, TP]]
    (label 0 = benign, 1 = malicious, as in sklearn).
    """
    if cm is None:
        return None
    (tn, fp), (fn, tp) = cm
    total = tn + fp + fn + tp
    benign = tn + fp
    malicious = fn + tp
    prior = malicious / total
    specificity = tn / benign if benign else np.nan
    recall = tp / malicious if malicious else np.nan
    return {
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "total": total,
        "benign": benign,
        "malicious": malicious,
        "malicious_share": prior,
        "predicted_malicious_share": (fp + tp) / total,
        "specificity": specificity,
        "recall": recall,
        "balanced_accuracy": (specificity + recall) / 2,
        # A trivial classifier that labels every record malicious:
        "trivial_accuracy": prior,
        "trivial_precision": prior,
        "trivial_f1": 2 * prior / (1 + prior),
    }


cm_5g = confusion_summary(baseline["confusion_matrix"]) if baseline else None
cm_cic = confusion_summary(cross_json["confusion_matrix"]) if cross_json else None


def build_algorithm_table():
    """
    One row per algorithm (in-domain and cross-domain), combining the
    10-tree Random Forest (from the lightweight result files) with the
    Decision Tree, Logistic Regression and XGBoost results. All four
    were trained on the same 80/20 stratified 5G-NIDD split
    (random_state=42) with the same seven features.
    """
    rows = []

    if has_columns(lightweight, ["Trees"]) and has_columns(cross_domain, ["Trees"]):
        rf_in = lightweight[lightweight["Trees"] == 10]
        rf_out = cross_domain[cross_domain["Trees"] == 10]
        if not rf_in.empty and not rf_out.empty:
            rf_in = rf_in.iloc[0]
            rf_out = rf_out.iloc[0]
            rows.append({
                "Model": "Random Forest (10 trees)",
                "Model_Size_MB": rf_in["Model_Size_MB"],
                "Training_Time_Sec": rf_in["Training_Time_Seconds"],
                "InDomain_Accuracy": rf_in["Accuracy"],
                "InDomain_Precision": rf_in["Precision"],
                "InDomain_Recall": rf_in["Recall"],
                "InDomain_F1": rf_in["F1"],
                "CrossDomain_Accuracy": rf_out["Accuracy"],
                "CrossDomain_Precision": rf_out["Precision"],
                "CrossDomain_Recall": rf_out["Recall"],
                "CrossDomain_F1": rf_out["F1"],
            })

    if other_models is not None:
        for _, row in other_models.iterrows():
            rows.append({key: row[key] for key in [
                "Model", "Model_Size_MB", "Training_Time_Sec",
                "InDomain_Accuracy", "InDomain_Precision",
                "InDomain_Recall", "InDomain_F1",
                "CrossDomain_Accuracy", "CrossDomain_Precision",
                "CrossDomain_Recall", "CrossDomain_F1",
            ]})

    if not rows:
        return None

    table = pd.DataFrame(rows)
    table["F1_Drop_pp"] = (table["InDomain_F1"] - table["CrossDomain_F1"]) * 100
    return table


algorithms = build_algorithm_table()


def size_label(mb):
    if pd.isna(mb):
        return "n/a"
    if mb < 0.01:
        # other_models_comparison.csv rounds sizes to 2 decimals,
        # so a value of 0.0 means "smaller than 0.01 MB".
        return "< 0.01 MB"
    return f"{mb:.2f} MB"


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
        .main-title { font-size: 34px; font-weight: 700; margin-bottom: 0; line-height: 1.2; }
        .subtitle   { font-size: 17px; color: #6b7280; margin-bottom: 12px; }
        .tag {
            display: inline-block; padding: 2px 10px; border-radius: 999px;
            font-size: 13px; font-weight: 600; margin-right: 6px;
        }
        .tag-5g  { background: rgba(42,120,214,0.14); color: #1d5fae; }
        .tag-cic { background: rgba(235,104,52,0.16); color: #b44a1f; }
        .tag-live { background: rgba(12,163,12,0.14); color: #0a7a0a; font-size: 15px; }
        .tag-stop { background: rgba(122,121,116,0.16); color: #52514e; font-size: 15px; }
        .tag-sim  { background: rgba(250,178,25,0.22); color: #8a5a00; }
        .verdict { border-radius: 10px; padding: 14px 18px; font-size: 26px; font-weight: 700; }
        .verdict-malicious { background: rgba(208,59,59,0.12); color: #a82727; border: 1px solid rgba(208,59,59,0.45); }
        .verdict-benign    { background: rgba(12,163,12,0.10); color: #0a7a0a; border: 1px solid rgba(12,163,12,0.40); }
        .verdict-none      { background: rgba(122,121,116,0.10); color: #52514e; border: 1px solid rgba(122,121,116,0.35); }
        .verdict small { display: block; font-size: 13px; font-weight: 500; opacity: 0.85; }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📡 Cross-Domain Evaluation of a Lightweight '
    'ML Intrusion Detection System for 5G / Mobile Network Traffic</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    '<span class="tag tag-5g">Train + in-domain test: 5G-NIDD (5G mobile network)</span>'
    '<span class="tag tag-cic">Cross-domain test, no retraining: CICIoT2023 (IoT lab network, not 5G)</span>'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Model Performance",
        "Algorithm Comparison",
        "Cross-Domain Evaluation",
        "Attack Analysis",
        "Feature Shift",
        "Prediction Demo",
        "Live IDS Monitoring"
    ]
)

st.sidebar.divider()
st.sidebar.markdown(
    f"**Colour key**  \n"
    f"<span style='color:{COLOR_5G}'>■</span> 5G-NIDD — in-domain  \n"
    f"<span style='color:{COLOR_CIC}'>■</span> CICIoT2023 — cross-domain",
    unsafe_allow_html=True
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.header("Research Question")

    st.info(
        "**How well does a lightweight machine-learning IDS trained on traffic "
        "from a 5G mobile network (5G-NIDD) detect attacks when it is applied, "
        "without retraining, to traffic from a different IoT network "
        "environment (CICIoT2023)?**"
    )

    # --------------------------------------------------------
    # PIPELINE DIAGRAM
    # --------------------------------------------------------

    st.subheader("Experimental Pipeline")

    st.graphviz_chart(
        """
        digraph pipeline {
            rankdir=LR;
            bgcolor="transparent";
            node [shape=box, style="rounded,filled", fontname="Helvetica",
                  fontsize=11, color="#b5b4ae", fillcolor="#f4f3ef"];
            edge [color="#7a7974", fontname="Helvetica", fontsize=9];

            subgraph cluster_source {
                label="Source domain: 5G mobile network";
                fontname="Helvetica"; fontsize=11; color="#2a78d6"; style="rounded";
                ue   [label="UEs / attackers\\nattached via 5G RAN", fillcolor="#dce9f8"];
                core [label="5G network\\n(5G-NIDD testbed)", fillcolor="#dce9f8"];
                flows[label="Flow records\\n(Argus)", fillcolor="#dce9f8"];
                ue -> core -> flows;
            }

            feat  [label="Common feature\\nextraction (7 features)"];
            train [label="Train IDS on 80%\\nof 5G-NIDD"];
            ids   [label="Trained IDS\\nRF / DT / LR / XGBoost", fillcolor="#e6e6e1"];
            indom [label="In-domain test\\n20% held-out 5G-NIDD", fillcolor="#dce9f8"];

            subgraph cluster_target {
                label="Target domain: different IoT network (not 5G)";
                fontname="Helvetica"; fontsize=11; color="#eb6834"; style="rounded";
                cic  [label="CICIoT2023\\nIoT lab traffic", fillcolor="#fbe1d6"];
                cfeat[label="Same 7 features", fillcolor="#fbe1d6"];
                cic -> cfeat;
            }

            cross [label="Cross-domain test\\n(no retraining)", fillcolor="#fbe1d6"];
            out   [label="Benign / Malicious\\n+ performance drop", fillcolor="#e6e6e1"];

            flows -> feat -> train -> ids;
            ids -> indom;
            ids -> cross;
            cfeat -> cross;
            indom -> out;
            cross -> out;
        }
        """,
        width="stretch"
    )

    # --------------------------------------------------------
    # MOBILE COMMUNICATIONS RELEVANCE
    # --------------------------------------------------------

    st.subheader("Why this is a Mobile Communications project")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            **The training data is real 5G network traffic.**
            5G-NIDD was generated on a 5G test network, with benign users and
            attackers connected through 5G base stations. Its attacks are the
            ones mobile operators worry about at the network layer: DoS/DDoS
            floods (ICMP, UDP, SYN, HTTP, slow-rate) and port scans.

            **5G connects very large numbers of devices.** Massive IoT
            (mMTC) means many low-cost, rarely patched devices share the same
            mobile network, and compromised devices can be used to flood or
            scan it. Network-level intrusion detection is one of the
            defences an operator can deploy.
            """
        )

    with col2:
        st.markdown(
            """
            **"Lightweight" matters in 5G.** An IDS placed close to the traffic
            (for example at the network edge / MEC) has limited compute and
            memory, which is why model size, training time and prediction
            time are reported alongside accuracy.

            **Mobile network environments change.** Different cells,
            operators, slices and IoT verticals produce different traffic.
            An IDS trained in one environment will inevitably be used in
            others. This project measures what happens to a 5G-trained IDS
            when the environment changes, using CICIoT2023 as the
            *different* environment.
            """
        )

    st.caption(
        "Scope note: the seven common features are generic IP-flow features "
        "(rate, packet count, packet size, TTL, protocol). They are not "
        "5G-specific radio or core-network signalling features. The 5G "
        "connection comes from the source of the training traffic and the "
        "deployment scenario, not from the feature set. CICIoT2023 is not a "
        "5G dataset."
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    st.subheader("Datasets")

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("#### 🔵 5G-NIDD — source / training domain")
            st.markdown(
                "5G mobile network intrusion detection dataset. "
                "Used for **training** and **in-domain testing**."
            )
            if baseline:
                rows = baseline["training_rows"] + baseline["testing_rows"]
                a, b, c = st.columns(3)
                a.metric("Processed records", f"{rows:,}")
                b.metric("Train / test", f"{baseline['training_rows']:,} / {baseline['testing_rows']:,}")
                if cm_5g:
                    c.metric("Malicious share", pct(cm_5g["malicious_share"], 1))

    with col2:
        with st.container(border=True):
            st.markdown("#### 🟠 CICIoT2023 — target / cross-domain")
            st.markdown(
                "IoT network attack dataset from a different lab environment "
                "(not 5G). Used **only for testing** — never for training."
            )
            if cross_json:
                a, b, c = st.columns(3)
                a.metric("Processed records", f"{cross_json['testing_rows']:,}")
                if attack_results is not None:
                    b.metric("Attack types", f"{len(attack_results)}")
                if cm_cic:
                    c.metric("Malicious share", pct(cm_cic["malicious_share"], 1))

    if cm_5g and cm_cic:
        st.caption(
            "Malicious shares are derived from the saved confusion matrices "
            "(5G-NIDD: stratified 20% test split; CICIoT2023: all processed "
            "records). The two environments have very different class balance, "
            "which affects how accuracy and precision should be read — see "
            "the Cross-Domain Evaluation page."
        )

    # --------------------------------------------------------
    # COMMON FEATURES
    # --------------------------------------------------------

    st.subheader("Common Feature Representation")

    st.write(
        "The two datasets use different schemas, so both were mapped onto the "
        "same seven features. This shared representation is what allows a "
        "model trained on 5G-NIDD to be applied to CICIoT2023 without "
        "retraining."
    )

    st.dataframe(
        pd.DataFrame({
            "Common feature": FEATURES,
            "5G-NIDD source (Argus flow record)": [
                "Rate",
                "TotPkts",
                "TotBytes / TotPkts",
                "sTtl",
                "Proto == tcp",
                "Proto == udp",
                "Proto in {icmp, ipv6-icmp}"
            ],
            "CICIoT2023 source": [
                "Rate",
                "Number",
                "AVG",
                "Time_To_Live",
                "TCP",
                "UDP",
                "ICMP"
            ],
            "Mapping caveat (from the processed data)": [
                "5G-NIDD median is 0; CICIoT2023 median is ~24,606",
                "CICIoT2023 values never exceed 100 (median 100); 5G-NIDD median is 2, max 3,997",
                "Both are bytes per packet; CICIoT2023 max (9,398) exceeds a standard 1,500-byte MTU",
                "Medians almost identical (63 vs 64); spread differs (std 55.9 vs 14.4)",
                "Binary (0/1) in 5G-NIDD; fractional in CICIoT2023 (median 0.99)",
                "Binary (0/1) in 5G-NIDD; fractional in CICIoT2023",
                "Binary (0/1) in 5G-NIDD; fractional in CICIoT2023",
            ]
        }),
        width="stretch",
        hide_index=True
    )

    st.caption(
        "Caveats are taken from results/feature_distribution_shift.csv. "
        "The fractional protocol values and the 100 cap on Packet_Count "
        "indicate that CICIoT2023 features are aggregated differently from "
        "5G-NIDD's per-flow records, so the mapped features are comparable "
        "in meaning but not identical in definition."
    )

    # --------------------------------------------------------
    # HEADLINE RESULT
    # --------------------------------------------------------

    if algorithms is not None:

        st.subheader("Headline Result")

        cols = st.columns(len(algorithms))
        for col, (_, row) in zip(cols, algorithms.iterrows()):
            with col:
                with st.container(border=True):
                    st.markdown(f"**{row['Model']}**")
                    st.metric("F1 on 5G-NIDD (in-domain)", pct(row["InDomain_F1"]))
                    st.metric(
                        "F1 on CICIoT2023 (cross-domain)",
                        pct(row["CrossDomain_F1"]),
                        delta=f"{-row['F1_Drop_pp']:.2f} pp",
                        delta_color="normal"
                    )

        st.warning(
            "All four algorithms reach moderate F1 on held-out 5G-NIDD traffic "
            "but almost completely fail to detect attacks in CICIoT2023 "
            "without retraining. Changing the algorithm does not fix this."
        )


# ============================================================
# MODEL PERFORMANCE (RANDOM FOREST, NUMBER OF TREES)
# ============================================================

elif page == "Model Performance":

    st.header("Random Forest: Effect of Model Size")

    st.write(
        "Random Forests with 10, 25, 50 and 100 trees were trained on the "
        "same 5G-NIDD 80/20 stratified split (random_state = 42) with the "
        "same seven features. The 100-tree model is the original baseline; "
        "the 10-tree model is the lightweight variant used in the "
        "Prediction Demo."
    )

    if lightweight is not None:

        df = lightweight.sort_values("Trees").reset_index(drop=True)

        display_df = pd.DataFrame({
            "Model": df["Model"],
            "Trees": df["Trees"].astype(int),
            "Accuracy (%)": (df["Accuracy"] * 100).round(2),
            "Precision (%)": (df["Precision"] * 100).round(2),
            "Recall (%)": (df["Recall"] * 100).round(2),
            "F1 (%)": (df["F1"] * 100).round(2),
            "Training time (s)": df["Training_Time_Seconds"].round(2),
            "Prediction time, 243,136 rows (s)": df["Prediction_Time_Seconds"].round(3),
            "Model size (MB)": df["Model_Size_MB"].round(2),
        })

        st.subheader("In-Domain Performance — 5G-NIDD held-out test set")

        st.dataframe(display_df, width="stretch", hide_index=True)

        # ----------------------------------------------------
        # METRICS BY NUMBER OF TREES
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("In-domain metrics by number of trees")

            fig, ax = plt.subplots(figsize=(7, 4.5))
            positions = np.arange(len(df))

            for metric, (color, marker) in METRIC_STYLE.items():
                ax.plot(
                    positions,
                    column(df, metric) * 100,
                    marker=marker,
                    markersize=8,
                    linewidth=2,
                    color=color,
                    label=metric
                )

            ax.set_xticks(positions)
            ax.set_xticklabels(df["Trees"].astype(int))
            ax.set_xlabel("Number of trees")
            ax.set_ylabel("Score on 5G-NIDD test set (%)")
            ax.set_ylim(0, 100)
            ax.grid(axis="y")
            ax.legend(loc="lower right", ncol=2)
            show(fig)

            st.caption(
                "Trees are plotted as ordered categories. Performance is "
                "essentially flat between 10 and 100 trees."
            )

        with col2:
            st.subheader("In-domain vs cross-domain F1 by number of trees")

            if cross_domain is not None:
                merged = df[["Trees", "F1"]].merge(
                    cross_domain[["Trees", "F1"]],
                    on="Trees",
                    suffixes=("_5G", "_CIC")
                ).sort_values("Trees")

                fig, ax = plt.subplots(figsize=(7, 4.5))
                positions = np.arange(len(merged))

                ax.plot(positions, column(merged, "F1_5G") * 100, marker="o",
                        markersize=8, linewidth=2, color=COLOR_5G,
                        label="5G-NIDD (in-domain)")
                ax.plot(positions, column(merged, "F1_CIC") * 100, marker="s",
                        markersize=8, linewidth=2, color=COLOR_CIC,
                        label="CICIoT2023 (cross-domain)")

                for i, value in enumerate(column(merged, "F1_CIC") * 100):
                    ax.annotate(f"{value:.2f}%", (i, value), textcoords="offset points",
                                xytext=(0, 8), ha="center", fontsize=9, color="#52514e")

                ax.set_xticks(positions)
                ax.set_xticklabels(merged["Trees"].astype(int))
                ax.set_xlabel("Number of trees")
                ax.set_ylabel("F1 score (%)")
                ax.set_ylim(0, 100)
                ax.grid(axis="y")
                ax.legend(loc="center right")
                show(fig)

                st.caption(
                    "Adding trees does not improve cross-domain F1."
                )

        # ----------------------------------------------------
        # MODEL SIZE / TIMES (separate charts, separate units)
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Model size")
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.bar(df["Trees"].astype(int).astype(str), column(df, "Model_Size_MB"),
                          color=COLOR_5G, width=0.6)
            ax.bar_label(bars, fmt="%.1f MB", padding=3, fontsize=9)
            ax.set_xlabel("Number of trees")
            ax.set_ylabel("Saved model size (MB)")
            ax.grid(axis="y")
            show(fig)

        with col2:
            st.subheader("Training time")
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.bar(df["Trees"].astype(int).astype(str), column(df, "Training_Time_Seconds"),
                          color=COLOR_5G, width=0.6)
            ax.bar_label(bars, fmt="%.1f s", padding=3, fontsize=9)
            ax.set_xlabel("Number of trees")
            ax.set_ylabel("Training time on 972,540 rows (s)")
            ax.grid(axis="y")
            show(fig)

        st.info(
            "The 10-tree forest keeps in-domain F1 at the level of the "
            "100-tree baseline while being about 10× smaller and faster to "
            "train. Note that “lightweight” is relative: at "
            f"{size_label(df.loc[df['Trees'] == 10, 'Model_Size_MB'].iloc[0])} it is still larger "
            "than the Decision Tree and XGBoost models on the Algorithm "
            "Comparison page, because its trees are grown without a depth limit."
        )

        # ----------------------------------------------------
        # FEATURE IMPORTANCE
        # ----------------------------------------------------

        if feature_importance is not None:
            st.subheader("Feature importance — 100-tree baseline")

            fi = feature_importance.sort_values("Importance")
            fig, ax = plt.subplots(figsize=(8, 3.8))
            bars = ax.barh(fi["Feature"], column(fi, "Importance") * 100,
                           color=COLOR_5G, height=0.6)
            ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
            ax.set_xlabel("Mean decrease in impurity (% of total)")
            ax.grid(axis="x")
            show(fig)

            st.caption(
                "Impurity-based importances from the 100-tree baseline trained "
                "on 5G-NIDD. They describe what the model relies on in the "
                "source domain; they do not by themselves explain the "
                "cross-domain result."
            )

    else:
        st.error("results/lightweight_model_comparison.csv could not be found.")


# ============================================================
# ALGORITHM COMPARISON
# ============================================================

elif page == "Algorithm Comparison":

    st.header("Machine Learning Algorithm Comparison")

    st.write(
        "Four algorithms were trained on the same 5G-NIDD training split "
        "with the same seven features, then tested on the same held-out "
        "5G-NIDD test set and on all CICIoT2023 records. The question is "
        "whether the cross-domain failure is specific to one algorithm."
    )

    if algorithms is not None:

        df = algorithms

        # ----------------------------------------------------
        # TABLE
        # ----------------------------------------------------

        st.subheader("Overall comparison")

        table = pd.DataFrame({
            "Model": df["Model"],
            "Model size": df["Model_Size_MB"].map(size_label),
            "Training time (s)": df["Training_Time_Sec"].round(2),
            "5G-NIDD Accuracy (%)": (df["InDomain_Accuracy"] * 100).round(2),
            "5G-NIDD Precision (%)": (df["InDomain_Precision"] * 100).round(2),
            "5G-NIDD Recall (%)": (df["InDomain_Recall"] * 100).round(2),
            "5G-NIDD F1 (%)": (df["InDomain_F1"] * 100).round(2),
            "CICIoT2023 Accuracy (%)": (df["CrossDomain_Accuracy"] * 100).round(2),
            "CICIoT2023 Precision (%)": (df["CrossDomain_Precision"] * 100).round(2),
            "CICIoT2023 Recall (%)": (df["CrossDomain_Recall"] * 100).round(2),
            "CICIoT2023 F1 (%)": (df["CrossDomain_F1"] * 100).round(2),
            "F1 drop (pp)": df["F1_Drop_pp"].round(2),
        })

        st.dataframe(table, width="stretch", hide_index=True)

        st.caption(
            "Random Forest (10 trees) values come from "
            "lightweight_model_comparison.csv and "
            "lightweight_cross_domain_comparison.csv; the other three come "
            "from other_models_comparison.csv (stored rounded to 4 decimals). "
            "pp = percentage points."
        )

        # ----------------------------------------------------
        # SMALL MULTIPLES: ONE PANEL PER ALGORITHM
        # ----------------------------------------------------

        st.subheader("In-domain vs cross-domain, per algorithm")

        metrics = ["Accuracy", "Precision", "Recall", "F1"]
        fig, axes = plt.subplots(1, len(df), figsize=(15, 4.6), sharey=True)
        axes = np.atleast_1d(axes)
        positions = np.arange(len(metrics))
        width = 0.38

        for ax, (_, row) in zip(axes, df.iterrows()):
            in_vals = [row[f"InDomain_{m}"] * 100 for m in metrics]
            cross_vals = [row[f"CrossDomain_{m}"] * 100 for m in metrics]

            b1 = ax.bar(positions - width / 2 - 0.01, in_vals, width,
                        color=COLOR_5G, label="5G-NIDD (in-domain)")
            b2 = ax.bar(positions + width / 2 + 0.01, cross_vals, width,
                        color=COLOR_CIC, label="CICIoT2023 (cross-domain)")
            ax.bar_label(b2, fmt="%.1f", padding=2, fontsize=8, color="#52514e")

            ax.set_title(row["Model"], fontsize=11)
            ax.set_xticks(positions)
            ax.set_xticklabels(metrics, fontsize=9)
            ax.set_ylim(0, 105)
            ax.grid(axis="y")

        axes[0].set_ylabel("Score (%)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2,
                   bbox_to_anchor=(0.5, 1.09))
        show(fig)

        st.caption(
            "Orange labels show the cross-domain values. Cross-domain "
            "precision stays fairly high only because almost every "
            "CICIoT2023 record is malicious — see the Cross-Domain "
            "Evaluation page for why that is not a good result."
        )

        # ----------------------------------------------------
        # F1 DROP
        # ----------------------------------------------------

        col1, col2 = st.columns([3, 2])

        with col1:
            st.subheader("F1: in-domain vs cross-domain")

            fig, ax = plt.subplots(figsize=(8, 4.4))
            positions = np.arange(len(df))
            width = 0.38

            b1 = ax.bar(positions - width / 2 - 0.01, column(df, "InDomain_F1") * 100, width,
                        color=COLOR_5G, label="5G-NIDD (in-domain)")
            b2 = ax.bar(positions + width / 2 + 0.01, column(df, "CrossDomain_F1") * 100, width,
                        color=COLOR_CIC, label="CICIoT2023 (cross-domain)")
            ax.bar_label(b1, fmt="%.2f%%", padding=2, fontsize=8.5)
            ax.bar_label(b2, fmt="%.2f%%", padding=2, fontsize=8.5)

            ax.set_xticks(positions)
            ax.set_xticklabels([m.replace(" (", "\n(") for m in df["Model"]], fontsize=9)
            ax.set_ylabel("F1 score (%)")
            ax.set_ylim(0, 100)
            ax.grid(axis="y")
            ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.12))
            show(fig)

        with col2:
            st.subheader("F1 drop")
            for _, row in df.iterrows():
                st.metric(
                    row["Model"],
                    f"{row['F1_Drop_pp']:.2f} pp",
                    help=(
                        f"{pct(row['InDomain_F1'])} → {pct(row['CrossDomain_F1'])}"
                    )
                )

        # ----------------------------------------------------
        # SIZE AND TRAINING TIME (separate units -> separate charts)
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Saved model size")
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.barh(df["Model"], column(df, "Model_Size_MB"),
                           color=COLOR_REFERENCE, height=0.55)
            ax.bar_label(bars, labels=[size_label(v) for v in df["Model_Size_MB"]],
                         padding=3, fontsize=9)
            ax.set_xlabel("Model size (MB)")
            ax.invert_yaxis()
            ax.grid(axis="x")
            show(fig)

        with col2:
            st.subheader("Training time")
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.barh(df["Model"], column(df, "Training_Time_Sec"),
                           color=COLOR_REFERENCE, height=0.55)
            ax.bar_label(bars, fmt="%.1f s", padding=3, fontsize=9)
            ax.set_xlabel("Training time on 972,540 rows (s)")
            ax.invert_yaxis()
            ax.grid(axis="x")
            show(fig)

        # ----------------------------------------------------
        # KEY OBSERVATIONS
        # ----------------------------------------------------

        st.subheader("Key observations")

        best_in = df.loc[df["InDomain_F1"].idxmax()]
        best_cross = df.loc[df["CrossDomain_F1"].idxmax()]

        observations = [
            f"In-domain F1 on 5G-NIDD ranges from **{df['InDomain_F1'].min() * 100:.2f}%** "
            f"to **{df['InDomain_F1'].max() * 100:.2f}%** (highest: {best_in['Model']}).",
            f"Cross-domain F1 on CICIoT2023 ranges from **{df['CrossDomain_F1'].min() * 100:.2f}%** "
            f"to **{df['CrossDomain_F1'].max() * 100:.2f}%** (highest: {best_cross['Model']}).",
            f"Every algorithm loses between **{df['F1_Drop_pp'].min():.2f}** and "
            f"**{df['F1_Drop_pp'].max():.2f}** percentage points of F1.",
            "Because a single tree, a forest, a boosted ensemble and a linear model all fail "
            "in the same way, the choice of algorithm alone does not explain or fix the "
            "cross-domain degradation.",
        ]

        lr = df[df["Model"] == "Logistic Regression"]
        if not lr.empty:
            observations.append(
                f"Logistic Regression's in-domain recall of {lr['InDomain_Recall'].iloc[0] * 100:.2f}% "
                "means it labels almost every 5G-NIDD test record as malicious, so its in-domain "
                "F1 is close to that of a trivial “always malicious” classifier (see the "
                "Cross-Domain Evaluation page). It was also trained on unscaled features."
            )

        st.markdown("\n".join(f"- {line}" for line in observations))

    else:
        st.error("The algorithm comparison result files could not be found.")


# ============================================================
# CROSS-DOMAIN EVALUATION
# ============================================================

elif page == "Cross-Domain Evaluation":

    st.header("Cross-Domain Evaluation: 5G-NIDD → CICIoT2023")

    st.write(
        "Each model was trained only on 5G-NIDD. It was then applied to "
        "every processed CICIoT2023 record with no retraining, no "
        "fine-tuning, no threshold change and no re-scaling. CICIoT2023 "
        "comes from a different (non-5G) IoT network environment."
    )

    # --------------------------------------------------------
    # MODEL SELECTOR
    # --------------------------------------------------------

    options = {}
    if algorithms is not None:
        for _, row in algorithms.iterrows():
            options[row["Model"]] = row
    if lightweight is not None and cross_domain is not None:
        rf100_in = lightweight[lightweight["Trees"] == 100]
        rf100_out = cross_domain[cross_domain["Trees"] == 100]
        if not rf100_in.empty and not rf100_out.empty:
            options["Random Forest (100 trees) — baseline"] = pd.Series({
                "InDomain_Accuracy": rf100_in.iloc[0]["Accuracy"],
                "InDomain_Precision": rf100_in.iloc[0]["Precision"],
                "InDomain_Recall": rf100_in.iloc[0]["Recall"],
                "InDomain_F1": rf100_in.iloc[0]["F1"],
                "CrossDomain_Accuracy": rf100_out.iloc[0]["Accuracy"],
                "CrossDomain_Precision": rf100_out.iloc[0]["Precision"],
                "CrossDomain_Recall": rf100_out.iloc[0]["Recall"],
                "CrossDomain_F1": rf100_out.iloc[0]["F1"],
            })

    if options:

        selected = st.selectbox("Model", list(options.keys()))
        row = options[selected]

        metrics = ["Accuracy", "Precision", "Recall", "F1"]

        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("**🔵 In-domain — 5G-NIDD held-out test set**")
                cols = st.columns(2) + st.columns(2)
                for c, m in zip(cols, metrics):
                    c.metric(m, pct(row[f"InDomain_{m}"]))
        with col2:
            with st.container(border=True):
                st.markdown("**🟠 Cross-domain — CICIoT2023 (no retraining)**")
                cols = st.columns(2) + st.columns(2)
                for c, m in zip(cols, metrics):
                    in_v = row[f"InDomain_{m}"]
                    out_v = row[f"CrossDomain_{m}"]
                    c.metric(m, pct(out_v), delta=f"{(out_v - in_v) * 100:+.2f} pp")

        fig, ax = plt.subplots(figsize=(10, 4))
        positions = np.arange(len(metrics))
        width = 0.38
        b1 = ax.bar(positions - width / 2 - 0.01, [row[f"InDomain_{m}"] * 100 for m in metrics],
                    width, color=COLOR_5G, label="5G-NIDD (in-domain)")
        b2 = ax.bar(positions + width / 2 + 0.01, [row[f"CrossDomain_{m}"] * 100 for m in metrics],
                    width, color=COLOR_CIC, label="CICIoT2023 (cross-domain)")
        ax.bar_label(b1, fmt="%.2f%%", padding=2, fontsize=9)
        ax.bar_label(b2, fmt="%.2f%%", padding=2, fontsize=9)
        ax.set_xticks(positions)
        ax.set_xticklabels(metrics)
        ax.set_ylabel("Score (%)")
        ax.set_ylim(0, 110)
        ax.set_title(selected)
        ax.grid(axis="y")
        ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.2))
        show(fig)

    # --------------------------------------------------------
    # CLASS BALANCE AND METRIC INTERPRETATION
    # --------------------------------------------------------

    if cm_5g and cm_cic:

        st.subheader("Reading the metrics correctly: class balance")

        st.write(
            "The two test sets have very different proportions of malicious "
            "traffic. This changes what “good” looks like for each metric."
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Malicious share — 5G-NIDD test", pct(cm_5g["malicious_share"]))
        col2.metric("Malicious share — CICIoT2023", pct(cm_cic["malicious_share"]))
        col3.metric(
            "CICIoT2023 benign records",
            f"{cm_cic['benign']:,}",
            help=f"out of {cm_cic['total']:,} records"
        )

        reference = pd.DataFrame({
            "Reference / model": [
                "Trivial: label everything malicious",
                "Random Forest (100 trees) — baseline",
            ],
            "5G-NIDD Accuracy (%)": [
                cm_5g["trivial_accuracy"] * 100,
                (cm_5g["TN"] + cm_5g["TP"]) / cm_5g["total"] * 100,
            ],
            "5G-NIDD F1 (%)": [
                cm_5g["trivial_f1"] * 100,
                baseline["f1_score"] * 100,
            ],
            "CICIoT2023 Accuracy (%)": [
                cm_cic["trivial_accuracy"] * 100,
                (cm_cic["TN"] + cm_cic["TP"]) / cm_cic["total"] * 100,
            ],
            "CICIoT2023 Precision (%)": [
                cm_cic["trivial_precision"] * 100,
                cross_json["precision"] * 100,
            ],
            "CICIoT2023 F1 (%)": [
                cm_cic["trivial_f1"] * 100,
                cross_json["f1_score"] * 100,
            ],
        }).round(2)

        st.dataframe(reference, width="stretch", hide_index=True)

        st.markdown(
            f"""
            - **Accuracy on CICIoT2023 is dominated by the
              {cm_cic['benign']:,} benign records.** The models predict
              nearly everything as benign, so the only records they get
              right are mostly the benign ones.
            - **Cross-domain precision is not a strength.** Because
              {pct(cm_cic['malicious_share'])} of CICIoT2023 records are
              malicious, *any* random selection of records would be about
              {pct(cm_cic['malicious_share'], 1)} malicious. Every model's
              cross-domain precision is below that base rate, so the few
              records they flag are not better than chance.
            - **Recall and F1 are the most informative metrics here:**
              recall shows that almost all attacks are missed.
            - **In-domain results are moderate, not strong.** On 5G-NIDD a
              trivial “always malicious” rule already scores
              {pct(cm_5g['trivial_f1'])} F1 and
              {pct(cm_5g['trivial_accuracy'])} accuracy. The models are
              better than this, but by a limited margin.
            """
        )

        # ----------------------------------------------------
        # CONFUSION MATRICES (100-tree baseline: only model with saved CMs)
        # ----------------------------------------------------

        st.subheader("Confusion matrices — Random Forest (100 trees) baseline")

        st.caption(
            "Confusion matrices were saved only for the 100-tree baseline "
            "(results/baseline_5g_results.json and "
            "results/cross_domain_results.json)."
        )

        def plot_cm(ax, summary, title, cmap):
            matrix = np.array([[summary["TN"], summary["FP"]],
                               [summary["FN"], summary["TP"]]])
            row_share = matrix / matrix.sum(axis=1, keepdims=True)
            ax.imshow(row_share, cmap=cmap, vmin=0, vmax=1)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{matrix[i, j]:,}\n({row_share[i, j] * 100:.1f}% of row)",
                            ha="center", va="center", fontsize=10,
                            color="white" if row_share[i, j] > 0.6 else "#1f1f1f")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Predicted benign", "Predicted malicious"])
            ax.set_yticklabels(["Actual benign", "Actual malicious"])
            ax.set_title(title, fontsize=11.5)
            for spine in ax.spines.values():
                spine.set_visible(False)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        plot_cm(axes[0], cm_5g, "5G-NIDD held-out test (in-domain)", "Blues")
        plot_cm(axes[1], cm_cic, "CICIoT2023 (cross-domain)", "Oranges")
        show(fig)

        derived = pd.DataFrame({
            "Derived from confusion matrix": [
                "Share of records predicted malicious",
                "Benign recall (specificity)",
                "Malicious recall (detection rate)",
                "Balanced accuracy",
            ],
            "5G-NIDD (%)": [
                cm_5g["predicted_malicious_share"] * 100,
                cm_5g["specificity"] * 100,
                cm_5g["recall"] * 100,
                cm_5g["balanced_accuracy"] * 100,
            ],
            "CICIoT2023 (%)": [
                cm_cic["predicted_malicious_share"] * 100,
                cm_cic["specificity"] * 100,
                cm_cic["recall"] * 100,
                cm_cic["balanced_accuracy"] * 100,
            ],
        }).round(2)

        st.dataframe(derived, width="stretch", hide_index=True)

        st.info(
            f"The model's behaviour flips between environments: it labels "
            f"{pct(cm_5g['predicted_malicious_share'], 1)} of 5G-NIDD test "
            f"records as malicious, but only "
            f"{pct(cm_cic['predicted_malicious_share'], 1)} of CICIoT2023 "
            f"records. The CICIoT2023 traffic falls into regions of the "
            f"feature space that the 5G-trained model associates with benign "
            f"traffic."
        )

    # --------------------------------------------------------
    # RANDOM FOREST TREES TABLE
    # --------------------------------------------------------

    if cross_domain is not None:
        st.subheader("Random Forest cross-domain results by number of trees")

        df = cross_domain.sort_values("Trees")
        st.dataframe(
            pd.DataFrame({
                "Model": df["Model"],
                "Accuracy (%)": (df["Accuracy"] * 100).round(2),
                "Precision (%)": (df["Precision"] * 100).round(2),
                "Recall (%)": (df["Recall"] * 100).round(2),
                "F1 (%)": (df["F1"] * 100).round(2),
                "Prediction time, 3,579,528 rows (s)": df["Prediction_Time_Seconds"].round(2),
                "Model size (MB)": df["Model_Size_MB"].round(2),
            }),
            width="stretch",
            hide_index=True
        )

    st.warning(
        "Conclusion: models that work moderately well on 5G-NIDD suffer a "
        "severe loss of detection performance when transferred to "
        "CICIoT2023 without retraining. This result is part of the "
        "finding, not a failure of the experiment."
    )


# ============================================================
# ATTACK ANALYSIS
# ============================================================

elif page == "Attack Analysis":

    st.header("Attack-Type Analysis — CICIoT2023")

    # 5G-NIDD contains these attack types (per the dataset documentation):
    # ICMP Flood, UDP Flood, SYN Flood, HTTP Flood, Slowrate DoS,
    # SYN Scan, TCP Connect Scan, UDP Scan.
    # The mapping below links CICIoT2023 labels to the closest 5G-NIDD
    # attack family. Edit it if your copy of 5G-NIDD differs.
    COMPARABLE_IN_5G = {
        "DDOS-ICMP_FLOOD": "ICMP Flood",
        "DDOS-UDP_FLOOD": "UDP Flood",
        "DOS-UDP_FLOOD": "UDP Flood",
        "DDOS-SYN_FLOOD": "SYN Flood",
        "DOS-SYN_FLOOD": "SYN Flood",
        "DDOS-HTTP_FLOOD": "HTTP Flood",
        "DOS-HTTP_FLOOD": "HTTP Flood",
        "DDOS-SLOWLORIS": "Slowrate DoS",
        "RECON-PORTSCAN": "SYN / TCP Connect / UDP Scan",
    }

    def cic_category(label):
        label = label.upper()
        if label.startswith("DDOS"):
            return "DDoS"
        if label.startswith("DOS"):
            return "DoS"
        if label.startswith("MIRAI"):
            return "Mirai"
        if label.startswith("RECON") or label == "VULNERABILITYSCAN":
            return "Reconnaissance"
        if label in {"DNS_SPOOFING", "MITM-ARPSPOOFING"}:
            return "Spoofing"
        if label == "DICTIONARYBRUTEFORCE":
            return "Brute force"
        return "Web-based"

    if has_columns(attack_results, ["Attack_Type", "Samples", "Detected_Malicious", "Detection_Rate"]):

        df = attack_results.copy()
        df["Category"] = df["Attack_Type"].map(cic_category)
        df["Comparable 5G-NIDD attack"] = df["Attack_Type"].map(COMPARABLE_IN_5G).fillna("—")

        total = df["Samples"].sum()
        detected = df["Detected_Malicious"].sum()

        st.write(
            "Detection rate = share of records of each CICIoT2023 attack type "
            "that the model labelled as malicious. Results were produced by "
            "**src/test_attack_types.py** using the **100-tree Random Forest "
            "baseline** trained on 5G-NIDD (the per-type totals match the "
            "baseline's cross-domain confusion matrix)."
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Attack types", f"{len(df)}")
        col2.metric("Attack records", f"{total:,}")
        col3.metric("Overall detection rate", pct(detected / total))
        col4.metric("Highest single-type rate", pct(df["Detection_Rate"].max()))

        # ----------------------------------------------------
        # BY CATEGORY
        # ----------------------------------------------------

        st.subheader("Detection rate by attack category")

        cat = df.groupby("Category", as_index=False)[["Samples", "Detected_Malicious"]].sum()
        cat["Detection_Rate"] = cat["Detected_Malicious"] / cat["Samples"]
        cat = cat.sort_values("Detection_Rate")

        fig, ax = plt.subplots(figsize=(10, 3.8))
        bars = ax.barh(cat["Category"], column(cat, "Detection_Rate") * 100,
                       color=COLOR_CIC, height=0.6)
        ax.bar_label(
            bars,
            labels=[f"{r * 100:.2f}%  (n = {n:,})" for r, n in zip(cat["Detection_Rate"], cat["Samples"])],
            padding=3, fontsize=9
        )
        ax.set_xlabel("Detection rate (%) — weighted by number of records")
        ax.set_xlim(0, max(10, cat["Detection_Rate"].max() * 100 * 1.6))
        ax.grid(axis="x")
        show(fig)

        # ----------------------------------------------------
        # PER ATTACK TYPE
        # ----------------------------------------------------

        st.subheader("Detection rate by attack type")

        plot_df = df.sort_values("Detection_Rate")
        comparable = plot_df["Attack_Type"].isin(COMPARABLE_IN_5G.keys())

        fig, ax = plt.subplots(figsize=(10, 11))
        bars = ax.barh(plot_df["Attack_Type"], column(plot_df, "Detection_Rate") * 100,
                       color=COLOR_CIC, height=0.65, edgecolor="white", linewidth=0.5)
        for bar, is_comparable in zip(bars, comparable):
            if is_comparable:
                bar.set_hatch("///")
                bar.set_edgecolor("#7a2e10")
        ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=8.5)
        for tick, is_comparable in zip(ax.get_yticklabels(), comparable):
            if is_comparable:
                tick.set_fontweight("bold")
        ax.set_xlabel("Detection rate (%)")
        ax.set_xlim(0, max(10, plot_df["Detection_Rate"].max() * 100 * 1.25))
        ax.grid(axis="x")
        from matplotlib.patches import Patch
        ax.legend(
            handles=[
                Patch(facecolor=COLOR_CIC, label="No comparable attack in 5G-NIDD"),
                Patch(facecolor=COLOR_CIC, hatch="///", edgecolor="#7a2e10",
                      label="Comparable attack family exists in 5G-NIDD (bold label)"),
            ],
            loc="lower right"
        )
        show(fig)

        st.caption(
            "The x-axis is limited to the observed range so differences are "
            "visible: no attack type exceeds "
            f"{df['Detection_Rate'].max() * 100:.2f}% detection."
        )

        # ----------------------------------------------------
        # SEEN VS UNSEEN ATTACK FAMILIES
        # ----------------------------------------------------

        st.subheader("Attacks with a comparable family in 5G-NIDD")

        comp = df[df["Attack_Type"].isin(COMPARABLE_IN_5G.keys())]
        other = df[~df["Attack_Type"].isin(COMPARABLE_IN_5G.keys())]

        col1, col2 = st.columns(2)
        col1.metric(
            "Comparable to a 5G-NIDD attack",
            pct(comp["Detected_Malicious"].sum() / comp["Samples"].sum()),
            help=f"{len(comp)} attack types, {comp['Samples'].sum():,} records"
        )
        col2.metric(
            "No comparable 5G-NIDD attack",
            pct(other["Detected_Malicious"].sum() / other["Samples"].sum()),
            help=f"{len(other)} attack types, {other['Samples'].sum():,} records"
        )

        st.dataframe(
            comp[["Attack_Type", "Comparable 5G-NIDD attack", "Samples",
                  "Detected_Malicious", "Detection_Rate"]]
            .assign(Detection_Rate=lambda d: (d["Detection_Rate"] * 100).round(3))
            .rename(columns={"Detection_Rate": "Detection rate (%)",
                             "Detected_Malicious": "Detected",
                             "Attack_Type": "CICIoT2023 attack"})
            .sort_values("Detection rate (%)"),
            width="stretch",
            hide_index=True
        )

        st.markdown(
            """
            **Interpretation**

            - Detection is very low for **every** category; the model does
              not fail equally, but the differences are between near-zero
              and single-digit percentages.
            - High-volume flood attacks (DDoS, DoS, Mirai) are almost never
              detected. Small-volume reconnaissance, spoofing and web-based
              attacks are flagged slightly more often.
            - Even attack families that **also exist in 5G-NIDD** (SYN, UDP,
              ICMP and HTTP floods, slow-rate DoS, port scans) are largely
              missed. This suggests the failure is not only about unseen
              attack types: the same kind of attack looks different in the
              common feature space when it comes from a different network
              environment.
            """
        )

        with st.expander("Full attack-type table"):
            st.dataframe(
                df[["Attack_Type", "Category", "Comparable 5G-NIDD attack", "Samples",
                    "Detected_Malicious", "Missed_As_Benign", "Detection_Rate"]]
                .assign(Detection_Rate=lambda d: (d["Detection_Rate"] * 100).round(4))
                .rename(columns={"Detection_Rate": "Detection rate (%)"}),
                width="stretch",
                hide_index=True
            )

        st.caption(
            "Comparable-family mapping is based on the attack types listed "
            "in the 5G-NIDD dataset documentation (ICMP/UDP/SYN/HTTP flood, "
            "slow-rate DoS, SYN/TCP-connect/UDP scans). CICIoT2023 categories "
            "follow the dataset's seven attack groups."
        )

    else:
        st.info("results/attack_type_results.csv could not be found.")


# ============================================================
# FEATURE SHIFT
# ============================================================

elif page == "Feature Shift":

    st.header("Feature and Protocol Distribution Shift")

    st.write(
        "Domain shift means the traffic seen during training differs "
        "statistically from the traffic seen during deployment. Here the "
        "seven common features are compared between 5G-NIDD (training "
        "environment) and CICIoT2023 (deployment environment)."
    )

    required = ["Feature", "5G_Mean", "CIC_Mean", "5G_Median", "CIC_Median", "5G_Std", "CIC_Std"]

    if has_columns(feature_shift, required):

        fs = feature_shift.set_index("Feature").reindex(FEATURES).reset_index()

        # Standardised mean difference (descriptive effect size):
        # (CIC mean - 5G mean) / pooled standard deviation
        pooled = np.sqrt((column(fs, "5G_Std") ** 2 + column(fs, "CIC_Std") ** 2) / 2)
        fs["SMD"] = (column(fs, "CIC_Mean") - column(fs, "5G_Mean")) / pooled

        # ----------------------------------------------------
        # TABLE
        # ----------------------------------------------------

        st.subheader("Summary statistics")

        table = pd.DataFrame({
            "Feature": fs["Feature"],
            "5G-NIDD median": fs["5G_Median"],
            "CICIoT2023 median": fs["CIC_Median"],
            "5G-NIDD mean": fs["5G_Mean"],
            "CICIoT2023 mean": fs["CIC_Mean"],
            "5G-NIDD std": fs["5G_Std"],
            "CICIoT2023 std": fs["CIC_Std"],
            "5G-NIDD min–max": [f"{a:,.4g} – {b:,.4g}" for a, b in zip(fs["5G_Min"], fs["5G_Max"])],
            "CICIoT2023 min–max": [f"{a:,.4g} – {b:,.4g}" for a, b in zip(fs["CIC_Min"], fs["CIC_Max"])],
            "Std. mean difference": fs["SMD"],
        })

        st.dataframe(
            table,
            width="stretch",
            hide_index=True,
            column_config={
                c: st.column_config.NumberColumn(format="%.3f")
                for c in table.columns if c not in ("Feature", "5G-NIDD min–max", "CICIoT2023 min–max")
            }
        )

        st.caption(
            "Source: results/feature_distribution_shift.csv. The standardised "
            "mean difference is derived from the mean and standard deviation "
            "columns: (CICIoT2023 mean − 5G-NIDD mean) / pooled std. It is a "
            "descriptive effect size, not a statistical test."
        )

        # ----------------------------------------------------
        # STANDARDISED DIFFERENCE CHART
        # ----------------------------------------------------

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Size of the shift per feature")

            plot = fs.iloc[::-1]
            smd = column(plot, "SMD")
            colors = [COLOR_CIC if v > 0 else COLOR_5G for v in smd]

            fig, ax = plt.subplots(figsize=(7, 4.6))
            bars = ax.barh(plot["Feature"], smd, color=colors, height=0.6)
            ax.bar_label(bars, fmt="%+.2f", padding=3, fontsize=9)
            ax.axvline(0, color="#52514e", linewidth=1)
            ax.set_xlabel("Standardised mean difference (pooled std units)")
            limit = np.nanmax(np.abs(smd)) * 1.3
            ax.set_xlim(-limit, limit)
            ax.grid(axis="x")
            from matplotlib.patches import Patch
            ax.legend(
                handles=[
                    Patch(color=COLOR_CIC, label="Higher in CICIoT2023"),
                    Patch(color=COLOR_5G, label="Higher in 5G-NIDD"),
                ],
                loc="upper left"
            )
            show(fig)

        with col2:
            st.subheader("Medians of the continuous features")

            fig, axes = plt.subplots(2, 2, figsize=(7, 4.6))
            for ax, feature in zip(axes.flat, CONTINUOUS_FEATURES):
                row = fs[fs["Feature"] == feature].iloc[0]
                values = [row["5G_Median"], row["CIC_Median"]]
                bars = ax.bar(["5G-NIDD", "CICIoT2023"], values,
                              color=[COLOR_5G, COLOR_CIC], width=0.55)
                ax.bar_label(bars, labels=[f"{v:,.0f}" if v >= 100 else f"{v:,.4g}" for v in values],
                             padding=2, fontsize=8.5)
                ax.set_title(feature, fontsize=10.5)
                ax.set_ylim(0, max(values) * 1.3 if max(values) > 0 else 1)
                ax.tick_params(axis="x", labelsize=8.5)
                ax.tick_params(axis="y", labelsize=8)
                ax.grid(axis="y")
            show(fig)

            st.caption("Each panel has its own y-axis scale because the features have different units.")

        # ----------------------------------------------------
        # DATA-DRIVEN OBSERVATIONS
        # ----------------------------------------------------

        def stat(feature, name):
            return fs.loc[fs["Feature"] == feature, name].iloc[0]

        st.subheader("Observations from the data")

        st.markdown(
            f"""
            - **Rate:** median {stat('Rate', '5G_Median'):,.0f} in 5G-NIDD vs
              {stat('Rate', 'CIC_Median'):,.0f} in CICIoT2023; the mean is
              {stat('Rate', 'CIC_Mean') / stat('Rate', '5G_Mean'):.1f}× higher in
              CICIoT2023. Most 5G-NIDD records have a rate of zero.
            - **Packet_Count:** median {stat('Packet_Count', '5G_Median'):,.0f} vs
              {stat('Packet_Count', 'CIC_Median'):,.0f} — the largest standardised
              shift ({stat('Packet_Count', 'SMD'):+.2f}). CICIoT2023 values are
              capped at {stat('Packet_Count', 'CIC_Max'):,.0f}, which points to a
              different definition of this field as well as different traffic.
            - **Mean_Packet_Size:** median {stat('Mean_Packet_Size', '5G_Median'):,.0f}
              vs {stat('Mean_Packet_Size', 'CIC_Median'):,.0f} bytes — the
              smallest standardised shift ({stat('Mean_Packet_Size', 'SMD'):+.2f}).
            - **TTL:** medians almost identical
              ({stat('TTL', '5G_Median'):,.0f} vs {stat('TTL', 'CIC_Median'):,.0f}),
              but the spread differs strongly (std {stat('TTL', '5G_Std'):.1f} vs
              {stat('TTL', 'CIC_Std'):.1f}). TTL is the most important feature
              of the 5G-trained Random Forest.
            - **Protocols:** the dominant protocol changes from UDP in 5G-NIDD
              to TCP in CICIoT2023 (see below).
            """
        )

    else:
        st.error(
            "results/feature_distribution_shift.csv is missing or does not "
            "have the expected columns: " + ", ".join(required)
        )

    # --------------------------------------------------------
    # PROTOCOL SHIFT
    # --------------------------------------------------------

    st.subheader("Protocol distribution shift")

    if has_columns(protocol_shift, ["Protocol", "5G_NIDD_Mean", "CICIoT2023_Mean"]):

        ps = protocol_shift.copy()

        chart_col, table_col = st.columns([3, 2])

        fig, ax = plt.subplots(figsize=(7, 4))
        positions = np.arange(len(ps))
        width = 0.38
        b1 = ax.bar(positions - width / 2 - 0.01, column(ps, "5G_NIDD_Mean") * 100, width,
                    color=COLOR_5G, label="5G-NIDD")
        b2 = ax.bar(positions + width / 2 + 0.01, column(ps, "CICIoT2023_Mean") * 100, width,
                    color=COLOR_CIC, label="CICIoT2023")
        ax.bar_label(b1, fmt="%.1f%%", padding=2, fontsize=9)
        ax.bar_label(b2, fmt="%.1f%%", padding=2, fontsize=9)
        ax.set_xticks(positions)
        ax.set_xticklabels(ps["Protocol"])
        ax.set_ylabel("Mean of protocol indicator (%)")
        ax.set_ylim(0, 100)
        ax.grid(axis="y")
        ax.legend(loc="upper right")
        with chart_col:
            show(fig)

        table_col.dataframe(
            pd.DataFrame({
                "Protocol": ps["Protocol"],
                "5G-NIDD (%)": (ps["5G_NIDD_Mean"] * 100).round(2),
                "CICIoT2023 (%)": (ps["CICIoT2023_Mean"] * 100).round(2),
                "Difference (pp)": ((ps["CICIoT2023_Mean"] - ps["5G_NIDD_Mean"]) * 100).round(2),
            }),
            width="stretch",
            hide_index=True
        )

        st.caption(
            "For 5G-NIDD the indicator is 0/1 per flow, so the value is the "
            "share of flows using that protocol. In CICIoT2023 the indicator "
            "is fractional, so the value is the average indicator rather than "
            "a strict share of records."
        )

    else:
        st.error("results/protocol_distribution_shift.csv is missing or has unexpected columns.")

    st.info(
        "**Interpretation.** Feature and protocol distribution differences "
        "provide evidence of domain shift between the 5G-NIDD and CICIoT2023 "
        "environments that may contribute to the observed cross-domain "
        "performance degradation. These statistics are descriptive: they do "
        "not prove which feature causes the failure, and other factors — "
        "different attack types, different class balance, and differences in "
        "how each dataset defines its features — are also likely to "
        "contribute."
    )


# ============================================================
# PREDICTION DEMO
# ============================================================

elif page == "Prediction Demo":

    st.header("Prediction Demo — 10-tree Random Forest")

    st.write(
        "Classify a single traffic record with the lightweight 10-tree "
        "Random Forest trained on 5G-NIDD. This is a demonstration of the "
        "trained model, not a production IDS."
    )

    if not LIGHTWEIGHT_MODEL.exists():

        st.warning(
            "The model file models/random_forest_10_trees.joblib is not "
            "available in this deployment. Model files are excluded from Git "
            "because of their size. Run `python src/test_lightweight_models.py` "
            "locally to create it. All evaluation pages still work."
        )

    else:

        model = load_model(LIGHTWEIGHT_MODEL)

        # Default inputs: 5G-NIDD medians from feature_distribution_shift.csv
        defaults = {"Rate": 0.0, "Packet_Count": 1.0, "Mean_Packet_Size": 0.0, "TTL": 64.0, "Protocol": "UDP"}
        ranges = {}
        if has_columns(feature_shift, ["Feature", "5G_Median", "5G_Min", "5G_Max"]):
            fs = feature_shift.set_index("Feature")
            for f in CONTINUOUS_FEATURES:
                if f in fs.index:
                    defaults[f] = float(fs.loc[f, "5G_Median"])
                    ranges[f] = (float(fs.loc[f, "5G_Min"]), float(fs.loc[f, "5G_Max"]))

        for key in CONTINUOUS_FEATURES + ["Protocol"]:
            st.session_state.setdefault(f"demo_{key}", defaults[key])

        # ----------------------------------------------------
        # REAL SAMPLES
        # ----------------------------------------------------

        if has_columns(demo_samples, FEATURES + ["Target"]):

            def load_sample():
                choice = st.session_state["demo_sample"]
                if choice == "Manual input":
                    return
                sample = demo_samples.iloc[int(choice.split("#")[1].split(" ")[0])]
                for f in CONTINUOUS_FEATURES:
                    st.session_state[f"demo_{f}"] = float(sample[f])
                st.session_state["demo_Protocol"] = (
                    "TCP" if sample["TCP"] == 1 else
                    "UDP" if sample["UDP"] == 1 else
                    "ICMP" if sample["ICMP"] == 1 else "Other"
                )

            labels = ["Manual input"] + [
                f"Sample #{i} (true label: {'Malicious' if t == 1 else 'Benign'})"
                for i, t in enumerate(demo_samples["Target"])
            ]
            st.selectbox(
                "Load a real record from the 5G-NIDD held-out test split",
                labels,
                key="demo_sample",
                on_change=load_sample
            )
            st.caption(
                "Samples come from results/demo_samples_5g_nidd.csv, created by "
                "src/extract_demo_samples.py from the same held-out test split "
                "used for evaluation (never from the training data)."
            )
        else:
            st.caption(
                "Default values are the 5G-NIDD medians from "
                "results/feature_distribution_shift.csv. To load real "
                "5G-NIDD test records, run `python src/extract_demo_samples.py`."
            )

        # ----------------------------------------------------
        # INPUTS
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:
            rate = st.number_input("Rate (packets per second)", min_value=0.0,
                                   key="demo_Rate", format="%.4f")
            packet_count = st.number_input("Packet Count (packets in the flow)", min_value=0.0,
                                           key="demo_Packet_Count")
            mean_packet_size = st.number_input("Mean Packet Size (bytes per packet)", min_value=0.0,
                                               key="demo_Mean_Packet_Size")

        with col2:
            ttl = st.number_input("TTL (source time-to-live)", min_value=0.0, max_value=255.0,
                                  key="demo_TTL")
            protocol = st.radio(
                "Protocol",
                ["TCP", "UDP", "ICMP", "Other"],
                key="demo_Protocol",
                horizontal=True,
                help=(
                    "In 5G-NIDD exactly one of TCP/UDP/ICMP is 1 per flow "
                    "(all three are 0 for other protocols), so the demo "
                    "uses the same one-hot encoding the model was trained on."
                )
            )

        values = {
            "Rate": rate,
            "Packet_Count": packet_count,
            "Mean_Packet_Size": mean_packet_size,
            "TTL": ttl,
            "TCP": 1.0 if protocol == "TCP" else 0.0,
            "UDP": 1.0 if protocol == "UDP" else 0.0,
            "ICMP": 1.0 if protocol == "ICMP" else 0.0,
        }

        outside = [
            f for f, (lo, hi) in ranges.items()
            if not (lo <= values[f] <= hi)
        ]
        if outside:
            st.warning(
                "Outside the range seen in 5G-NIDD: " + ", ".join(outside)
                + ". The model's output for these values is an extrapolation."
            )

        if st.button("🔍 Classify record", type="primary"):

            input_data = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)

            prediction = int(model.predict(input_data)[0])
            classes = list(model.classes_)
            malicious_probability = float(model.predict_proba(input_data)[0][classes.index(1)])

            col1, col2 = st.columns([2, 1])
            with col1:
                if prediction == 1:
                    st.error("🚨 Classified as MALICIOUS")
                else:
                    st.success("✅ Classified as BENIGN")
            with col2:
                st.metric(
                    "Share of trees voting malicious",
                    f"{malicious_probability * 100:.0f}%",
                    help="Random Forest predict_proba: the fraction of the 10 trees that vote malicious."
                )

            choice = st.session_state.get("demo_sample", "Manual input")
            if choice != "Manual input":
                st.caption(f"Selected record — {choice.split('(')[1].rstrip(')')}.")

            st.dataframe(input_data, width="stretch", hide_index=True)

        st.caption(
            "Remember the main finding: this model was only validated on 5G-NIDD. "
            "Its predictions on traffic from other networks (such as CICIoT2023) "
            "are unreliable."
        )


# ============================================================
# LIVE IDS MONITORING
# ============================================================

COLOR_BENIGN = "#0ca30c"
COLOR_MALICIOUS = "#d03b3b"
VERDICT_SCALE = alt.Scale(
    domain=["BENIGN", "MALICIOUS", "NO TRAFFIC"],
    range=[COLOR_BENIGN, COLOR_MALICIOUS, COLOR_REFERENCE],
)
VERDICT_SHAPES = alt.Scale(
    domain=["BENIGN", "MALICIOUS", "NO TRAFFIC"],
    range=["circle", "triangle-up", "square"],
)

LIVE_INFO = (
    "This live demonstration uses a trained machine-learning IDS model to "
    "analyse network traffic generated within a controlled test environment. "
    "The model was trained previously using 5G-NIDD data and is not retrained "
    "during live monitoring."
)


@st.cache_resource
def load_live_predictor(path):
    # Re-use the model already cached for the Prediction Demo; LivePredictor
    # checks feature names/order and class labels before it is used.
    model = load_model(path) if path.exists() else None
    return LivePredictor(path, model=model)


@st.cache_data(ttl=30, show_spinner=False)
def cached_interfaces():
    return list_interfaces()


def live_monitor():
    return st.session_state.get("live_monitor")


def monitor_running():
    monitor = live_monitor()
    return monitor is not None and monitor.running


def probability_chart(flows, windows):
    points = alt.Chart(flows).mark_point(size=70, filled=True, opacity=0.85).encode(
        x=alt.X("Time:T", title="Time"),
        y=alt.Y("Malicious_Probability:Q", title="Malicious probability",
                scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        color=alt.Color("Prediction:N", scale=VERDICT_SCALE, title="Flow prediction"),
        shape=alt.Shape("Prediction:N", scale=VERDICT_SHAPES, title="Flow prediction"),
        tooltip=[
            alt.Tooltip("Time:T", format="%H:%M:%S"), "Source:N", "Destination:N",
            "Protocol:N", "Prediction:N",
            alt.Tooltip("Malicious_Probability:Q", title="Malicious probability", format=".0%"),
            alt.Tooltip("Packet_Count:Q", title="Packets"),
            alt.Tooltip("Rate:Q", title="Rate (pkt/s)", format=".2f"),
        ],
    )
    peak = alt.Chart(windows).mark_line(color=COLOR_REFERENCE, strokeWidth=2).encode(
        x="Time:T",
        y="Max_Malicious_Probability:Q",
        tooltip=[alt.Tooltip("Time:T", format="%H:%M:%S"),
                 alt.Tooltip("Max_Malicious_Probability:Q", title="Highest flow probability", format=".0%")],
    )
    threshold = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(
        strokeDash=[5, 4], color="#52514e"
    ).encode(y="y:Q")
    return (threshold + peak + points).properties(height=290)


def rate_chart(windows):
    base = alt.Chart(windows).encode(
        x=alt.X("Time:T", title="Time"),
        y=alt.Y("Traffic_Rate:Q", title="Traffic rate (packets / second)"),
    )
    line = base.mark_line(color=COLOR_REFERENCE, strokeWidth=2)
    dots = base.mark_point(size=80, filled=True).encode(
        color=alt.Color("Verdict:N", scale=VERDICT_SCALE, title="Window verdict"),
        shape=alt.Shape("Verdict:N", scale=VERDICT_SHAPES, title="Window verdict"),
        tooltip=[
            alt.Tooltip("Time:T", format="%H:%M:%S"), "Verdict:N",
            alt.Tooltip("Traffic_Rate:Q", title="Rate (pkt/s)", format=".1f"),
            alt.Tooltip("Packet_Count:Q", title="Packets"),
            alt.Tooltip("Flows:Q", title="Flows"),
            alt.Tooltip("Flagged_Flows:Q", title="Flagged flows"),
        ],
    )
    return (line + dots).properties(height=290)


def flow_table(rows):
    table = pd.DataFrame(rows)
    table["Prediction"] = table["Prediction"].map(
        {"MALICIOUS": "🔴 MALICIOUS", "BENIGN": "🟢 BENIGN"}
    )
    table["Malicious_Probability"] = table["Malicious_Probability"] * 100
    columns = ["Time", "Source", "Destination", "Protocol"] + FEATURES + [
        "Prediction", "Malicious_Probability"
    ]
    return table[columns]


FLOW_COLUMNS = {
    "Time": st.column_config.DatetimeColumn("Time", format="HH:mm:ss"),
    "Packet_Count": st.column_config.NumberColumn("Packet Count", format="%d"),
    "Rate": st.column_config.NumberColumn("Rate (pkt/s)", format="%.2f"),
    "Mean_Packet_Size": st.column_config.NumberColumn("Mean Packet Size (B)", format="%.1f"),
    "TTL": st.column_config.NumberColumn("TTL", format="%d"),
    "Malicious_Probability": st.column_config.ProgressColumn(
        "Malicious Probability", min_value=0, max_value=100, format="%.2f%%"
    ),
}


def render_live_page():

    st.header("Live IDS Monitoring")
    st.info(LIVE_INFO, icon="ℹ️")

    if LIVE_IMPORT_ERROR is not None:
        st.error(
            f"The live IDS modules (src/live_*.py) could not be loaded: "
            f"{LIVE_IMPORT_ERROR}. All other pages still work."
        )
        return

    try:
        predictor = load_live_predictor(LIGHTWEIGHT_MODEL)
    except ModelCompatibilityError as error:
        st.error(f"The live IDS needs the trained 10-tree Random Forest. {error}")
        return

    running = monitor_running()

    # --------------------------------------------------------
    # STATUS AND CONTROLS
    # --------------------------------------------------------

    status_html = (
        '<span class="tag tag-live">🟢 Monitoring</span>' if running
        else '<span class="tag tag-stop">⏸️ Stopped</span>'
    )
    if running and live_monitor().mode == "demo":
        status_html += '<span class="tag tag-sim">SIMULATION — NOT REAL NETWORK TRAFFIC</span>'
    st.markdown(f"**Status:** {status_html}", unsafe_allow_html=True)

    mode = st.radio(
        "Mode",
        ["Live Capture Mode", "Demo Mode"],
        horizontal=True,
        disabled=running,
        key="live_mode",
        help="Live Capture analyses real packets from this laptop's network "
             "interface. Demo Mode generates synthetic packets in memory.",
    )

    interface = host_filter = None
    scenario = "auto"
    capture_ready = True

    col1, col2 = st.columns(2)

    if mode == "Live Capture Mode":
        available, capture_message = capture_status()
        if not available:
            capture_ready = False
            st.warning(
                f"**Live capture is not available.** {capture_message}  \n"
                "You can still present the dashboard with **Demo Mode**."
            )
        else:
            try:
                interfaces = cached_interfaces()
            except Exception as error:
                interfaces = []
                st.warning(f"Network interfaces could not be listed: {error}")
            if not interfaces:
                capture_ready = False
                st.warning(
                    "No capture interfaces were found. Check that Npcap is "
                    "installed and the network adapter is enabled."
                )
            else:
                labels = {item["label"]: item["name"] for item in interfaces}
                with col1:
                    chosen = st.selectbox(
                        "Network interface", list(labels), disabled=running,
                        key="live_interface",
                        help="Choose the adapter the test traffic passes through: "
                             "your Wi-Fi adapter, or the 'Wi-Fi Direct Virtual "
                             "Adapter' when the phone joins this laptop's Mobile hotspot.",
                    )
                    interface = labels[chosen]
                with col2:
                    host_filter = st.text_input(
                        "Only analyse traffic of this device IP (optional)",
                        placeholder="e.g. 192.168.137.25 (your phone)",
                        disabled=running, key="live_host",
                        help="Adds a capture filter so only packets to/from this "
                             "IP are analysed. Leave empty to analyse all IP traffic.",
                    )
    else:
        st.warning(
            "**SIMULATION — NOT REAL NETWORK TRAFFIC.** Demo Mode creates "
            "synthetic packets in memory (nothing is sent on any network) and "
            "feeds them through the same feature extraction and the same trained "
            "model. Suspicious bursts imitate the statistics of 5G-NIDD attack "
            "flows; the benign/malicious decision is made by the model.",
            icon="🧪",
        )
        with col1:
            scenario = st.selectbox(
                "Simulated traffic", list(DEMO_SCENARIOS),
                format_func=DEMO_SCENARIOS.get, disabled=running, key="live_scenario",
            )

    window_seconds = st.slider(
        "Monitoring window (seconds)", min_value=1, max_value=30, value=5,
        disabled=running, key="live_window",
        help="Packets are grouped into flows per window. 5 s matches the Argus "
             "status interval of the 5G-NIDD flow records the model was trained on.",
    )

    b1, b2, b3, _ = st.columns([1, 1, 1, 3])
    start = b1.button("▶ Start Monitoring", type="primary", width="stretch",
                      disabled=running or not capture_ready)
    stop = b2.button("⏹ Stop Monitoring", width="stretch", disabled=not running)
    clear = b3.button("🗑 Clear History", width="stretch", disabled=live_monitor() is None)

    if start:
        monitor = LiveMonitor(
            predictor,
            mode="capture" if mode == "Live Capture Mode" else "demo",
            window_seconds=window_seconds,
            interface=interface,
            host_filter=host_filter,
            demo_scenario=scenario,
        )
        try:
            with st.spinner("Starting…"):
                monitor.start()
            st.session_state["live_monitor"] = monitor
            st.rerun()
        except CaptureError as error:
            st.error(f"**Monitoring could not start.** {error}")

    if stop and live_monitor() is not None:
        live_monitor().stop()
        st.rerun()

    if clear and live_monitor() is not None:
        live_monitor().clear_history()

    # --------------------------------------------------------
    # LIVE PANEL (refreshes itself without rerunning the page)
    # --------------------------------------------------------

    @st.fragment(run_every=1.0 if running else None)
    def live_panel():
        monitor = live_monitor()
        if monitor is None:
            st.info("Press **Start Monitoring** to begin. Results appear after the first window.")
            return

        state = monitor.snapshot()
        if running and not state["running"]:
            st.rerun()  # stopped by an error or the watchdog: refresh controls

        if state["error"]:
            st.error(state["error"])
        if state["notice"]:
            st.info(state["notice"])

        source = (
            "SIMULATION — synthetic packets" if state["mode"] == "demo"
            else f"Interface {state['interface']}"
            + (f", device {state['host_filter']}" if state["host_filter"] else "")
        )
        st.caption(
            f"{source} · window {state['window_seconds']:.0f} s · "
            f"{state['window_count']} windows analysed · "
            f"{state['packets_seen']:,} IP packets · "
            f"{state['non_ip_ignored']:,} non-IP frames ignored · "
            f"{state['malformed']:,} malformed packets skipped"
            + (f" · {state['dropped']:,} packets over the flow limit" if state["dropped"] else "")
        )

        windows = state["windows"]
        if not windows:
            st.info(f"Collecting the first {state['window_seconds']:.0f}-second window…")
            return

        if state["mode"] == "capture" and state["empty_windows_in_row"] >= 2:
            st.warning(
                "**No packets received** in the last "
                f"{state['empty_windows_in_row']} windows. Check that the selected "
                "interface carries the test traffic, that the device IP filter is "
                "correct, and generate some traffic (e.g. open a web page on the phone)."
            )

        latest = windows[-1]

        # ----------------------------------------------------
        # CURRENT TRAFFIC
        # ----------------------------------------------------

        st.subheader("Current traffic")
        st.caption(
            f"Latest window ending {latest['Time']:%H:%M:%S}. Totals over all "
            f"{latest['Flows']} flows; the model itself classifies each flow separately."
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Packet Count", f"{latest['Packet_Count']:,}")
        m2.metric("Traffic Rate", f"{latest['Traffic_Rate']:.1f} pkt/s")
        m3.metric("Mean Packet Size", f"{latest['Mean_Packet_Size']:.1f} B")
        m4.metric("TTL (most common)", "–" if pd.isna(latest["TTL"]) else f"{latest['TTL']:.0f}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("TCP flows", latest["TCP"])
        m2.metric("UDP flows", latest["UDP"])
        m3.metric("ICMP flows", latest["ICMP"])
        m4.metric("Flows in window", latest["Flows"])

        # ----------------------------------------------------
        # IDS RESULT
        # ----------------------------------------------------

        st.subheader("IDS result")
        verdict = latest["Verdict"]
        css = {"MALICIOUS": "verdict-malicious", "BENIGN": "verdict-benign"}.get(verdict, "verdict-none")
        icon = {"MALICIOUS": "🚨", "BENIGN": "✅"}.get(verdict, "⏳")
        detail = (
            f"{latest['Flagged_Flows']} of {latest['Flows']} flows classified malicious"
            if latest["Flows"] else "No IP packets in this window"
        )
        r1, r2 = st.columns([2, 1])
        r1.markdown(
            f'<div class="verdict {css}">{icon} Prediction: {verdict}'
            f"<small>{detail}</small></div>",
            unsafe_allow_html=True,
        )
        r2.metric(
            "Malicious Probability",
            f"{latest['Max_Malicious_Probability'] * 100:.2f}%",
            help="Highest predict_proba (class 1) among the window's flows: the "
                 "10 trees' average malicious-class share at the leaves reached. "
                 "A model score, not a calibrated probability.",
        )
        if state["mode"] == "demo":
            st.caption("🧪 SIMULATION — NOT REAL NETWORK TRAFFIC")

        if state["latest_flows"]:
            with st.expander("Flows in the latest window — exact model input", expanded=False):
                st.dataframe(
                    flow_table(state["latest_flows"]), column_config=FLOW_COLUMNS,
                    width="stretch", hide_index=True, height=260,
                )

        # ----------------------------------------------------
        # LIVE GRAPHS
        # ----------------------------------------------------

        st.subheader("Live graphs")
        window_df = pd.DataFrame(windows)
        flow_df = pd.DataFrame(state["flows"])

        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**Malicious probability over time**")
            if not flow_df.empty:
                st.altair_chart(probability_chart(flow_df, window_df), width="stretch")
                st.caption("One marker per flow (▲ malicious, ● benign); grey line = "
                           "highest flow probability per window; dashed line = 0.5 decision threshold.")
            else:
                st.caption("No flows observed yet.")
        with g2:
            st.markdown("**Traffic rate over time**")
            st.altair_chart(rate_chart(window_df), width="stretch")
            st.caption("Packets per second in each window, marked by the window's verdict.")

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        st.subheader("Live traffic history")
        if flow_df.empty:
            st.caption("No flows observed yet.")
        else:
            st.caption(
                f"Most recent {len(flow_df)} flow observations (newest first, "
                "rolling history; packets themselves are not stored)."
            )
            st.dataframe(
                flow_table(state["flows"]), column_config=FLOW_COLUMNS,
                width="stretch", hide_index=True, height=360,
            )

    live_panel()

    # --------------------------------------------------------
    # EXPLANATIONS
    # --------------------------------------------------------

    st.divider()

    with st.expander("How the live pipeline works and how the 7 features are computed"):
        st.markdown(
            "**Packets → flows → 7 features → existing model → prediction.** "
            "5G-NIDD records are Argus bidirectional flow records (a status record "
            "every ~5 s), so live packets are grouped the same way: one flow "
            "(protocol + the two endpoints, both directions) in one window = one "
            "record = one prediction."
        )
        st.dataframe(
            pd.DataFrame([
                ("Rate", "Argus Rate", "(packets − 1) / (last − first packet time); 0 for a single packet"),
                ("Packet_Count", "TotPkts", "packets of the flow in the window (both directions)"),
                ("Mean_Packet_Size", "TotBytes / TotPkts", "mean frame size = IP length + 14-byte Ethernet header"),
                ("TTL", "sTtl", "IP TTL (IPv6 hop limit) of the flow originator's packets"),
                ("TCP / UDP / ICMP", "one-hot of Proto", "1 for the flow's protocol, all 0 for other IP protocols"),
            ], columns=["Feature", "5G-NIDD source (training)", "Live computation"]),
            width="stretch", hide_index=True,
        )
        info = predictor.describe()
        st.caption(
            f"Model: {info['file']} ({info['type']}, {info['trees']} trees) · "
            f"verified feature order {info['features']} · classes {info['classes']} "
            "(0 = benign, 1 = malicious). The model is loaded read-only and never retrained."
        )

    with st.expander("📱 Controlled demonstration with an Android phone"):
        st.markdown(
            "The phone simply **generates normal network traffic** inside your own "
            "test network; the laptop captures it passively.\n\n"
            "```\nAndroid phone → Wi-Fi / hotspot → Windows laptop → packet capture "
            "→ feature extraction → trained IDS model → this dashboard\n```\n"
            "1. On the laptop: **Settings → Network & internet → Mobile hotspot → On**.\n"
            "2. Connect the phone to that hotspot. Its address appears in the hotspot "
            "page (usually `192.168.137.x`).\n"
            "3. Above, choose **Live Capture Mode** and the **Microsoft Wi-Fi Direct "
            "Virtual Adapter** interface whose IP is `192.168.137.1` "
            "(if not listed yet, reload the page after turning the hotspot on).\n"
            "4. Optionally enter the phone's IP so only its traffic is analysed.\n"
            "5. Press **Start Monitoring**, then use the phone normally: browse, "
            "stream a video, or open this dashboard on the phone at "
            "`http://192.168.137.1:8501`.\n\n"
            "Only use devices and networks you own or control."
        )

    st.caption(
        "Limitation: this model was validated only on 5G-NIDD and degraded sharply on "
        "CICIoT2023 (cross-domain F1 ≈ 2%). Predictions on home Wi-Fi or phone traffic "
        "are therefore cross-domain too and should be read as a pipeline demonstration, "
        "not as a reliable security verdict."
    )


if page == "Live IDS Monitoring":
    render_live_page()


# ============================================================
# LIVE MONITOR STATUS (sidebar, every page)
# ============================================================

# While monitoring runs, a small sidebar panel keeps the background monitor
# alive and shows its status on every page. When the browser is closed it
# stops being polled and the monitor stops itself after a few minutes.
if LIVE_IMPORT_ERROR is None and monitor_running():

    @st.fragment(run_every=2.0)
    def live_sidebar_status():
        state = live_monitor().snapshot()
        if not state["running"]:
            st.caption("⏸️ Live IDS stopped")
            return
        last = state["windows"][-1]["Verdict"] if state["windows"] else "waiting…"
        icon = {"MALICIOUS": "🚨", "BENIGN": "✅"}.get(last, "⏳")
        label = "Demo (simulation)" if state["mode"] == "demo" else "Live capture"
        st.markdown(f"**🟢 Live IDS — {label}**  \n{icon} Last window: {last}")

    with st.sidebar:
        st.divider()
        live_sidebar_status()


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()
st.sidebar.caption("CS427 Mobile Communications — 5G IDS cross-domain study")
st.sidebar.caption("Train: 5G-NIDD → Test without retraining: CICIoT2023")
