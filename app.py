import os
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Cross-Domain IDS Evaluation",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\cs427-ids-project"

RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")


BASELINE_RESULTS = os.path.join(
    RESULTS_DIR,
    "baseline_5g_results.json"
)

LIGHTWEIGHT_RESULTS = os.path.join(
    RESULTS_DIR,
    "lightweight_model_comparison.csv"
)

CROSS_DOMAIN_RESULTS = os.path.join(
    RESULTS_DIR,
    "lightweight_cross_domain_comparison.csv"
)

FEATURE_IMPORTANCE = os.path.join(
    RESULTS_DIR,
    "baseline_feature_importance.csv"
)

ATTACK_RESULTS = os.path.join(
    RESULTS_DIR,
    "attack_type_results.csv"
)

FEATURE_SHIFT = os.path.join(
    RESULTS_DIR,
    "feature_shift_results.csv"
)

BASELINE_MODEL = os.path.join(
    MODELS_DIR,
    "random_forest_baseline.joblib"
)

LIGHTWEIGHT_MODEL = os.path.join(
    MODELS_DIR,
    "random_forest_10_trees.joblib"
)


# ============================================================
# HEADER
# ============================================================

st.title("🛡️ Cross-Domain Intrusion Detection System")

st.markdown(
    """
    ### 5G-NIDD → CICIoT2023

    This dashboard presents the evaluation of a lightweight
    Random Forest intrusion detection model trained on 5G-NIDD
    and tested on both the original 5G-NIDD network and a
    different CICIoT2023 network.
    """
)

st.divider()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as file:
            return json.load(file)
    return None


baseline = load_json(BASELINE_RESULTS)
lightweight = load_csv(LIGHTWEIGHT_RESULTS)
cross_domain = load_csv(CROSS_DOMAIN_RESULTS)
feature_importance = load_csv(FEATURE_IMPORTANCE)
attack_results = load_csv(ATTACK_RESULTS)
feature_shift = load_csv(FEATURE_SHIFT)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Model Performance",
        "Cross-Domain Evaluation",
        "Attack Analysis",
        "Feature Shift",
        "Prediction Demo"
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.header("Project Overview")

    st.write(
        """
        The purpose of this experiment is to measure how a lightweight
        intrusion detection model performs when it is trained on one
        network and tested on a different network without retraining.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "5G-NIDD Records",
            "1,215,676"
        )

    with col2:
        st.metric(
            "CICIoT2023 Records",
            "3,579,528"
        )

    with col3:
        st.metric(
            "Common Features",
            "7"
        )

    with col4:
        st.metric(
            "RF Baseline",
            "100 Trees"
        )

    st.subheader("Common Features")

    features = pd.DataFrame({
        "Feature": [
            "Rate",
            "Packet_Count",
            "Mean_Packet_Size",
            "TTL",
            "TCP",
            "UDP",
            "ICMP"
        ],
        "5G-NIDD Source": [
            "Rate",
            "TotPkts",
            "TotBytes / TotPkts",
            "sTtl",
            "Derived from Proto",
            "Derived from Proto",
            "Derived from Proto"
        ],
        "CICIoT2023 Source": [
            "Rate",
            "Number",
            "AVG",
            "Time_To_Live",
            "TCP",
            "UDP",
            "ICMP"
        ]
    })

    st.dataframe(
        features,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.header("Random Forest Model Performance")

    if lightweight is not None:

        display_df = lightweight.copy()

        display_df["Accuracy"] *= 100
        display_df["Precision"] *= 100
        display_df["Recall"] *= 100
        display_df["F1"] *= 100

        st.subheader("In-Domain Performance — 5G-NIDD")

        st.dataframe(
            display_df.style.format({
                "Accuracy": "{:.2f}%",
                "Precision": "{:.2f}%",
                "Recall": "{:.2f}%",
                "F1": "{:.2f}%",
                "Training_Time_Seconds": "{:.2f}",
                "Prediction_Time_Seconds": "{:.2f}",
                "Model_Size_MB": "{:.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.subheader("F1 Score by Number of Trees")

        fig, ax = plt.subplots()

        ax.plot(
            display_df["Trees"],
            display_df["F1"],
            marker="o"
        )

        ax.set_xlabel("Number of Trees")
        ax.set_ylabel("F1 Score (%)")
        ax.set_title("In-Domain F1 Score")

        st.pyplot(fig)

        st.subheader("Model Size")

        fig, ax = plt.subplots()

        ax.bar(
            display_df["Trees"],
            display_df["Model_Size_MB"]
        )

        ax.set_xlabel("Number of Trees")
        ax.set_ylabel("Model Size (MB)")
        ax.set_title("Random Forest Model Size")

        st.pyplot(fig)


# ============================================================
# CROSS-DOMAIN
# ============================================================

elif page == "Cross-Domain Evaluation":

    st.header("Cross-Domain Evaluation")

    st.write(
        """
        Models were trained using 5G-NIDD and then tested directly
        on CICIoT2023. No retraining or parameter changes were made.
        """
    )

    if cross_domain is not None:

        df = cross_domain.copy()

        col1, col2, col3, col4 = st.columns(4)

        baseline_row = df[df["Trees"] == 100].iloc[0]

        with col1:
            st.metric(
                "Accuracy",
                f"{baseline_row['Accuracy'] * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Precision",
                f"{baseline_row['Precision'] * 100:.2f}%"
            )

        with col3:
            st.metric(
                "Malicious Recall",
                f"{baseline_row['Recall'] * 100:.2f}%"
            )

        with col4:
            st.metric(
                "F1 Score",
                f"{baseline_row['F1'] * 100:.2f}%"
            )

        st.subheader("All Models on CICIoT2023")

        display_df = df.copy()

        display_df["Accuracy"] *= 100
        display_df["Precision"] *= 100
        display_df["Recall"] *= 100
        display_df["F1"] *= 100

        st.dataframe(
            display_df.style.format({
                "Accuracy": "{:.2f}%",
                "Precision": "{:.2f}%",
                "Recall": "{:.2f}%",
                "F1": "{:.2f}%",
                "Prediction_Time_Seconds": "{:.2f}",
                "Model_Size_MB": "{:.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Cross-Domain F1 Score")

        fig, ax = plt.subplots()

        ax.plot(
            df["Trees"],
            df["F1"] * 100,
            marker="o"
        )

        ax.set_xlabel("Number of Trees")
        ax.set_ylabel("F1 Score (%)")
        ax.set_title("Cross-Domain F1 Score on CICIoT2023")

        st.pyplot(fig)

        st.warning(
            "The models show a severe drop in intrusion detection "
            "performance when transferred from 5G-NIDD to CICIoT2023."
        )


# ============================================================
# ATTACK ANALYSIS
# ============================================================

elif page == "Attack Analysis":

    st.header("Attack-Type Analysis")

    if attack_results is not None:

        st.write(
            """
            Detection rate represents the percentage of samples from
            each CICIoT2023 attack category that were classified as
            malicious.
            """
        )

        st.dataframe(
            attack_results,
            use_container_width=True,
            hide_index=True
        )

        # Try to identify useful columns automatically
        attack_column = None
        rate_column = None

        for col in attack_results.columns:

            lower = col.lower()

            if "attack" in lower or "label" in lower:
                attack_column = col

            if "rate" in lower or "detection" in lower:
                rate_column = col

        if attack_column and rate_column:

            plot_df = attack_results.copy()

            plot_df = plot_df.sort_values(
                rate_column,
                ascending=True
            )

            fig, ax = plt.subplots(figsize=(10, 8))

            ax.barh(
                plot_df[attack_column],
                plot_df[rate_column] * 100
            )

            ax.set_xlabel("Detection Rate (%)")
            ax.set_ylabel("Attack Type")
            ax.set_title(
                "CICIoT2023 Attack Detection Rate"
            )

            st.pyplot(fig)

    else:

        st.info(
            "Attack analysis results file was not found."
        )


# ============================================================
# FEATURE SHIFT
# ============================================================

elif page == "Feature Shift":

    st.header("Feature Distribution Shift")

    st.write(
        """
        The same seven features were compared between 5G-NIDD and
        CICIoT2023 to identify differences in their distributions.
        """
    )

    if feature_shift is not None:

        st.dataframe(
            feature_shift,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Feature shift results file was not found."
        )

    st.subheader("Important Feature Differences")

    st.markdown(
        """
        **Large differences were observed in:**

        - Rate
        - Packet Count
        - TCP
        - UDP
        - ICMP

        **A moderate difference was observed in:**

        - Mean Packet Size

        **TTL showed a comparatively smaller distribution difference.**
        """
    )


# ============================================================
# PREDICTION DEMO
# ============================================================

elif page == "Prediction Demo":

    st.header("Intrusion Detection Prediction")

    st.write(
        """
        Enter the seven common features below. The lightweight
        10-tree Random Forest will classify the traffic as
        benign or malicious.
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        rate = st.number_input(
            "Rate",
            min_value=0.0,
            value=100.0
        )

        packet_count = st.number_input(
            "Packet Count",
            min_value=1.0,
            value=10.0
        )

        mean_packet_size = st.number_input(
            "Mean Packet Size",
            min_value=0.0,
            value=100.0
        )

        ttl = st.number_input(
            "TTL",
            min_value=0.0,
            max_value=255.0,
            value=64.0
        )

    with col2:

        tcp = st.number_input(
            "TCP",
            min_value=0.0,
            max_value=1.0,
            value=0.0
        )

        udp = st.number_input(
            "UDP",
            min_value=0.0,
            max_value=1.0,
            value=1.0
        )

        icmp = st.number_input(
            "ICMP",
            min_value=0.0,
            max_value=1.0,
            value=0.0
        )

    if st.button(
        "🔍 Analyze Traffic",
        type="primary"
    ):

        if os.path.exists(LIGHTWEIGHT_MODEL):

            model = joblib.load(
                LIGHTWEIGHT_MODEL
            )

            input_data = pd.DataFrame(
                [[
                    rate,
                    packet_count,
                    mean_packet_size,
                    ttl,
                    tcp,
                    udp,
                    icmp
                ]],
                columns=[
                    "Rate",
                    "Packet_Count",
                    "Mean_Packet_Size",
                    "TTL",
                    "TCP",
                    "UDP",
                    "ICMP"
                ]
            )

            prediction = model.predict(
                input_data
            )[0]

            probability = model.predict_proba(
                input_data
            )[0]

            malicious_probability = probability[1]

            if prediction == 1:

                st.error(
                    "🚨 MALICIOUS TRAFFIC DETECTED"
                )

            else:

                st.success(
                    "✅ TRAFFIC CLASSIFIED AS BENIGN"
                )

            st.metric(
                "Malicious Probability",
                f"{malicious_probability * 100:.2f}%"
            )

        else:

            st.error(
                "The 10-tree Random Forest model was not found."
            )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "CS427 Intrusion Detection System"
)

st.sidebar.caption(
    "5G-NIDD → CICIoT2023"
)