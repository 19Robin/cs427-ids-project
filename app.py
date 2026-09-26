from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


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

BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

BASELINE_RESULTS = RESULTS_DIR / "baseline_5g_results.json"
LIGHTWEIGHT_RESULTS = RESULTS_DIR / "lightweight_model_comparison.csv"
CROSS_DOMAIN_RESULTS = RESULTS_DIR / "lightweight_cross_domain_comparison.csv"
OTHER_MODELS_RESULTS = RESULTS_DIR / "other_models_comparison.csv"
FEATURE_IMPORTANCE = RESULTS_DIR / "baseline_feature_importance.csv"
ATTACK_RESULTS = RESULTS_DIR / "attack_type_results.csv"
FEATURE_SHIFT = RESULTS_DIR / "feature_distribution_shift.csv"
PROTOCOL_SHIFT = RESULTS_DIR / "protocol_distribution_shift.csv"

BASELINE_MODEL = MODELS_DIR / "random_forest_baseline.joblib"
LIGHTWEIGHT_MODEL = MODELS_DIR / "random_forest_10_trees.joblib"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_json(path):
    if path.exists():
        with open(path, "r") as file:
            return json.load(file)
    return None


baseline = load_json(BASELINE_RESULTS)
lightweight = load_csv(LIGHTWEIGHT_RESULTS)
cross_domain = load_csv(CROSS_DOMAIN_RESULTS)
other_models = load_csv(OTHER_MODELS_RESULTS)
feature_importance = load_csv(FEATURE_IMPORTANCE)
attack_results = load_csv(ATTACK_RESULTS)
feature_shift = load_csv(FEATURE_SHIFT)
protocol_shift = load_csv(PROTOCOL_SHIFT)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, keywords, exclude_keywords=None):
    """
    Safely find the first column whose name contains all
    required keywords and none of the excluded keywords.
    """
    if exclude_keywords is None:
        exclude_keywords = []

    for column in df.columns:
        name = str(column).lower()

        if all(keyword.lower() in name for keyword in keywords):
            if not any(
                keyword.lower() in name
                for keyword in exclude_keywords
            ):
                return column

    return None


def get_unique_columns(df):
    """
    Remove duplicate column names safely.
    """
    return df.loc[
        :,
        ~df.columns.duplicated()
    ].copy()


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 0px;
        }

        .subtitle {
            font-size: 20px;
            color: #6b7280;
            margin-bottom: 20px;
        }

        .section-note {
            color: #6b7280;
            font-size: 15px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛡️ Cross-Domain Intrusion Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">5G-NIDD → CICIoT2023</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    This dashboard presents the evaluation of machine learning
    intrusion detection models trained on 5G-NIDD and tested on
    both the original 5G-NIDD network and a different CICIoT2023
    network.
    """
)

st.divider()


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
        The purpose of this experiment is to measure how
        lightweight intrusion detection models perform when
        they are trained on one network and tested on a
        different network without retraining.
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
            "Algorithms",
            "4"
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
        width="stretch",
        hide_index=True
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.header("Random Forest Model Performance")

    st.write(
        """
        This section compares Random Forest models using
        different numbers of trees while keeping the same
        seven common features and the same 5G-NIDD train/test
        split.
        """
    )

    if lightweight is not None:

        df = lightweight.copy()

        st.subheader("In-Domain Performance — 5G-NIDD")

        display_df = df.copy()

        for column in [
            "Accuracy",
            "Precision",
            "Recall",
            "F1"
        ]:
            if column in display_df.columns:
                display_df[column] *= 100

        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True
        )

        # ----------------------------------------------------
        # COMBINED LINE GRAPH
        # ----------------------------------------------------

        st.subheader("Performance by Number of Trees")

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(
            df["Trees"],
            df["Accuracy"] * 100,
            marker="o",
            linewidth=2,
            label="Accuracy"
        )

        ax.plot(
            df["Trees"],
            df["Precision"] * 100,
            marker="o",
            linewidth=2,
            label="Precision"
        )

        ax.plot(
            df["Trees"],
            df["Recall"] * 100,
            marker="o",
            linewidth=2,
            label="Recall"
        )

        ax.plot(
            df["Trees"],
            df["F1"] * 100,
            marker="o",
            linewidth=2,
            label="F1"
        )

        ax.set_xlabel("Number of Trees")
        ax.set_ylabel("Score (%)")
        ax.set_title(
            "Random Forest In-Domain Performance"
        )

        ax.set_xticks(df["Trees"])

        ax.grid(
            True,
            alpha=0.25
        )

        ax.legend()

        st.pyplot(fig)

        plt.close(fig)

        # ----------------------------------------------------
        # MODEL SIZE
        # ----------------------------------------------------

        st.subheader("Random Forest Model Size")

        fig, ax = plt.subplots(figsize=(10, 5))

        bars = ax.bar(
            df["Trees"],
            df["Model_Size_MB"]
        )

        ax.set_xlabel("Number of Trees")
        ax.set_ylabel("Model Size (MB)")
        ax.set_title(
            "Random Forest Model Size by Number of Trees"
        )

        ax.set_xticks(df["Trees"])

        ax.bar_label(
            bars,
            fmt="%.2f MB",
            padding=3
        )

        st.pyplot(fig)

        plt.close(fig)

    else:

        st.error(
            "The lightweight model comparison results "
            "could not be found."
        )


# ============================================================
# ALGORITHM COMPARISON
# ============================================================

elif page == "Algorithm Comparison":

    st.header("Machine Learning Algorithm Comparison")

    st.write(
        """
        Four machine learning algorithms were evaluated using
        the same seven common features, the same 5G-NIDD training
        and testing split, and the same CICIoT2023 cross-domain
        test data.
        """
    )

    if other_models is not None:

        df = other_models.copy()

        # ----------------------------------------------------
        # DISPLAY TABLE
        # ----------------------------------------------------

        st.subheader("Overall Comparison")

        display_df = df.copy()

        display_df["Model Size (MB)"] = (
            display_df["Model_Size_MB"].round(2)
        )

        display_df["Training Time (s)"] = (
            display_df["Training_Time_Sec"].round(2)
        )

        display_df["5G Accuracy (%)"] = (
            display_df["InDomain_Accuracy"] * 100
        ).round(2)

        display_df["5G Precision (%)"] = (
            display_df["InDomain_Precision"] * 100
        ).round(2)

        display_df["5G Recall (%)"] = (
            display_df["InDomain_Recall"] * 100
        ).round(2)

        display_df["5G F1 (%)"] = (
            display_df["InDomain_F1"] * 100
        ).round(2)

        display_df["CICIoT Accuracy (%)"] = (
            display_df["CrossDomain_Accuracy"] * 100
        ).round(2)

        display_df["CICIoT Precision (%)"] = (
            display_df["CrossDomain_Precision"] * 100
        ).round(2)

        display_df["CICIoT Recall (%)"] = (
            display_df["CrossDomain_Recall"] * 100
        ).round(2)

        display_df["CICIoT F1 (%)"] = (
            display_df["CrossDomain_F1"] * 100
        ).round(2)

        display_table = display_df[
            [
                "Model",
                "Model Size (MB)",
                "Training Time (s)",
                "5G Accuracy (%)",
                "5G Precision (%)",
                "5G Recall (%)",
                "5G F1 (%)",
                "CICIoT Accuracy (%)",
                "CICIoT Precision (%)",
                "CICIoT Recall (%)",
                "CICIoT F1 (%)"
            ]
        ]

        st.dataframe(
            display_table,
            width="stretch",
            hide_index=True
        )

        # ----------------------------------------------------
        # IN-DOMAIN PERFORMANCE
        # ----------------------------------------------------

        st.subheader("In-Domain Performance — 5G-NIDD")

        fig, ax = plt.subplots(figsize=(11, 6))

        x = range(len(df))
        width = 0.18

        ax.bar(
            [i - 1.5 * width for i in x],
            df["InDomain_Accuracy"] * 100,
            width,
            label="Accuracy"
        )

        ax.bar(
            [i - 0.5 * width for i in x],
            df["InDomain_Precision"] * 100,
            width,
            label="Precision"
        )

        ax.bar(
            [i + 0.5 * width for i in x],
            df["InDomain_Recall"] * 100,
            width,
            label="Recall"
        )

        ax.bar(
            [i + 1.5 * width for i in x],
            df["InDomain_F1"] * 100,
            width,
            label="F1"
        )

        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Score (%)")
        ax.set_title(
            "In-Domain Performance Comparison"
        )

        ax.set_xticks(list(x))
        ax.set_xticklabels(df["Model"])

        ax.grid(
            axis="y",
            alpha=0.25
        )

        ax.legend()

        st.pyplot(fig)

        plt.close(fig)

        # ----------------------------------------------------
        # CROSS-DOMAIN PERFORMANCE
        # ----------------------------------------------------

        st.subheader("Cross-Domain Performance — CICIoT2023")

        fig, ax = plt.subplots(figsize=(11, 6))

        ax.bar(
            [i - 1.5 * width for i in x],
            df["CrossDomain_Accuracy"] * 100,
            width,
            label="Accuracy"
        )

        ax.bar(
            [i - 0.5 * width for i in x],
            df["CrossDomain_Precision"] * 100,
            width,
            label="Precision"
        )

        ax.bar(
            [i + 0.5 * width for i in x],
            df["CrossDomain_Recall"] * 100,
            width,
            label="Recall"
        )

        ax.bar(
            [i + 1.5 * width for i in x],
            df["CrossDomain_F1"] * 100,
            width,
            label="F1"
        )

        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Score (%)")
        ax.set_title(
            "Cross-Domain Performance Comparison"
        )

        ax.set_xticks(list(x))
        ax.set_xticklabels(df["Model"])

        ax.grid(
            axis="y",
            alpha=0.25
        )

        ax.legend()

        st.pyplot(fig)

        plt.close(fig)

        # ----------------------------------------------------
        # IN-DOMAIN VS CROSS-DOMAIN F1
        # ----------------------------------------------------

        st.subheader("In-Domain vs Cross-Domain F1")

        fig, ax = plt.subplots(figsize=(11, 6))

        ax.bar(
            [i - width / 2 for i in x],
            df["InDomain_F1"] * 100,
            width,
            label="5G-NIDD"
        )

        ax.bar(
            [i + width / 2 for i in x],
            df["CrossDomain_F1"] * 100,
            width,
            label="CICIoT2023"
        )

        ax.set_xlabel("Algorithm")
        ax.set_ylabel("F1 Score (%)")
        ax.set_title(
            "F1 Score: In-Domain vs Cross-Domain"
        )

        ax.set_xticks(list(x))
        ax.set_xticklabels(df["Model"])

        ax.grid(
            axis="y",
            alpha=0.25
        )

        ax.legend()

        st.pyplot(fig)

        plt.close(fig)

        # ----------------------------------------------------
        # F1 DROP
        # ----------------------------------------------------

        df["F1_Drop"] = (
            df["InDomain_F1"] -
            df["CrossDomain_F1"]
        )

        st.subheader("F1 Score Change")

        fig, ax = plt.subplots(figsize=(11, 5))

        bars = ax.bar(
            df["Model"],
            df["F1_Drop"] * 100
        )

        ax.set_xlabel("Algorithm")
        ax.set_ylabel(
            "F1 Score Drop (percentage points)"
        )

        ax.set_title(
            "In-Domain to Cross-Domain F1 Score Change"
        )

        ax.grid(
            axis="y",
            alpha=0.25
        )

        ax.bar_label(
            bars,
            fmt="%.2f",
            padding=3
        )

        st.pyplot(fig)

        plt.close(fig)

        # ----------------------------------------------------
        # MODEL SIZE AND TRAINING TIME
        # ----------------------------------------------------

        st.subheader("Model Size and Training Time")

        fig, ax = plt.subplots(figsize=(11, 6))

        ax.bar(
            [i - width / 2 for i in x],
            df["Model_Size_MB"],
            width,
            label="Model Size (MB)"
        )

        ax.bar(
            [i + width / 2 for i in x],
            df["Training_Time_Sec"],
            width,
            label="Training Time (seconds)"
        )

        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Value")
        ax.set_title(
            "Model Size and Training Time Comparison"
        )

        ax.set_xticks(list(x))
        ax.set_xticklabels(df["Model"])

        ax.grid(
            axis="y",
            alpha=0.25
        )

        ax.legend()

        st.pyplot(fig)

        plt.close(fig)

        # ----------------------------------------------------
        # KEY OBSERVATIONS
        # ----------------------------------------------------

        st.subheader("Key Observations")

        st.markdown(
            """
            - All four algorithms achieved substantially higher
              performance on 5G-NIDD than on CICIoT2023.
            - Cross-domain F1 scores were very low for all models.
            - Changing the machine learning algorithm alone did
              not resolve the cross-domain generalisation problem.
            - The algorithms differed in model size, training time,
              and prediction time.
            """
        )

    else:

        st.error(
            "The other model comparison results could not be found."
        )


# ============================================================
# CROSS-DOMAIN EVALUATION
# ============================================================

elif page == "Cross-Domain Evaluation":

    st.header("Cross-Domain Evaluation")

    st.write(
        """
        Models were trained using 5G-NIDD and then tested
        directly on CICIoT2023. No retraining or parameter
        changes were made.
        """
    )

    if cross_domain is not None:

        df = cross_domain.copy()

        baseline_rows = df[df["Trees"] == 100]

        if not baseline_rows.empty:

            baseline_row = baseline_rows.iloc[0]

            col1, col2, col3, col4 = st.columns(4)

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

        st.subheader("Random Forest Cross-Domain Results")

        display_df = df.copy()

        for column in [
            "Accuracy",
            "Precision",
            "Recall",
            "F1"
        ]:
            if column in display_df.columns:
                display_df[column] *= 100

        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True
        )

        # ----------------------------------------------------
        # COMBINED LINE GRAPH
        # ----------------------------------------------------

        st.subheader("Cross-Domain Performance by Number of Trees")

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(
            df["Trees"],
            df["Accuracy"] * 100,
            marker="o",
            linewidth=2,
            label="Accuracy"
        )

        ax.plot(
            df["Trees"],
            df["Precision"] * 100,
            marker="o",
            linewidth=2,
            label="Precision"
        )

        ax.plot(
            df["Trees"],
            df["Recall"] * 100,
            marker="o",
            linewidth=2,
            label="Recall"
        )

        ax.plot(
            df["Trees"],
            df["F1"] * 100,
            marker="o",
            linewidth=2,
            label="F1"
        )

        ax.set_xlabel("Number of Trees")
        ax.set_ylabel("Score (%)")
        ax.set_title(
            "Random Forest Cross-Domain Performance"
        )

        ax.set_xticks(df["Trees"])

        ax.grid(
            True,
            alpha=0.25
        )

        ax.legend()

        st.pyplot(fig)

        plt.close(fig)

        # ----------------------------------------------------
        # CROSS-DOMAIN F1 VS MODEL SIZE
        # ----------------------------------------------------

        st.subheader("F1 Score and Model Size")

        fig, ax1 = plt.subplots(figsize=(10, 5))

        ax1.plot(
            df["Trees"],
            df["F1"] * 100,
            marker="o",
            linewidth=2,
            label="F1 Score"
        )

        ax1.set_xlabel("Number of Trees")
        ax1.set_ylabel("F1 Score (%)")
        ax1.set_xticks(df["Trees"])

        ax1.grid(
            True,
            alpha=0.25
        )

        ax2 = ax1.twinx()

        ax2.plot(
            df["Trees"],
            df["Model_Size_MB"],
            marker="s",
            linewidth=2,
            linestyle="--",
            label="Model Size"
        )

        ax2.set_ylabel("Model Size (MB)")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        ax1.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="center right"
        )

        ax1.set_title(
            "Cross-Domain F1 Score vs Model Size"
        )

        st.pyplot(fig)

        plt.close(fig)

        st.warning(
            "The models show a severe drop in intrusion "
            "detection performance when transferred from "
            "5G-NIDD to CICIoT2023."
        )

    else:

        st.error(
            "The cross-domain results file could not be found."
        )


# ============================================================
# ATTACK ANALYSIS
# ============================================================

elif page == "Attack Analysis":

    st.header("Attack-Type Analysis")

    if attack_results is not None:

        st.write(
            """
            Detection rate represents the percentage of samples
            from each CICIoT2023 attack category that were
            classified as malicious.
            """
        )

        st.dataframe(
            attack_results,
            width="stretch",
            hide_index=True
        )

        attack_column = None
        rate_column = None

        for col in attack_results.columns:

            lower = str(col).lower()

            if attack_column is None and (
                "attack" in lower
                or "label" in lower
                or "type" in lower
            ):
                attack_column = col

            if rate_column is None and (
                "rate" in lower
                or "detection" in lower
            ):
                rate_column = col

        if attack_column and rate_column:

            plot_df = attack_results.copy()

            plot_df = plot_df.sort_values(
                rate_column,
                ascending=True
            )

            fig, ax = plt.subplots(
                figsize=(11, 13)
            )

            bars = ax.barh(
                plot_df[attack_column],
                plot_df[rate_column] * 100
            )

            ax.set_xlabel(
                "Detection Rate (%)"
            )

            ax.set_ylabel(
                "Attack Type"
            )

            ax.set_title(
                "CICIoT2023 Attack Detection Rate"
            )

            ax.grid(
                axis="x",
                alpha=0.25
            )

            ax.bar_label(
                bars,
                fmt="%.2f%%",
                padding=3
            )

            st.pyplot(fig)

            plt.close(fig)

    else:

        st.info(
            "The attack-type results file could not be found."
        )


# ============================================================
# FEATURE SHIFT
# ============================================================

elif page == "Feature Shift":

    st.header("Feature Distribution Shift")

    st.write(
        """
        The same seven common features were compared between
        5G-NIDD and CICIoT2023 to identify differences in their
        distributions.
        """
    )

    # --------------------------------------------------------
    # FEATURE SHIFT TABLE
    # --------------------------------------------------------

    if feature_shift is not None:

        feature_df = get_unique_columns(
            feature_shift.copy()
        )

        st.subheader("Feature Distribution Results")

        st.dataframe(
            feature_df,
            width="stretch",
            hide_index=True
        )

        # ----------------------------------------------------
        # FIND FEATURE NAME COLUMN
        # ----------------------------------------------------

        feature_column = None

        for col in feature_df.columns:

            lower = str(col).lower()

            if lower in [
                "feature",
                "feature_name",
                "name"
            ]:
                feature_column = col
                break

        if feature_column is None:

            for col in feature_df.columns:

                lower = str(col).lower()

                if "feature" in lower:
                    feature_column = col
                    break

        # ----------------------------------------------------
        # FIND 5G MEAN COLUMN
        # ----------------------------------------------------

        five_g_column = None

        for col in feature_df.columns:

            lower = str(col).lower()

            if (
                "5g" in lower
                and "mean" in lower
            ):
                five_g_column = col
                break

        # ----------------------------------------------------
        # FIND CIC MEAN COLUMN
        # ----------------------------------------------------

        cic_column = None

        for col in feature_df.columns:

            lower = str(col).lower()

            if (
                "cic" in lower
                and "mean" in lower
            ):
                cic_column = col
                break

        # ----------------------------------------------------
        # CREATE GRAPH ONLY WHEN COLUMNS EXIST
        # ----------------------------------------------------

        if (
            feature_column is not None
            and five_g_column is not None
            and cic_column is not None
        ):

            st.subheader(
                "Mean Feature Values Across Networks"
            )

            chart_df = pd.DataFrame({
                "Feature": feature_df[
                    feature_column
                ].astype(str).values,

                "5G-NIDD": pd.to_numeric(
                    feature_df[five_g_column],
                    errors="coerce"
                ).values,

                "CICIoT2023": pd.to_numeric(
                    feature_df[cic_column],
                    errors="coerce"
                ).values
            })

            chart_df = chart_df.dropna()

            if not chart_df.empty:

                fig, ax = plt.subplots(
                    figsize=(12, 7)
                )

                x = list(range(len(chart_df)))
                width = 0.35

                ax.bar(
                    [
                        i - width / 2
                        for i in x
                    ],
                    chart_df["5G-NIDD"].tolist(),
                    width,
                    label="5G-NIDD"
                )

                ax.bar(
                    [
                        i + width / 2
                        for i in x
                    ],
                    chart_df["CICIoT2023"].tolist(),
                    width,
                    label="CICIoT2023"
                )

                ax.set_xlabel("Feature")
                ax.set_ylabel("Mean Value")

                ax.set_title(
                    "Feature Distribution Comparison"
                )

                ax.set_xticks(x)

                ax.set_xticklabels(
                    chart_df["Feature"].tolist(),
                    rotation=30,
                    ha="right"
                )

                ax.grid(
                    axis="y",
                    alpha=0.25
                )

                ax.legend()

                st.pyplot(fig)

                plt.close(fig)

        else:

            st.info(
                "The feature distribution file was loaded, "
                "but the expected mean-value columns could "
                "not be identified for the graph."
            )

    else:

        st.error(
            "The feature distribution shift file could not "
            "be found."
        )

    # ========================================================
    # PROTOCOL SHIFT
    # ========================================================

    if protocol_shift is not None:

        protocol_df = get_unique_columns(
            protocol_shift.copy()
        )

        st.subheader("Protocol Distribution Shift")

        st.dataframe(
            protocol_df,
            width="stretch",
            hide_index=True
        )

        # ----------------------------------------------------
        # FIND PROTOCOL COLUMN
        # ----------------------------------------------------

        protocol_column = None

        for col in protocol_df.columns:

            lower = str(col).lower()

            if lower in [
                "protocol",
                "protocol_name",
                "feature",
                "name"
            ]:
                protocol_column = col
                break

        if protocol_column is None:

            for col in protocol_df.columns:

                lower = str(col).lower()

                if "protocol" in lower:
                    protocol_column = col
                    break

        # ----------------------------------------------------
        # FIND 5G COLUMN
        # ----------------------------------------------------

        five_g_protocol = None

        for col in protocol_df.columns:

            lower = str(col).lower()

            if "5g" in lower:
                five_g_protocol = col
                break

        # ----------------------------------------------------
        # FIND CIC COLUMN
        # ----------------------------------------------------

        cic_protocol = None

        for col in protocol_df.columns:

            lower = str(col).lower()

            if "cic" in lower:
                cic_protocol = col
                break

        # ----------------------------------------------------
        # CREATE PROTOCOL GRAPH
        # ----------------------------------------------------

        if (
            protocol_column is not None
            and five_g_protocol is not None
            and cic_protocol is not None
        ):

            st.subheader(
                "Protocol Distribution Comparison"
            )

            chart_df = pd.DataFrame({
                "Protocol": protocol_df[
                    protocol_column
                ].astype(str).values,

                "5G-NIDD": pd.to_numeric(
                    protocol_df[five_g_protocol],
                    errors="coerce"
                ).values,

                "CICIoT2023": pd.to_numeric(
                    protocol_df[cic_protocol],
                    errors="coerce"
                ).values
            })

            chart_df = chart_df.dropna()

            if not chart_df.empty:

                fig, ax = plt.subplots(
                    figsize=(10, 6)
                )

                x = list(range(len(chart_df)))
                width = 0.35

                ax.bar(
                    [
                        i - width / 2
                        for i in x
                    ],
                    chart_df["5G-NIDD"].tolist(),
                    width,
                    label="5G-NIDD"
                )

                ax.bar(
                    [
                        i + width / 2
                        for i in x
                    ],
                    chart_df["CICIoT2023"].tolist(),
                    width,
                    label="CICIoT2023"
                )

                ax.set_xlabel("Protocol")
                ax.set_ylabel(
                    "Mean / Proportion (%)"
                )

                ax.set_title(
                    "Protocol Distribution Comparison"
                )

                ax.set_xticks(x)

                ax.set_xticklabels(
                    chart_df["Protocol"].tolist()
                )

                ax.grid(
                    axis="y",
                    alpha=0.25
                )

                ax.legend()

                st.pyplot(fig)

                plt.close(fig)

        else:

            st.info(
                "The protocol distribution file was loaded, "
                "but the expected columns could not be "
                "identified for the graph."
            )

    st.subheader("Main Feature Differences")

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

    if not LIGHTWEIGHT_MODEL.exists():

        st.warning(
            """
            The 10-tree Random Forest model is not available
            in this Streamlit deployment.

            The trained model files are intentionally excluded
            from GitHub because of their file size. The
            evaluation results and charts remain available
            throughout the dashboard.
            """
        )

        st.info(
            "The Prediction Demo works when running the project "
            "locally because the model file exists in the local "
            "models/ folder."
        )

    else:

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