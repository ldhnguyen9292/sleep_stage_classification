import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import mne
import os
import tempfile

from utils.edf_handler import feature_extraction, filter_wake_epochs


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_my_model():
    model_path = os.path.join(
        'models',
        'random_forest_model.joblib'
    )
    return joblib.load(model_path)


model = load_my_model()


# =========================
# AUTO CLASSIFICATION
# =========================
def run_auto_classification(raw_psg, model):

    epochs = mne.make_fixed_length_epochs(
        raw_psg,
        duration=30,
        preload=True
    )

    epochs_filtered = filter_wake_epochs(epochs)

    df = feature_extraction(epochs_filtered)

    X_test = df.drop(
        columns=['label', 'index', 'start_time']
    )

    y_true = df['label']
    y_pred = model.predict(X_test)

    return epochs, y_true, y_pred


# =========================
# PLOT SINGLE EPOCH
# =========================
def plot_epoch_signal(epoch_data, true_label, pred_label, epoch_idx):

    data = epoch_data.get_data()[0]

    sfreq = epoch_data.info['sfreq']

    times = np.arange(data.shape[1]) / sfreq

    n_channels = data.shape[0]

    fig, axes = plt.subplots(
        n_channels,
        1,
        figsize=(14, 8),
        sharex=True
    )

    if n_channels == 1:
        axes = [axes]

    for i in range(n_channels):

        axes[i].plot(times, data[i])

        axes[i].set_ylabel(
            epoch_data.ch_names[i],
            rotation=0,
            labelpad=40
        )

        axes[i].grid(True)

    axes[-1].set_xlabel("Time (s)")

    correct = true_label == pred_label

    title_color = "green" if correct else "red"

    fig.suptitle(
        f"Epoch {epoch_idx} | "
        f"True: {true_label} | "
        f"Pred: {pred_label}",
        fontsize=16,
        color=title_color
    )

    plt.tight_layout()

    return fig


# =========================
# HYPNOGRAM PLOT
# =========================
def plot_hypnogram(y_true, y_pred):

    stage_mapping = {
        "Sleep stage W": 0,
        "Sleep stage 1": 1,
        "Sleep stage 2": 2,
        "Sleep stage 3": 3,
        "Sleep stage 4": 3,
        "Sleep stage R": 4
    }

    y_true_num = [stage_mapping.get(y, -1) for y in y_true]
    y_pred_num = [stage_mapping.get(y, -1) for y in y_pred]

    fig, ax = plt.subplots(figsize=(15, 4))

    ax.plot(
        y_true_num,
        label='Ground Truth',
        linewidth=2
    )

    ax.plot(
        y_pred_num,
        label='Prediction',
        linestyle='--'
    )

    ax.set_yticks([0, 1, 2, 3, 4])

    ax.set_yticklabels([
        'W',
        'N1',
        'N2',
        'N3',
        'REM'
    ])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Sleep Stage")

    ax.set_title("Sleep Hypnogram")

    ax.legend()

    ax.grid(True)

    return fig


# =========================
# STREAMLIT UI
# =========================
# st.set_page_config(
#     page_title="Auto Sleep Staging",
#     layout="wide"
# )

st.title("🤖 Auto Sleep Staging & Validator")

uploaded_psg = st.file_uploader(
    "Upload PSG.edf",
    type=["edf"]
)

uploaded_hypno = st.file_uploader(
    "Upload Hypnogram.edf",
    type=["edf"]
)

try:

    if (
        uploaded_psg
        and uploaded_hypno
        and st.button("🚀 Chạy phân loại tự động")
    ):

        # =========================
        # SAVE TEMP FILES
        # =========================
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".edf"
        ) as tmp_file:

            tmp_file.write(uploaded_psg.getvalue())
            tmp_path = tmp_file.name

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".edf"
        ) as tmp_hypno_file:

            tmp_hypno_file.write(
                uploaded_hypno.getvalue()
            )

            tmp_hypno_path = tmp_hypno_file.name

        # =========================
        # LOAD DATA
        # =========================
        raw = mne.io.read_raw_edf(
            tmp_path,
            preload=True
        )

        annots = mne.read_annotations(
            tmp_hypno_path
        )

        raw.set_annotations(annots)

        # =========================
        # BASIC INFO
        # =========================
        st.subheader("📊 PSG Information")

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Channels:** {len(raw.ch_names)}")
            st.write(
                f"**Sampling Rate:** "
                f"{raw.info['sfreq']} Hz"
            )

        with col2:
            st.write(
                f"**Duration:** "
                f"{raw.n_times / raw.info['sfreq']:.2f} s"
            )

            st.write(
                f"**Start Time:** "
                f"{raw.info['meas_date']}"
            )

        # =========================
        # RUN MODEL
        # =========================
        with st.spinner("Đang phân loại..."):

            epochs, y_true, y_pred = (
                run_auto_classification(raw, model)
            )

        # =========================
        # METRICS
        # =========================
        st.subheader("📈 Overall Performance")

        acc = accuracy_score(y_true, y_pred)

        st.success(
            f"Accuracy: {acc:.2%}"
        )

        # =========================
        # CONFUSION MATRIX
        # =========================
        st.subheader("🧠 Confusion Matrix")

        fig_cm, ax = plt.subplots(figsize=(6, 5))

        sns.heatmap(
            confusion_matrix(y_true, y_pred),
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=ax
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        st.pyplot(fig_cm)

        # =========================
        # CLASSIFICATION REPORT
        # =========================
        st.subheader("📄 Classification Report")

        report = classification_report(
            y_true,
            y_pred,
            output_dict=True
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(report_df)

        # =========================
        # HYPNOGRAM
        # =========================
        st.subheader("🌙 Sleep Hypnogram")

        fig_hyp = plot_hypnogram(
            y_true,
            y_pred
        )

        st.pyplot(fig_hyp)

        # =========================
        # EPOCH VIEWER
        # =========================
        st.subheader("🔍 Epoch Viewer")

        epoch_idx = st.slider(
            "Select Epoch",
            min_value=0,
            max_value=len(epochs) - 1,
            value=0
        )

        epoch_data = epochs[epoch_idx]

        true_label = y_true.iloc[epoch_idx]
        pred_label = y_pred[epoch_idx]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Ground Truth",
                true_label
            )

        with col2:
            st.metric(
                "Prediction",
                pred_label
            )

        with col3:

            if true_label == pred_label:
                st.success("✅ Correct")
            else:
                st.error("❌ Wrong")

        # =========================
        # PLOT EPOCH SIGNAL
        # =========================
        fig_epoch = plot_epoch_signal(
            epoch_data,
            true_label,
            pred_label,
            epoch_idx
        )

        st.pyplot(fig_epoch)

        # =========================
        # REMOVE TEMP FILES
        # =========================
        os.remove(tmp_path)
        os.remove(tmp_hypno_path)

except Exception as e:

    st.error(f"Đã xảy ra lỗi: {str(e)}")

    st.stop()

# =========================
# FOOTER
# =========================
st.divider()

st.info(
    "📌 Lưu ý: "
    "Kết quả phân loại tự động "
    "chỉ mang tính tham khảo."
)
