import streamlit as st
import pandas as pd
import os
from datetime import timedelta

import utils.data_utils as du
import utils.plot_utils as pu

st.set_page_config(layout="wide", page_title="Sleep Feature Validator Pro")

# --- SIDEBAR ---
st.sidebar.title("🛠️ Validator Pro")
csv_files = du.get_file_lists()
selected_csv = st.sidebar.selectbox("1. Chọn file Features:", csv_files)

# Logic tìm file
psg_f, hypno_f, p_id = du.find_edf_files(selected_csv)

if not psg_f:
    st.error(f"Không tìm thấy file PSG cho ID: {p_id}")
    st.stop()

# --- LOAD DATA ---
try:
    raw = du.load_mne_raw(psg_f, hypno_f)
    df_features = pd.read_csv(os.path.join(du.CSV_DIR, selected_csv))

    # Header Info
    start_t = raw.info['meas_date']
    duration_h = (raw.n_times / raw.info['sfreq']) / 3600
    st.markdown(f"### 📂 Đối chiếu: `{psg_f}` ({duration_h:.2f} giờ)")

    # Sidebar settings
    epoch_num = st.sidebar.slider("2. Chọn Epoch:", 1, len(df_features), 1)
    selected_channels = st.sidebar.multiselect("3. Kênh:", raw.ch_names,
                                               default=raw.ch_names[:3] if len(raw.ch_names) > 3 else raw.ch_names)

    # --- HIỂN THỊ CHI TIẾT ---
    col_feat, col_plot = st.columns([1, 3])
    curr_row = df_features.iloc[epoch_num - 1]

    with col_feat:
        st.subheader("📊 Features")
        st.metric("Label", f"Stage {int(curr_row.get('label', 0))}")
        st.dataframe(curr_row.to_frame(), height=400)

    with col_plot:
        fig_signals = pu.plot_signals(
            raw, selected_channels, (epoch_num-1)*30, raw.info['sfreq'])
        st.plotly_chart(fig_signals, use_container_width=True)

    # --- TIMELINE ---
    st.divider()
    st.subheader("🎞️ Real-time Sleep Stages Timeline")
    df_ann = du.get_hypnogram_data(raw)
    if not df_ann.empty:
        fig_timeline = pu.plot_hypnogram_timeline(df_ann)
        st.plotly_chart(fig_timeline, use_container_width=True)
    else:
        st.warning("Không tìm thấy dữ liệu Hypnogram.")

except Exception as e:
    st.error(f"Lỗi hệ thống: {e}")
