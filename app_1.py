import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import streamlit as st
import joblib
import mne


@st.cache_resource
def load_my_model():
    return joblib.load('models/random_forest_model.joblib')


model = load_my_model()


def run_auto_classification(raw_psg, expert_annots, model):
    # 1. Cắt epoch 30s
    epochs = mne.make_fixed_length_epochs(raw_psg, duration=30, preload=True)

    # 2. Trích xuất features (Hàm này phải giống hệt lúc bạn train model)
    # X_test = extract_features(epochs)

    # 3. Dự đoán
    y_pred = model.predict(X_test)

    # 4. Lấy nhãn thực tế từ file Hypnogram
    # (Cần khớp thời gian giữa Annotations và Epochs)
    y_true = expert_annots.get_labels()

    return y_true, y_pred


# Giao diện Web
st.title("🤖 Auto Sleep Staging & Validator")

uploaded_psg = st.file_uploader("Upload PSG.edf", type=["edf"])
uploaded_hypno = st.file_uploader("Upload Hypnogram.edf", type=["edf"])

if uploaded_psg and uploaded_hypno and st.button("Chạy phân loại tự động"):
    # Load dữ liệu
    raw = mne.io.read_raw_edf(uploaded_psg.name)
    annots = mne.read_annotations(uploaded_hypno.name)
    raw.set_annotations(annots)

    # Chạy model (giả sử bạn đã load model Random Forest bằng joblib)
    y_true, y_pred = run_auto_classification(raw, annots, my_rf_model)

    # Hiển thị kết quả so sánh
    acc = accuracy_score(y_true, y_pred)
    st.success(f"Độ chính xác so với chuyên gia: {acc:.2%}")

    # Vẽ Confusion Matrix
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', ax=ax)
    st.pyplot(fig)
