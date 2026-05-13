import mne
import numpy as np
import pandas as pd
from datetime import timedelta

valid_stages = ['Sleep stage W', 'Sleep stage 1', 'Sleep stage 2',
                'Sleep stage 3', 'Sleep stage 4', 'Sleep stage R']


def generate_epochs_from_edf(psg_file, hyp_file, duration=30):
    raw = mne.io.read_raw_edf(psg_file, preload=True)
    annotations = mne.read_annotations(hyp_file)

    # Lọc annotations: chỉ giữ lại những cái nằm trong list valid_stages
    annotations = annotations[np.isin(annotations.description, valid_stages)]

    # Gán nhãn vào tín hiệu raw
    raw.set_annotations(annotations, emit_warning=False)

    # Định nghĩa mapping cụ thể để quản lý ID cho dễ
    mapping = {
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3': 3,
        'Sleep stage 4': 3,  # Gộp stage 3 và 4 thành stage 3 (N3)
        'Sleep stage R': 4
    }

    events, event_id = mne.events_from_annotations(
        raw,
        event_id=mapping,
        chunk_duration=float(duration)
    )

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=0,
        tmax=duration - (1/raw.info['sfreq']),
        baseline=None,
        preload=True,
        on_missing='warn'  # Nếu thiếu stage nào thì chỉ cảnh báo, không văng lỗi
    )

    return epochs


def filter_wake_epochs(epochs, wake_id=0, padding_minutes=30):
    # 1. Lấy danh sách nhãn của tất cả epochs
    labels = epochs.events[:, -1]

    # 2. Tìm vị trí các epoch không phải là Wake (0)
    non_wake_indices = np.where(labels != wake_id)[0]

    if len(non_wake_indices) == 0:
        return epochs  # Trả về nguyên bản nếu cả đêm không ngủ (hiếm gặp)

    # 3. Xác định mốc bắt đầu và kết thúc của vùng "ngủ"
    first_sleep_idx = non_wake_indices[0]
    last_sleep_idx = non_wake_indices[-1]

    # 4. Tính toán phần đệm (padding)
    # 30 phút = 1800 giây. Mỗi epoch 30s => padding = 1800 / 30 = 60 epochs
    padding_epochs = int(padding_minutes * 60 / 30)

    start_idx = max(0, first_sleep_idx - padding_epochs)
    end_idx = min(len(epochs) - 1, last_sleep_idx + padding_epochs)

    # 5. Cắt lấy vùng dữ liệu mong muốn
    filtered_epochs = epochs[start_idx: end_idx + 1]

    print(
        f"Đã cắt bớt Wake. Từ {len(epochs)} còn {len(filtered_epochs)} epochs.")
    return filtered_epochs


def feature_extraction(epochs_obj):
    # Lay thong tin co ban tu doi tuong epochs
    meas_date = epochs_obj.info['meas_date']  # Thời gian bắt đầu ghi file
    sfreq = epochs_obj.info['sfreq']         # Tần số lấy mẫu (100Hz)

    # events[:, 0] chứa sample index bắt đầu của từng epoch
    onsets_samples = epochs_obj.events[:, 0]
    labels = epochs_obj.events[:, -1]

    # Tinh toan PSD cho cac epoch va cac kenh EEG (0), EOG (2), EMG (4)
    psd_obj = epochs_obj.compute_psd(
        method='welch', fmin=0.5, fmax=30.0, n_fft=256, verbose=False)
    psds = psd_obj.get_data()
    freqs = psd_obj.freqs
    data = epochs_obj.get_data()

    all_features = []
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'spindle': (12, 14)
    }

    for i in range(len(epochs_obj)):
        # Tính thời gian bắt đầu của epoch hiện tại dựa trên sample index và tần số lấy mẫu
        onset_seconds = onsets_samples[i] / sfreq
        current_start_time = meas_date + timedelta(seconds=onset_seconds)

        f = {
            'index': i + 1,
            'start_time': current_start_time.strftime('%H:%M:%S'),
        }

        # Trich xuat EEG (Kênh 0) va tinh toan PSD cho cac band
        eeg_psd = psds[i, 0, :]
        for band, (fmin, fmax) in bands.items():
            # Tim cac chi so trong bang tan so va tinh trung binh cong
            idx = (freqs >= fmin) & (freqs <= fmax)
            f[f'eeg_{band}_power'] = np.mean(eeg_psd[idx])

        # Trich xuat EOG (Kênh 2)
        f['eog_var'] = np.var(data[i, 2, :])

        # Trich xuat EMG (Kênh 4)
        f['emg_rms'] = np.sqrt(np.mean(data[i, 4, :]**2))

        # Gán nhãn sleep stage
        f['label'] = labels[i]

        all_features.append(f)

    return pd.DataFrame(all_features)
