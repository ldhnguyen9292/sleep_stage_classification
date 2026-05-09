# Sleep-EDF EEG Processing Pipeline

## 1. Dataset Overview

Dataset sử dụng: entity["other","Sleep-EDF Expanded","sleep EEG dataset"]

Dataset chứa:

- EEG
- EOG
- EMG
- Hypnogram (sleep stage labels)

### File types

#### PSG file

Ví dụ:

```text
SC4001E0-PSG.edf
```

Chứa tín hiệu sinh học thô:

- EEG
- EOG
- EMG

---

#### Hypnogram file

Ví dụ:

```text
SC4001EC-Hypnogram.edf
```

Chứa nhãn sleep stages:

- W
- REM
- N1
- N2
- N3

---

## 2. EEG Information

### EEG channels

- Fpz-Cz
- Pz-Oz

### Sampling rate

```text
100 Hz
```

Nghĩa là:

- 100 sample / second

---

## 3. Install Libraries

```bash
pip install mne numpy scipy matplotlib
```

Optional:

```bash
pip install antropy pywavelets
```

---

# 4. Read EDF File

```python
import mne

raw = mne.io.read_raw_edf(
    "SC4001E0-PSG.edf",
    preload=True
)
```

---

## 5. Inspect Recording Information

### Print raw object

```python
print(raw)
```

---

### Channel names

```python
print(raw.ch_names)
```

---

### Sampling rate

```python
print(raw.info['sfreq'])
```

---

## 6. Extract EEG Channel

```python
eeg = raw.get_data(picks=['Fpz-Cz'])
```

Shape:

```text
(channel, sample)
```

Ví dụ:

```text
(1, 720000)
```

Convert thành vector:

```python
eeg = eeg.flatten()
```

---

## 7. Visualize EEG

```python
raw.plot()
```

Plot riêng EEG channel:

```python
raw.plot(picks=['Fpz-Cz'])
```

---

# 8. EEG Preprocessing

### Bandpass filter

```python
raw.filter(0.5, 30)
```

Mục tiêu:

- loại bỏ noise
- giữ EEG frequencies quan trọng

---

# 9. Read Hypnogram Labels

```python
annotations = mne.read_annotations(
    "SC4001EC-Hypnogram.edf"
)
```

Attach labels vào raw EEG:

```python
raw.set_annotations(annotations)
```

---

# 10. Convert EEG into 30-second Epochs

Sleep staging chuẩn dùng:

```text
30-second epochs
```

Vì:

- hypnogram labels được annotate theo 30s

---

### Create events

```python
events, event_id = mne.events_from_annotations(raw)
```

---

### Create epochs

```python
epochs = mne.Epochs(
    raw,
    events,
    event_id,
    tmin=0,
    tmax=30,
    preload=True
)
```

---

### Epoch shape

```python
print(epochs.get_data().shape)
```

Ví dụ:

```text
(number_epoch, channel, sample)
```

```text
(900, 2, 3000)
```

Giải thích:

- 900 epochs
- 2 EEG channels
- 3000 samples mỗi epoch

Vì:

```text
100 Hz × 30 s = 3000 samples
```

---

# 11. Feature Extraction

Mỗi epoch EEG sẽ được convert thành feature vector.

Ví dụ features:

- delta power
- theta power
- alpha power
- beta power
- entropy
- variance

---

## Frequency bands

| Band  | Frequency |
| ----- | --------- |
| Delta | 0.5–4 Hz  |
| Theta | 4–8 Hz    |
| Alpha | 8–13 Hz   |
| Beta  | 13–30 Hz  |

---

## PSD / Welch Example

```python
from scipy.signal import welch

freqs, psd = welch(epoch, fs=100)
```

Extract alpha power:

```python
alpha = psd[(freqs >= 8) & (freqs <= 13)].mean()
```

---

## Example feature vector

```text
[
 delta_power,
 theta_power,
 alpha_power,
 beta_power,
 entropy
]
```

---

# 12. Machine Learning Pipeline

```text
Raw EEG
↓
Filtering
↓
Epoching
↓
Feature Extraction
↓
ML Model
↓
Sleep Stage Prediction
```

---

# 13. Deep Learning Approach

Thay vì extract feature thủ công:

```text
Raw EEG Epoch
↓
CNN / LSTM
↓
Automatic Feature Learning
↓
Sleep Stage Prediction
```

---

# 14. Important Notes

## EDF format

EDF là binary biomedical format.
Không phải CSV.

---

## Shape convention

MNE thường dùng:

```text
(channel, sample)
```

Epochs:

```text
(epoch, channel, sample)
```

---

## Sleep staging labels

Thông thường sẽ merge:

- Stage 3
- Stage 4

thành:

```text
N3
```

---

# 15. Recommended Next Steps

- Clean labels
- Remove movement / unknown labels
- Normalize EEG
- Extract PSD features
- Build baseline Random Forest / SVM
- Try CNN/LSTM later

---

# 16. Useful Libraries

- MNE-Python
- NumPy
- SciPy
- AntroPy
- PyWavelets

---

# 17. References

- Sleep-EDF Expanded
- MNE documentation
- PhysioNet
