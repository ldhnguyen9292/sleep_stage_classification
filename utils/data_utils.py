import mne
import os
import pandas as pd
from datetime import timedelta

from config import BASE_DIR, INTERNAL_DIR, EXTERNAL_DIR

origin_folder_path = 'sleep-edf-database-expanded-1.0.0'
folder_names = ['sleep-cassette', 'sleep-telemetry']


def get_file_lists(path):
    csv_files = sorted([f for f in os.listdir(path) if f.endswith('.csv')])
    return csv_files


def find_edf_files(selected_csv):
    patient_id = selected_csv.split('_')[-1].replace('.csv', '')
    edf_files_in_dir = os.listdir(os.path.join(
        origin_folder_path, folder_names[0]))

    psg_file = next(
        (f for f in edf_files_in_dir if patient_id in f and 'PSG.edf' in f), None)
    hypno_file = next(
        (f for f in edf_files_in_dir if patient_id in f and 'Hypnogram.edf' in f), None)

    return psg_file, hypno_file, patient_id


def load_mne_raw(psg_name, hypno_name=None):
    psg_path = os.path.join(origin_folder_path, folder_names[0], psg_name)
    raw = mne.io.read_raw_edf(psg_path, preload=False, verbose=False)

    if hypno_name:
        hypno_path = os.path.join(
            origin_folder_path, folder_names[0], hypno_name)
        annot = mne.read_annotations(hypno_path)
        raw.set_annotations(annot)

    return raw


def get_hypnogram_data(raw):
    if not raw.annotations:
        return pd.DataFrame()

    m_date = raw.info['meas_date']
    ann_data = []
    for ann in raw.annotations:
        ann_data.append({
            'Start': m_date + timedelta(seconds=ann['onset']),
            'End': m_date + timedelta(seconds=ann['onset'] + ann['duration']),
            'Stage': ann['description'],
            'Duration': ann['duration']
        })
    return pd.DataFrame(ann_data)
