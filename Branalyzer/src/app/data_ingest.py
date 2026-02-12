from mne.datasets import eegbci
import os

DATA_PATH = "./data"
os.makedirs(DATA_PATH, exist_ok=True)

subjects = [1]

imagery_runs_right_left = [4, 8, 12]
motor_runs_right_left = [3, 7, 11]
baseline_runs = [1, 2]
# Info on motor imagery loading can be found at: https://mne.tools/stable/generated/mne.datasets.eegbci.load_data.html

for subject in subjects:
    eegbci.load_data(subject, imagery_runs_right_left, path=DATA_PATH)

print("Data ingestion complete.")