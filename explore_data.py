import mne
from mne.datasets import eegbci
import numpy as np
import matplotlib.pyplot as plt


mne.set_log_level('ERROR')

subject = 1
runs = [4]

eegbci.load_data(subject, runs, update_path=True)

file_paths = eegbci.load_data(subject, runs)
raw_file = file_paths[0]

print(f"Data for subject {subject}, run {runs[0]} is located at:")
print(raw_file)

raw = mne.io.read_raw_edf(raw_file, preload=True)

print("\n--- BASIC INFO ---")
print(raw)
print("\n--- DETAILED INFO ---")
print(raw.info)



raw.plot(n_channels=10, duration=20, scalings='auto')



# Extract the events from the raw data.
events, event_dict = mne.events_from_annotations(raw)
print("\n--- EVENT DICTIONARY ---")
print(event_dict)
print("\n--- FIRST 10 EVENTS ---")
print(events[:10])

fig = raw.plot(events=events, n_channels=10, duration=20, scalings='auto')
plt.show()


