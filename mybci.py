import numpy as np
import mne
import matplotlib.pyplot as plt


sample_data_folder = mne.datasets.sample.data_path()
print(sample_data_folder)

sample_data_raw_file = (
    sample_data_folder / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"
)

raw = mne.io.read_raw_fif(sample_data_raw_file)

print(raw)
print(raw.info)

# raw.compute_psd(fmax=50).plot(picks="data", exclude="bads", amplitude=False)
raw.plot()


plt.show()
