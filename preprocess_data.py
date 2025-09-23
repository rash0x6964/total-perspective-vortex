import mne
from mne.datasets import eegbci
import numpy as np
import matplotlib.pyplot as plt

def preprocess_data(subject=1, runs=[4], use_advanced_features=True):
    """
    complete preprocessing pipeline for eeg motor movement data.
    returns: x (features), y (labels)
    """

    custom_data_path = '/home/rash/1337/total-perspective-vortex/dataset'
    mne.set_config('MNE_DATA_DIR', custom_data_path)

    file_paths = eegbci.load_data(subject, runs, update_path=True)
    raw = mne.io.read_raw_edf(file_paths[0], preload=True)

    fig_raw = raw.plot(n_channels=10, duration=20, scalings='auto', show=True)
    plt.show()

    # select relevant channels for hand movement
    key_channels = [
        'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.',  # fronto-central
        'C3..', 'C1..', 'Cz..', 'C2..', 'C4..',   # central (most important!)
        'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.',   # centro-parietal
    ]

    # filter to only available channels
    available_channels = [ch for ch in key_channels if ch in raw.ch_names]
    raw.pick_channels(available_channels)

    raw_filtered = raw.copy().filter(7., 30., method='iir', verbose=False)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    raw.plot_psd(fmax=50, ax=plt.gca(), show=False, average=True)
    plt.title('Power Spectral Density (RAW)')
    plt.ylim([0, 25])

    plt.subplot(1, 2, 2)
    raw_filtered.plot_psd(fmax=50, ax=plt.gca(), show=False, average=True)
    plt.title('Power Spectral Density (Filtered 7-30 Hz)')
    plt.ylim([0, 25])

    plt.tight_layout()
    plt.show()

    # extract events and create epochs
    events, event_dict = mne.events_from_annotations(raw_filtered)

    # focus on left vs right hand movements
    event_id = {'T1': 2, 'T2': 3}
    tmin, tmax = -1., 3.
    epochs = mne.Epochs(raw_filtered, events, event_id=event_id,
                       tmin=tmin, tmax=tmax, baseline=(None, 0),
                       preload=True, verbose=False)

    # feature extraction
    if use_advanced_features:
        # log-variance
        X_logvar = np.log(epochs.get_data().var(axis=2))
        print(f"✓ Log-variance features: {X_logvar.shape}")

        # spectral power by frequency bands
        freqs = np.arange(8, 31, 2)  # 8-30 hz in 2 hz steps
        n_cycles = freqs / 2.

        # compute time-frequency decomposition
        power = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles,
                                            use_fft=True, return_itc=False, decim=3,
                                            n_jobs=1, verbose=False)

        # average power across time for each frequency band
        X_spectral = power.data.mean(axis=-1)
        X_spectral = X_spectral.reshape(X_spectral.shape[0], -1)  # flatten to 2d
        print(f"✓ Spectral power features: {X_spectral.shape}")

        # combine both feature types
        X = np.concatenate([X_logvar, X_spectral], axis=1)

    else:
        # log-variance
        X = np.log(epochs.get_data().var(axis=2))
        print(f"✓ Simple log-variance features: {X.shape}")

    y = epochs.events[:, -1]  # labels

    print("\n=== PREPROCESSING COMPLETE ===")
    print(f"Final feature matrix X shape: {X.shape}")
    print(f"Label vector y shape: {y.shape}")

    return X, y, epochs

if __name__ == "__main__":
    # run preprocessing
    X, y, epochs = preprocess_data()

    # save the processed data for next steps
    np.save('X_features.npy', X)
    np.save('y_labels.npy', y)
