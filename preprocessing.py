import mne
import numpy as np


DATA_PATH = "/home/rben-mou/goinfre/dataset/physionet.org/files/eegmmidb/1.0.0"
EXPERIMENT_RUNS = {
    0: [3, 7, 11],  # real left fist vs right fist
    1: [4, 8, 12],  # imagined left fist vs right fist
    2: [5, 9, 13],  # real fists vs feet
    3: [6, 10, 14],  # imagined fists vs feet
    4: {  # real fists vs imagined fists
        'event':    'T1',
        'real':     [5, 9, 13],
        'imagined': [6, 10, 14],
    },
    5: {  # real feet vs imagined feet
        'event':    'T2',
        'real':     [5, 9, 13],
        'imagined': [6, 10, 14],
    }
}


def load_raw(subject, run):
    path = f"{DATA_PATH}/S{subject:03d}/S{subject:03d}R{run:02d}.edf"
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    return raw


def filter_raw(raw, l_freq=8.0, h_freq=30.0):
    raw.filter(l_freq, h_freq, fir_window='hamming', verbose=False)
    return raw


def get_events(raw, event_id=None):
    if event_id is None:
        event_id = {'T1': 1, 'T2': 2}

    events, _ = mne.events_from_annotations(raw, verbose=False)

    # keep only T1 and T2 events
    mask = np.isin(events[:, 2], list(event_id.values()))
    events = events[mask]

    return events, event_id


def make_epochs(raw, events, event_id, tmin=0.0, tmax=4.0):
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False
    )
    return epochs


def load_standard_experiment(subject, runs):
    all_epochs = []

    for run in runs:
        raw = load_raw(subject, run)
        raw = filter_raw(raw)
        events, eid = get_events(raw)
        epochs = make_epochs(raw, events, eid)
        all_epochs.append(epochs)

    combined = mne.concatenate_epochs(all_epochs, verbose="error")

    X = combined.get_data()             # (n_epochs, n_channels, n_times)
    y = combined.events[:, 2]           # 1 for T1, 2 for T2

    return X, y


def load_realvsimagined_experiment(subject, config):
    target_event = config['event']
    event_id = {target_event: 1}

    all_X = []
    all_y = []

    for run in config['real']:
        raw = load_raw(subject, run)
        raw = filter_raw(raw)
        events, eid = get_events(raw, event_id)
        epochs = make_epochs(raw, events, eid)
        X = epochs.get_data()
        y = np.ones(len(X), dtype=int)
        all_X.append(X)
        all_y.append(y)

    for run in config['imagined']:
        raw = load_raw(subject, run)
        raw = filter_raw(raw)
        events, eid = get_events(raw, event_id)
        epochs = make_epochs(raw, events, eid)
        X = epochs.get_data()
        y = np.ones(len(X), dtype=int) * 2
        all_X.append(X)
        all_y.append(y)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    return X, y


def load_experiment(subject, experiment):
    config = EXPERIMENT_RUNS[experiment]

    if isinstance(config, list):
        # experiments 0-3: standard T1/T2 labeling
        return load_standard_experiment(subject, config)
    else:
        # experiments 4-5: real vs imagined labeling
        return load_realvsimagined_experiment(subject, config)


def visualize_raw(subject=1, run=3):
    raw = load_raw(subject, run)
    raw_copy = raw.copy()
    raw = filter_raw(raw)

    raw_copy.plot(n_channels=10, title='Before filtering', block=True)
    raw.plot(n_channels=10, title='After filtering (8-30 Hz)', block=True)


if __name__ == "__main__":
    print("Loading experiment 0 for subject 1...")
    X, y = load_experiment(subject=1, experiment=0)
    print(f"X shape: {X.shape}")   # (n_epochs, 64, n_times)
    print(f"y shape: {y.shape}")   # (n_epochs,)
    print(f"Classes: {np.unique(y)}")

    print("\nLoading experiment 4 for subject 1...")
    X, y = load_experiment(subject=1, experiment=4)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Classes: {np.unique(y)}")
