import mne
import numpy as np


DATA_PATH = "/media/rben-mou/FEDORA-WS-L/dataset/physionet.org/files/eegmmidb/1.0.0"
# DATA_PATH = "/home/rben-mou/goinfre/dataset/physionet.org/files/eegmmidb/1.0.0"
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


def filter_raw(raw, l_freq=7.0, h_freq=30.0):
    raw.filter(l_freq, h_freq, verbose=False)
    return raw


def get_events(raw, event_id=None):
    events, mne_event_id = mne.events_from_annotations(raw, verbose=False)

    if event_id is None:
        wanted = ['T1', 'T2']
    else:
        wanted = list(event_id.keys())

    real_codes = {name: mne_event_id[name]
                  for name in wanted if name in mne_event_id}

    if not real_codes:
        raise ValueError(
            f"None of {wanted} found in this file's annotations: {mne_event_id}"
        )

    mask = np.isin(events[:, 2], list(real_codes.values()))
    events = events[mask]

    final_event_id = {}
    for i, name in enumerate(wanted, start=1):
        if name in real_codes:
            old_code = real_codes[name]
            events[events[:, 2] == old_code, 2] = i
            final_event_id[name] = i

    return events, final_event_id


def make_epochs(raw, events, event_id, tmin=0.5, tmax=4.5):
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


def plotting(subject=1, run=3):
    raw = load_raw(subject, run)

    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False, on_missing='ignore')

    raw.plot(n_channels=10, title='Before filtering', block=True)

    raw_filtered = filter_raw(raw.copy())

    fig1 = raw_filtered.compute_psd(fmax=80).plot(show=False)
    fig2 = raw_filtered.compute_psd(fmax=80).plot(show=False, average=True)

    import matplotlib.pyplot as plt
    plt.show(block=True)


if __name__ == "__main__":
    X, y = load_experiment(subject=1, experiment=0)
    plotting()
    print(f"X shape:  {X.shape}")
    print(f"Classes:  {np.unique(y)}")
    print(f"Class 1:  {np.sum(y == 1)}, Class 2: {np.sum(y == 2)}")
