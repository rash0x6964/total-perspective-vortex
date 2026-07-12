import mne
import numpy as np
import os
import warnings
 
 
warnings.filterwarnings('ignore', category=RuntimeWarning)
mne.set_log_level('ERROR')

DATA_PATH = "/media/rben-mou/FEDORA-WS-L/dataset_/BCICIV_2a_gdf"
N_SUBJECTS = 9
L_FREQ = 8.0
H_FREQ = 30.0
TMIN = 0.5
TMAX = 4.5

EVENT_ID = {
    '769': 1,    # left hand
    '770': 2,    # right hand
    '771': 3,    # feet
    '772': 4,    # tongue
}

EEG_CHANNELS = [
    'EEG-Fz', 'EEG-0',  'EEG-1',  'EEG-2',  'EEG-3',
    'EEG-4',  'EEG-5',  'EEG-C3', 'EEG-6',  'EEG-Cz',
    'EEG-7',  'EEG-C4', 'EEG-8',  'EEG-9',  'EEG-10',
    'EEG-11', 'EEG-12', 'EEG-13', 'EEG-14', 'EEG-Pz',
    'EEG-15', 'EEG-16',
]


EXPERIMENT_RUNS = {
    0: ('769', '770'),   # left hand  vs right hand
    1: ('769', '771'),   # left hand  vs feet
    2: ('769', '772'),   # left hand  vs tongue
    3: ('770', '771'),   # right hand vs feet
    4: ('770', '772'),   # right hand vs tongue
    5: ('771', '772'),   # feet       vs tongue
}


def load_raw(subject, session='T'):
    path = os.path.join(DATA_PATH, f"A{subject:02d}{session}.gdf")
    raw = mne.io.read_raw_gdf(path, preload=True, verbose=False)

    raw.pick_channels(EEG_CHANNELS)

    return raw


def filter_raw(raw, l_freq=L_FREQ, h_freq=H_FREQ):
    raw.filter(l_freq, h_freq, fir_window='hamming', verbose=False)
    return raw


def get_events(raw, class1, class2):
    events, mne_event_id = mne.events_from_annotations(raw, verbose=False)

    wanted = {class1: 1, class2: 2}
    real_codes = {
        name: mne_event_id[name]
        for name in wanted
        if name in mne_event_id
    }

    if len(real_codes) == 0:
        raise ValueError(
            f"Classes {class1}, {class2} not found in {list(mne_event_id.keys())}")

    mask = np.isin(events[:, 2], list(real_codes.values()))
    events = events[mask].copy()

    final_event_id = {}
    for name, label in wanted.items():
        if name in real_codes:
            old_code = real_codes[name]
            events[events[:, 2] == old_code, 2] = label
            final_event_id[name] = label

    return events, final_event_id


def make_epochs(raw, events, event_id, tmin=TMIN, tmax=TMAX):
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


def load_experiment(subject, experiment):
    class1, class2 = EXPERIMENT_RUNS[experiment]

    all_X = []
    all_y = []

    for session in ['T']:
        try:
            raw = load_raw(subject, session)
            raw = filter_raw(raw)
            events, eid = get_events(raw, class1, class2)
            epochs = make_epochs(raw, events, eid)
            all_X.append(epochs.get_data())
            all_y.append(epochs.events[:, 2])
        except Exception as e:
            print(f"Warning: subject {subject} session {session} failed: {e}")

    if not all_X:
        raise RuntimeError(
            f"No data loaded for subject {subject} experiment {experiment}")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    return X, y
