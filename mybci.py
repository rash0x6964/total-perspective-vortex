import sys
import os
import time
import numpy as np
import joblib
import mne

from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from preprocessing import load_experiment, EXPERIMENT_RUNS


MODELS_DIR = "models"

TEST_SIZE = 0.2
N_SUBJECTS = 109
STREAM_DELAY = 2.0


mne.set_log_level('ERROR')


def build_pipeline():
    pipeline = Pipeline([
        ('csp',     CSP(n_components=4, log=True, norm_trace=True)),
        ('scaler',  StandardScaler()),
        ('clf',     SVC(kernel='rbf', C=1.0))
    ])
    return pipeline


def train(subject, experiment, realtime=False):
    if not realtime:
        print(
            f"\nLoading data — subject {subject:03d}, experiment {experiment}...")
    
    X, y = load_experiment(subject, experiment)

    if not realtime:
        print(f"Data shape: {X.shape}, classes: {np.unique(y)}")

    n_total = len(X)
    n_test = int(n_total * TEST_SIZE)
    n_train = n_total - n_test

    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    if not realtime:
        print(f"Train: {n_train} epochs | Test: {n_test} epochs")

    pipeline = build_pipeline()
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train,
                             cv=cv, scoring='accuracy')

    if not realtime:
        print(f"\ncross_val_score: {np.round(scores, 4)}")
        print(f"mean CV accuracy: {scores.mean():.4f}")

    pipeline.fit(X_train, y_train)

    test_accuracy = pipeline.score(X_test, y_test)
    if not realtime:
        print(f"test accuracy:    {test_accuracy:.4f}")

        os.makedirs(MODELS_DIR, exist_ok=True)
        model_path = _model_path(subject, experiment)
        joblib.dump(pipeline, model_path)
        print(f"\nModel saved to {model_path}")
    else:
        print(
            f"experiment {experiment}: subject {subject:03d}: accuracy = {test_accuracy:.4f}")

    return scores, test_accuracy


def predict(subject, experiment):
    model_path = _model_path(subject, experiment)

    if not os.path.exists(model_path):
        print(f"No trained model found at {model_path}")
        print(f"Run: python mybci.py {subject} {experiment} train")
        sys.exit(1)

    print(f"\nLoading model from {model_path}...")
    pipeline = joblib.load(model_path)

    print(f"Loading data — subject {subject:03d}, experiment {experiment}...")
    X, y = load_experiment(subject, experiment)

    # use only the test portion (last 20%) to avoid evaluating on training data
    n_total = len(X)
    n_test = int(n_total * TEST_SIZE)
    X_test = X[n_total - n_test:]
    y_test = y[n_total - n_test:]

    print(f"\nepoch nb: [prediction] [truth] equal?")

    correct = 0
    for i, (epoch, truth) in enumerate(zip(X_test, y_test)):
        # epoch reshapeing from (n_channels, n_times) to (1, n_channels, n_times)
        epoch_2d = epoch[np.newaxis, ...]

        t_start = time.time()
        prediction = pipeline.predict(epoch_2d)[0]
        t_elapsed = time.time() - t_start

        if t_elapsed > STREAM_DELAY:
            print(
                f"WARNING: prediction took {t_elapsed:.3f}s (>{STREAM_DELAY}s limit)")

        match = prediction == truth
        correct += int(match)

        print(
            f"epoch {i:02d}: "
            f"[{prediction}] [{truth}] "
            f"{'True' if match else 'False'}"
        )

    accuracy = correct / len(y_test)
    print(f"\nAccuracy: {accuracy:.4f}")

    return accuracy


def evaluate_all():
    experiment_accuracies = {exp: [] for exp in EXPERIMENT_RUNS}

    for experiment in EXPERIMENT_RUNS:
        for subject in range(1, N_SUBJECTS + 1):
            try:
                _, test_accuracy = train(subject, experiment, True)
                experiment_accuracies[experiment].append(test_accuracy)
            except Exception as e:
                print(
                    f"experiment {experiment}: subject {subject:03d}: FAILED — {e}")
                experiment_accuracies[experiment].append(0.0)

    print(
        f"\nMean accuracy of the six experiments for all {N_SUBJECTS} subjects:")

    all_means = []
    for experiment, accuracies in experiment_accuracies.items():
        mean = np.mean(accuracies)
        all_means.append(mean)
        print(f"experiment {experiment}: accuracy = {mean:.4f}")

    print(f"\nMean accuracy of 6 experiments: {np.mean(all_means):.4f}")


def _model_path(subject, experiment):
    return os.path.join(MODELS_DIR, f"subject_{subject:03d}_exp_{experiment}.pkl")


def _parse_args():
    args = sys.argv[1:]

    if len(args) == 0:
        return None, None, 'all'

    if len(args) == 3:
        subject = int(args[0])
        experiment = int(args[1])
        mode = args[2].lower()

        if mode not in ('train', 'predict'):
            print(f"Unknown mode '{mode}'. Use 'train' or 'predict'.")
            sys.exit(1)

        if not (1 <= subject <= N_SUBJECTS):
            print(f"Subject must be between 1 and {N_SUBJECTS}.")
            sys.exit(1)

        if experiment not in EXPERIMENT_RUNS:
            print(f"Experiment must be one of {list(EXPERIMENT_RUNS.keys())}.")
            sys.exit(1)

        return subject, experiment, mode

    print("Usage:")
    print("  python mybci.py                          # evaluate all")
    print("  python mybci.py <subject> <exp> train    # train")
    print("  python mybci.py <subject> <exp> predict  # predict")
    sys.exit(1)


if __name__ == "__main__":
    subject, experiment, mode = _parse_args()

    if mode == 'train':
        train(subject, experiment)

    elif mode == 'predict':
        predict(subject, experiment)

    elif mode == 'all':
        evaluate_all()
