import numpy as np
import pywt
from sklearn.base import BaseEstimator, TransformerMixin


class WaveletFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, wavelet='db4', level=5):
        self.wavelet = wavelet
        self.level = level

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return extract_features(X, self.wavelet, self.level)


def extract_features(epochs_matrix, wavelet='db4', level=5):
    n_epochs, n_channels, _ = epochs_matrix.shape
    all_epoch_features = []

    for e in range(n_epochs):
        channel_features = []

        for ch in range(n_channels):
            signal = epochs_matrix[e, ch, :]

            # get 5 level like this [A5, D5, D4, D3, D2, D1]
            coeffs = pywt.wavedec(signal, wavelet, level=level)

            feat = []
            for band_coeffs in coeffs:
                energy = np.sum(np.square(band_coeffs))
                log_energy = np.log10(energy + 1e-5)
                std_dev = np.std(band_coeffs)
                mav = np.mean(np.abs(band_coeffs))
                feat.extend([log_energy, std_dev, mav])

            # 6 levels × 3 features = 18 features per channel
            channel_features.append(feat)

        # average across all 64 channels = shape (18,)
        epoch_features_averaged = np.mean(channel_features, axis=0)
        all_epoch_features.append(epoch_features_averaged)

    return np.array(all_epoch_features)    # shape (n_epochs, 18)
