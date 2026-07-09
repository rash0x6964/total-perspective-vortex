import pywt
import numpy as np
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
        epoch_features = []

        for ch in range(n_channels):
            signal = epochs_matrix[e, ch, :]

            # 5-level DWT: [A5, D5, D4, D3, D2, D1]
            coeffs = pywt.wavedec(signal, wavelet, level=level)

            for band_coeffs in coeffs:
                energy = np.sum(np.square(band_coeffs))
                log_energy = np.log10(energy + 1e-5)
                std_dev = np.std(band_coeffs)
                mav = np.mean(np.abs(band_coeffs))
                epoch_features.extend([log_energy, std_dev, mav])

        # 64 channels × 6 levels × 3 features = 1152 features per epoch
        all_epoch_features.append(epoch_features)

    return np.array(all_epoch_features)    # shape (n_epochs, 1152)
