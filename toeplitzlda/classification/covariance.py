from typing import Optional, Tuple

import numpy as np
from blockmatrix import SpatioTemporalMatrix, linear_taper
from sklearn import config_context
from sklearn.covariance import LedoitWolf, ledoit_wolf
from sklearn.preprocessing import StandardScaler


def shrinkage(
    X: np.ndarray,
    gamma: Optional[float] = None,
    T: Optional[np.ndarray] = None,
    S: Optional[np.ndarray] = None,
    standardize: bool = True,
) -> Tuple[np.ndarray, float]:
    p, n = X.shape

    if standardize:
        sc = StandardScaler()
        X = sc.fit_transform(X.T).T
    Xn = X - np.repeat(np.mean(X, axis=1, keepdims=True), n, axis=1)
    if S is None:
        S = np.matmul(Xn, Xn.T)
    Xn2 = np.square(Xn)
    idxdiag = np.diag_indices(p)

    nu = np.mean(S[idxdiag])
    if T is None:
        T = nu * np.eye(p, p)

    # Ledoit Wolf
    V = 1.0 / (n - 1) * (np.matmul(Xn2, Xn2.T) - np.square(S) / n)
    if gamma is None:
        gamma = n * np.sum(V) / np.sum(np.square(S - T))
    if gamma > 1:
        print("logger.warning('forcing gamma to 1')")
        gamma = 1
    elif gamma < 0:
        print("logger.warning('forcing gamma to 0')")
        gamma = 0
    Cstar = (gamma * T + (1 - gamma) * S) / (n - 1)
    if standardize:  # scale back
        Cstar = sc.scale_[np.newaxis, :] * Cstar * sc.scale_[:, np.newaxis]

    return Cstar, gamma


def subtract_classwise_means(xTr, y, ext_mean=None):
    n_classes = len(np.unique(y))
    n_features = xTr.shape[0]
    X = np.zeros((n_features, 0))
    cl_mean = np.zeros((n_features, n_classes))
    for ci, cur_class in enumerate(np.unique(y)):
        class_idxs = y == cur_class
        cl_mean[:, ci] = np.mean(xTr[:, class_idxs], axis=1)

        if ext_mean is None:
            X = np.concatenate(
                [
                    X,
                    xTr[:, class_idxs]
                    - np.dot(cl_mean[:, ci].reshape(-1, 1), np.ones((1, np.sum(class_idxs)))),
                ],
                axis=1,
            )
        else:
            X = np.concatenate(
                [
                    X,
                    xTr[:, class_idxs]
                    - np.dot(ext_mean[:, ci].reshape(-1, 1), np.ones((1, np.sum(class_idxs)))),
                ],
                axis=1,
            )
    return X, cl_mean


def calc_n_times(dim, n_channels, n_times):
    if type(n_times) is int:
        return n_times
    elif n_times == "infer":
        if dim % n_channels != 0:
            raise ValueError(f"Could not infer time samples. Remainder is non-zero.")
        else:
            return dim // n_channels
    else:
        raise ValueError(f"Unknown value for n_times: {n_times}")


class ToepTapLW(LedoitWolf):
    """An sklearn compatible covariance estimator"""

    def __init__(
        self,
        n_channels=None,
        *,
        n_times="infer",
        data_is_channel_prime=True,
        standardize=True,
        tapering=linear_taper,
        only_lw=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.standardize = standardize
        self.tapering = tapering
        if self.tapering is None:
            print(
                "WARNING: using block-Toeplitz structure without tapering can lead to numerical "
                "instabilities / singular matrices"
            )
            # Use a no-op taper that always returns 1 as scaling factor
            self.tapering = lambda d, dmax: 1
        self.n_times = n_times
        self.n_channels = n_channels
        self.data_is_channel_prime = data_is_channel_prime
        self.only_lw = only_lw

    def fit(self, X, y=None):
        """Fit the covariance model to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = self._validate_data(X)
        dim = X.shape[1]
        if self.assume_centered:
            self.location_ = np.zeros(X.shape[1])
        else:
            self.location_ = X.mean(0)
        with config_context(assume_finite=True):
            X_cent = X - self.location_
            sc = None
            if self.standardize:
                sc = StandardScaler()
                X_cent = sc.fit_transform(X_cent)
            covariance, shrinkage_gamma = ledoit_wolf(
                X_cent,
                assume_centered=True,
                block_size=self.block_size,
            )
            # covariance = empirical_covariance(
            #     X_cent, assume_centered=True
            # )
            # covariance, shrinkage = oas(X_cent, assume_centered=True)
            if self.standardize:  # scale back
                covariance = sc.scale_[np.newaxis, :] * covariance * sc.scale_[:, np.newaxis]
        self.shrinkage_ = shrinkage_gamma

        nt = calc_n_times(dim, self.n_channels, self.n_times)
        stm = SpatioTemporalMatrix(
            covariance,
            n_times=nt,
            n_chans=self.n_channels,
            channel_prime=self.data_is_channel_prime,
        )

        if not self.only_lw:
            if not self.data_is_channel_prime:
                stm.swap_primeness()
            stm.force_toeplitz_offdiagonals()
            stm.taper_offdiagonals(self.tapering)
            if not self.data_is_channel_prime:
                stm.swap_primeness()
            covariance = stm.mat
        self.stm_ = stm
        self._set_covariance(covariance)

        return self
