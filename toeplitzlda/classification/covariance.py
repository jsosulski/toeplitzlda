import numpy as np
from blockmatrix import SpatioTemporalMatrix, linear_taper
from sklearn import config_context
from sklearn.covariance import LedoitWolf, ledoit_wolf
from sklearn.preprocessing import StandardScaler


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
        standardize=True,
        tapering=linear_taper,
        n_times=None,
        n_channels=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.standardize = standardize
        self.tapering = tapering
        if self.tapering is None:
            print(
                "WARNING: using block-Toeplitz structure without tapering can lead to numerical "
                "instabilities"
            )
            # Use a no-op taper
            self.tapering = lambda d, dmax: 1
        self.n_times = n_times
        self.n_channels = n_channels

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
            covariance, shrinkage = ledoit_wolf(
                X_cent,
                assume_centered=True,
                block_size=self.block_size,
            )
            # covariance = empirical_covariance(
            #     X_cent, assume_centered=True
            # )
            # covariance, shrinkage = oas(X_cent, assume_centered=True)
            if self.standardize:  # scale back
                covariance = (
                    sc.scale_[np.newaxis, :] * covariance * sc.scale_[:, np.newaxis]
                )
        self.shrinkage_ = shrinkage

        nt = calc_n_times(dim, self.n_channels, self.n_times)
        stm = SpatioTemporalMatrix(covariance, n_times=nt, n_chans=self.n_channels)
        stm.force_toeplitz_offdiagonals()
        stm.taper_offdiagonals(self.tapering)
        covariance = stm.mat
        self._set_covariance(covariance)

        return self
