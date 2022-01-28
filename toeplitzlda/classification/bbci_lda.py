from typing import Optional, Tuple

import mne
import numpy as np
import sklearn
import sklearn.linear_model
import sklearn.utils.multiclass
from pyriemann.utils.geodesic import geodesic
from scipy.stats import invwishart
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import empirical_covariance, fast_mcd, ledoit_wolf, oas
from sklearn.preprocessing import StandardScaler

from blockmatrix import SpatioTemporalMatrix, fortran_block_levinson

from toeplitzlda.usup_replay.llp import shrinkage


class EpochsVectorizer(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            permute_channels_and_time=True,
            select_ival=None,
            jumping_mean_ivals=None,
            averaging_samples=None,
            rescale_to_uv=True,
            mne_scaler=None,
            pool_times=False,
            to_numpy_only=False,
            copy=True,
    ):
        self.permute_channels_and_time = permute_channels_and_time
        self.jumping_mean_ivals = jumping_mean_ivals
        self.select_ival = select_ival
        self.averaging_samples = averaging_samples
        self.input_type = mne.BaseEpochs
        self.rescale_to_uv = rescale_to_uv
        self.scaling = 1e6 if self.rescale_to_uv else 1
        self.pool_times = pool_times
        self.to_numpy_only = to_numpy_only
        self.copy = copy
        self.mne_scaler = mne_scaler
        if self.select_ival is None and self.jumping_mean_ivals is None:
            raise ValueError("jumping_mean_ivals or select_ival is required")

    def fit(self, X, y=None):
        """fit."""
        return self

    def transform(self, X: mne.BaseEpochs):
        """transform."""
        e = X.copy() if self.copy else X
        if self.to_numpy_only:
            X = e.get_data() * self.scaling
            return X
        if self.jumping_mean_ivals is not None:
            self.averaging_samples = np.zeros(len(self.jumping_mean_ivals))
            X = e.get_data() * self.scaling
            new_X = np.zeros((X.shape[0], X.shape[1], len(self.jumping_mean_ivals)))
            for i, ival in enumerate(self.jumping_mean_ivals):
                np_idx = e.time_as_index(ival)
                idx = list(range(np_idx[0], np_idx[1]))
                self.averaging_samples[i] = len(idx)
                new_X[:, :, i] = np.mean(X[:, :, idx], axis=2)
            X = new_X
        elif self.select_ival is not None:
            e.crop(tmin=self.select_ival[0], tmax=self.select_ival[1])
            X = e.get_data() * self.scaling
        elif self.pool_times:
            X = e.get_data() * self.scaling
            raise ValueError("This should never be entered though.")
        else:
            assert False, "In the constructor, pass either select ival or jumping means."
        if self.mne_scaler is not None:
            X = self.mne_scaler.fit_transform(X)
        if self.permute_channels_and_time and not self.pool_times:
            X = X.transpose((0, 2, 1))
        if self.pool_times:
            X = np.reshape(X, (-1, X.shape[1]))
        else:
            X = np.reshape(X, (X.shape[0], -1))
        return X


def diag_indices_with_offset(p, offset):
    idxdiag = np.diag_indices(p)
    idxdiag_with_offset = list()
    idxdiag_with_offset.append(np.array([i + offset for i in idxdiag[0]]))
    idxdiag_with_offset.append(np.array([i + offset for i in idxdiag[1]]))
    return tuple(idxdiag_with_offset)


def iw_correction(X, standardize=True):
    p, n = X.shape

    if standardize:
        sc = StandardScaler()  # standardize features
        X = sc.fit_transform(X.T).T

    Xn = X - np.repeat(np.mean(X, axis=1, keepdims=True), n, axis=1)
    cov = np.matmul(Xn, Xn.T)
    idxdiag = np.diag_indices(p)

    # Target = B
    nu = np.mean(cov[idxdiag])

    iw_scale = np.sqrt(nu)
    perturbation = iw_scale * invwishart.rvs(p + 2, scale=np.eye(p))
    cov = cov + perturbation  # geodesic(cov, perturbation, 0.1)
    if standardize:  # scale back
        cov = sc.scale_[np.newaxis, :] * cov * sc.scale_[:, np.newaxis]
    return cov


# corresponds to train_RLDAshrink.m and clsutil_shrinkage.m from bbci_public
class ShrinkageLinearDiscriminantAnalysis(
    sklearn.base.BaseEstimator,
    sklearn.discriminant_analysis.LinearClassifierMixin,
):
    def __init__(
        self,
        priors=None,
        only_block=False,
        n_times=5,
        n_channels=31,
        pool_cov=True,
        standardize_shrink=True,
        shrink_block=False,
        use_xcov=False,
        calculate_oracle_mean=None,
        unit_w=False,
        fixed_gamma=None,
        use_invwishart_correction=False,
        enforce_toeplitz=False,
        use_fortran_solver=False,
        banding=None,
        tapering=None,
        data_is_channel_prime=True,
    ):
        self.only_block = only_block
        self.priors = priors
        self.n_times = n_times
        self.n_channels = n_channels
        self.pool_cov = pool_cov
        self.standardize_shrink = standardize_shrink
        self.shrink_block = shrink_block
        self.use_xcov = use_xcov
        self.calculate_oracle_mean = calculate_oracle_mean
        self.unit_w = unit_w
        self.fixed_gamma = fixed_gamma
        self.use_invwishart_correction = use_invwishart_correction
        self.enforce_toeplitz = enforce_toeplitz
        self.use_fortran_solver = use_fortran_solver
        if self.use_fortran_solver and not self.enforce_toeplitz:
            raise ValueError("Can only use Fortran solver when enforce_toeplitz=True")
        self.banding = banding
        self.tapering = tapering
        self.data_is_channel_prime = data_is_channel_prime

    def fit(self, X_train, y, xcov=None, oracle_data=None):
        # Section: Basic setup
        if self.calculate_oracle_mean is None:
            oracle_data = None
        self.classes_ = sklearn.utils.multiclass.unique_labels(y)
        if set(self.classes_) != {0, 1}:
            raise ValueError("currently only binary class supported")
        assert len(X_train) == len(y)
        xTr = X_train.T
        if not self.use_xcov:
            xcov = None

        n_classes = 2
        if self.priors is None:
            # here we deviate from the bbci implementation and
            # use the sample priors by default
            _, y_t = np.unique(y, return_inverse=True)  # non-negative ints
            priors = np.bincount(y_t) / float(len(y))
            # self.priors = np.array([1./n_classes] * n_classes)
        else:
            priors = self.priors

        # Section: covariance / mean calculation
        X, cl_mean = subtract_classwise_means(xTr, y)
        if self.calculate_oracle_mean == "clmean_and_covmean":
            _, cl_mean = subtract_classwise_means(oracle_data["x"].T, oracle_data["y"])
            X, _ = subtract_classwise_means(xTr, y, ext_mean=cl_mean)
        elif self.calculate_oracle_mean == "only_clmean":
            _, cl_mean = subtract_classwise_means(oracle_data["x"].T, oracle_data["y"])
        if self.pool_cov:
            if not self.use_invwishart_correction:
                C_cov, C_gamma = shrinkage(
                    X if xcov is None else xcov.T,
                    n_channels=self.n_channels,
                    n_times=self.n_times,
                    standardize=self.standardize_shrink,
                    block=self.shrink_block,
                    gamma=self.fixed_gamma,
                )
            else:
                C_cov = iw_correction(X if xcov is None else xcov.T)
        else:
            n_classes = 2
            C_cov = np.zeros((xTr.shape[0], xTr.shape[0]))
            for cur_class in range(n_classes):
                class_idxs = y == cur_class
                x_slice = X[:, class_idxs]
                C_cov += priors[cur_class] * shrinkage(x_slice)[0]

        stm = SpatioTemporalMatrix(
            C_cov,
            n_chans=self.n_channels,
            n_times=self.n_times,
            channel_prime=self.data_is_channel_prime,
        )
        if not self.data_is_channel_prime:
            stm.swap_primeness()
        if self.enforce_toeplitz:
            stm.force_toeplitz_offdiagonals()
        if self.banding is not None:
            stm.band_offdiagonals(self.banding)
        if self.tapering is not None:
            stm.taper_offdiagonals(taper_f=self.tapering)
        if not self.data_is_channel_prime:
            stm.swap_primeness()

        self.stored_stm = stm
        C_cov = stm.mat

        if self.only_block:
            C_cov_new = np.zeros_like(C_cov)
            for i in range(self.n_times):
                idx_start = i * self.n_channels
                idx_end = idx_start + self.n_channels
                C_cov_new[idx_start:idx_end, idx_start:idx_end] = C_cov[
                    idx_start:idx_end, idx_start:idx_end
                ]
            C_cov = C_cov_new

        # w = np.linalg.lstsq(C_cov, cl_mean, rcond=None)[0]
        if n_classes == 2:
            cl_mean = cl_mean[:, 1] - cl_mean[:, 0]
            prior_offset = np.log(priors[1] / priors[0])
        else:
            prior_offset = np.log(priors)

        if self.use_fortran_solver:
            w = fortran_block_levinson(C_cov, cl_mean, nch=self.n_channels, ntim=self.n_times)
        else:
            w = np.linalg.solve(C_cov, cl_mean)
        w = w / np.linalg.norm(w) if self.unit_w else w
        b = -0.5 * np.sum(cl_mean * w, axis=0).T + prior_offset

        # if n_classes == 2:
        #     w = w[:, 1] - w[:, 0]
        #     b = b[1] - b[0]

        self.coef_ = w.reshape((1, -1))
        self.intercept_ = b

    def predict_proba(self, X):
        """Estimate probability.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        C : array, shape (n_samples, n_classes)
            Estimated probabilities.
        """
        prob = self.decision_function(X)
        prob *= -1
        np.exp(prob, prob)
        prob += 1
        np.reciprocal(prob, prob)

        return np.column_stack([1 - prob, prob])

    def predict_log_proba(self, X):
        """Estimate log probability.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        C : array, shape (n_samples, n_classes)
            Estimated log probabilities.
        """
        return np.log(self.predict_proba(X))


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