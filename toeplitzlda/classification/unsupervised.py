from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import sklearn
import sklearn.linear_model
import sklearn.utils.multiclass
from blockmatrix import (
    SpatioTemporalMatrix,
    fortran_block_levinson,
    fortran_cov_mean_transformation,
    linear_taper,
)
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted, check_X_y

from toeplitzlda.classification import ShrinkageLinearDiscriminantAnalysis
from toeplitzlda.classification.covariance import (
    calc_n_times,
    shrinkage,
    subtract_classwise_means,
)


class ExternalLDA(ShrinkageLinearDiscriminantAnalysis):
    """No capabilities except calculating w from external cov/mean information."""

    def __init__(
        self,
        means=None,
        cov=None,
    ):
        self.means = means
        self.cov = cov

    def calc(self):
        if self.means is None or self.cov is None:
            raise ValueError(f"Cannot calculate weights, missing information.")

        w = np.linalg.solve(self.cov, self.means)
        self.classes_ = np.arange(0, w.shape[1])
        # TODO: implement prior_offset
        b = -0.5 * np.sum(self.means * w, axis=0).T  # + prior_offset

        self.coef_ = w.T
        self.intercept_ = b

    def fit(self, X_train, y, oracle_data=None):
        pass
