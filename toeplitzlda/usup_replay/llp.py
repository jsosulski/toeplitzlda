import numpy as np
from blockmatrix import SpatioTemporalMatrix, fortran_block_levinson
from sklearn.base import BaseEstimator

from toeplitzlda.classification.covariance import calc_n_times
from toeplitzlda.classification.toeplitzlda import shrinkage, subtract_classwise_means


class LearningFromLabelProportions(BaseEstimator):
    """
    Learning from label proportions SLDA from Hübner et al 2017 [1]_.

    Parameters
    ----------
    ratio_matrix: np.ndarray of shape (2, 2)
        | ratio seq1/target   ratio seq1/non-target |
        | ratio seq2/target   ratio seq2/non-target |

    References
    ----------
    .. [1] Hübner, D., Verhoeven, T., Schmid, K., Müller, K. R., Tangermann, M., & Kindermans, P. J. (2017)
           Learning from label proportions in brain-computer interfaces: Online unsupervised learning with guarantees.
           PLOS ONE 12(4): e0175856.
           https://doi.org/10.1371/journal.pone.0175856
    """

    def __init__(
        self,
        ratio_matrix: np.ndarray = None,
        toeplitz_time=False,
        taper_time=None,
        toeplitz_spatial=False,
        taper_spatial=None,
        use_fortran_solver=False,
        n_times=None,
        n_channels=31,
    ):
        if ratio_matrix is None:
            self.ratio_matrix = np.array([[3 / 8, 5 / 8], [2 / 18, 16 / 18]])
        self.pinv_ratio_matrix = np.linalg.inv(self.ratio_matrix)
        self.w = None
        self.b = None
        self.n_times = n_times
        self.n_channels = n_channels
        self.toeplitz_time = toeplitz_time
        self.taper_time = taper_time
        self.toeplitz_spatial = toeplitz_spatial
        self.taper_spatial = taper_spatial
        self.use_fortran_solver = use_fortran_solver

        self.mu_T = None
        self.mu_NT = None

        self.stm_info = None

    def fit(self, X, y):
        """
        Parameters
        ----------
        X: np.ndarray
            Input data of shape (n_samples, n_chs, n_time)
        y: np.ndarray
            Sequence labels of X (not target/non-target labels),
             must be 1 (=sequence1) or 2 (=sequence2).
        """
        X1 = X[np.where(y == 1)]
        X2 = X[np.where(y == 2)]

        X = X.reshape(X.shape[0], -1)
        X1 = X1.reshape(X1.shape[0], -1)
        X2 = X2.reshape(X2.shape[0], -1)

        # Compute global covariance matrix
        C_cov, gamma = shrinkage(X.T)
        nt = calc_n_times(C_cov.shape[0], self.n_channels, self.n_times)
        stm = SpatioTemporalMatrix(C_cov, n_chans=self.n_channels, n_times=nt)

        if self.toeplitz_time:
            stm.force_toeplitz_offdiagonals()
        if self.taper_time is not None:
            stm.taper_offdiagonals(self.taper_time)
        stm.swap_primeness()
        if self.toeplitz_spatial:
            stm.force_toeplitz_offdiagonals(raise_spatial=False)
        if self.taper_spatial is not None:
            stm.taper_offdiagonals(self.taper_spatial)
        stm.swap_primeness()

        self.stored_stm = stm
        C_cov = stm.mat

        # Compute sequence-wise means
        average_O1 = np.mean(X1, axis=0)
        average_O2 = np.mean(X2, axis=0)

        mu_T = self.pinv_ratio_matrix[0, 0] * average_O1 + self.pinv_ratio_matrix[0, 1] * average_O2
        mu_NT = (
            self.pinv_ratio_matrix[1, 0] * average_O1 + self.pinv_ratio_matrix[1, 1] * average_O2
        )

        # use reconstructed means to compute w and b
        C_diff = mu_T - mu_NT
        C_mean = 0.5 * (mu_T + mu_NT)

        self.stored_cl_mean = np.vstack([mu_NT, mu_T]).T

        if self.use_fortran_solver:
            if not self.toeplitz_time:
                raise ValueError("Cannot use fortran solver without block-Toeplitz structure")
            C_w = fortran_block_levinson(C_cov, C_diff, nch=self.n_channels, ntim=self.n_times)
        else:
            C_w = np.linalg.solve(C_cov, C_diff)
        C_w = 2 * C_w / np.dot(C_w.T, C_diff)
        C_b = np.dot(-C_w.T, C_mean)

        self.w = C_w
        self.b = C_b

        self.mu_T = mu_T
        self.mu_NT = mu_NT

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = X.reshape(X.shape[0], -1)
        return np.dot(X, self.w) + self.b

    def predict(self, X: np.ndarray):
        return self.decision_function(X) > 0
