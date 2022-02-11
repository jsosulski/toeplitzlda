from typing import Iterable

import numpy as np
from blockmatrix import linear_taper, SpatioTemporalMatrix
from sklearn.datasets import make_spd_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from toeplitzlda.classification import (
    ShrinkageLinearDiscriminantAnalysis,
    ToeplitzLDA,
)
from toeplitzlda.classification.covariance import ToepTapLW

np.random.seed(123)


def generate_stationary_data(
    samples_per_class: Iterable,
    class_means=None,
    noise_cov=None,
    n_channels=16,
    n_times=60,
):
    if noise_cov is None:
        # how much stronger is the noise compared to the signal? (class means)
        noise_factor = 15
        noise_cov = noise_factor ** 2 * make_spd_matrix(n_channels * n_times)
        # Force generated matrix to be stationary
        # TODO how to generate "mostly" stationary data?
        stm = SpatioTemporalMatrix(noise_cov, n_chans=n_channels, n_times=n_times)
        stm.force_toeplitz_offdiagonals()
        stm.taper_offdiagonals(linear_taper)
        noise_cov = stm.mat
    X = []
    y = []
    clmeans = []
    for i, samples in enumerate(samples_per_class):
        if class_means is None:
            clmean = np.random.normal(0, 1, (n_channels * n_times,))
        else:
            clmean = class_means[:, i]
        clmeans.append(clmean)
        X.append(np.random.multivariate_normal(clmean, noise_cov, size=samples).T)
        # fmt: off
        y.append(i * np.ones(samples,).T)
        # fmt: on
    X = np.hstack(X).T
    y = np.hstack(y)
    return X, y, np.vstack(clmeans).T, noise_cov


nch = 16
ntimes = 30
samples_per_class = np.array([250, 50])

X_train, y_train, means, noise = generate_stationary_data(
    samples_per_class, n_channels=nch, n_times=ntimes
)
X_test, y_test, _, _ = generate_stationary_data(
    10 * samples_per_class,
    class_means=means,
    noise_cov=noise,
    n_channels=nch,
    n_times=ntimes,
)

# %%
clfs = dict(sup=dict(), usup=dict())

# Straightforward use toeplitz lda
clfs["sup"]["toeplitz_lda"] = make_pipeline(
    ToeplitzLDA(n_channels=nch),
)

# Straightforward use toeplitz lda with fortran solver
try:
    import toeplitz

    clfs["sup"]["toeplitz_lda_fortran"] = make_pipeline(
        ToeplitzLDA(n_channels=nch, use_fortran_solver=True),
    )
except:
    print("Skipping LDA with fortran solver, as it is not installed.")

# Can also be used manually using our SLDA implementation
clfs["sup"]["our_slda_with_toeplitz"] = make_pipeline(
    ShrinkageLinearDiscriminantAnalysis(
        n_channels=nch, enforce_toeplitz=True, tapering=linear_taper
    ),
)

# Normal SLDA our implementation
clfs["sup"]["our_slda"] = make_pipeline(
    ShrinkageLinearDiscriminantAnalysis(n_channels=nch),
)

# Use provided covariance estimator to improve sklearn lda
clfs["sup"]["skl_lda_toep"] = make_pipeline(
    LinearDiscriminantAnalysis(
        solver="lsqr", covariance_estimator=ToepTapLW(n_times="infer", n_channels=nch)
    ),
)

# Compare with plain sklearn
clfs["sup"]["skl_slda"] = make_pipeline(
    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
)

# # Currently not yet implemented in upstream sklearn
# clfs["sup"]["skl_lda_toep_pooled"] = make_pipeline(
#     EpochsVectorizer(
#         select_ival=[0.05, 0.7],
#     ),
#     LinearDiscriminantAnalysis(
#         solver="lsqr",
#         covariance_estimator=ToepTapLW(n_times="infer", n_channels=nch),
#         pooled_cov=True,
#     ),
# )

print("Fitting supervised...")
for k in clfs["sup"]:
    print(f" {k}")
    clfs["sup"][k].fit(X_train, y=y_train)

print("Fitting times (NOTE: numpy probably faster due to multi-threading):")
for cat in clfs:
    for k in clfs[cat]:
        c = clfs[cat][k][-1]
        if hasattr(c, "fit_time_"):
            print(f" {cat}.{k}: {c.fit_time_*1000} ms")
        else:
            print(f" {cat}.{k}: NOT MEASURED")

print("Predicting...")
roc_aucs = dict()
bal_accs = dict()
for cat in clfs:
    roc_aucs[cat] = dict()
    bal_accs[cat] = dict()
    for k in clfs[cat]:
        print(f" {cat}.{k}")
        y_df = clfs[cat][k].decision_function(X_test)
        if y_df.ndim > 1:
            ohe = OneHotEncoder()
            y_test_auc = ohe.fit_transform(y_test[:, np.newaxis]).toarray()
        else:
            y_test_auc = y_test
        roc_aucs[cat][k] = roc_auc_score(y_test_auc, y_df, multi_class="ovr")
        y_pred = clfs[cat][k].predict(X_test)
        bal_accs[cat][k] = balanced_accuracy_score(y_test, y_pred)

print("\nScores\n======")
print("\n AUCs\n")
for cat in clfs:
    for k in clfs[cat]:
        print(f" {cat}.{k}: {roc_aucs[cat][k]:.4f}")

print("\n Balanced acc\n")
for cat in clfs:
    for k in clfs[cat]:
        print(f" {cat}.{k}: {bal_accs[cat][k]:.4f}")
