import mne
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline

from toeplitzlda.classification import (
    ShrinkageLinearDiscriminantAnalysis,
    ToeplitzLDA,
    EpochsVectorizer,
)
from blockmatrix import linear_taper

from pathlib import Path

from toeplitzlda.classification.covariance import ToepTapLW
from toeplitzlda.usup_replay.llp import LearningFromLabelProportions
from toeplitzlda.usup_replay.visual_speller import (
    VisualMatrixSpellerLLPDataset,
    seq_labels_from_epoch,
)

mne.set_log_level("ERROR")

# Load only first block of first subject
sub = 1
block = 1
dataset = VisualMatrixSpellerLLPDataset()
dataset.subject_list = [sub]
tmp_f = Path("/tmp/cache")
tmp_f.mkdir(exist_ok=True)
cache_f = tmp_f / f"epos_s{sub}_b{block}-epo.fif"
if not cache_f.exists():
    print("Preprocessing Data.")
    epochs, _ = dataset.load_epochs(
        block_nrs=[block], fband=[0.5, 16], sampling_rate=50
    )
    epochs.save(cache_f)
else:
    print("Loading cached")
    epochs = mne.read_epochs(cache_f)
nch = len(epochs.ch_names)
# %%
# Train on first letter
e_train = epochs[0 : (68 * 1)]
llp_train, labels_train = seq_labels_from_epoch(e_train)
# Test on letters 31-63
e_test = epochs[(68 * 30) :]
_, labels_test = seq_labels_from_epoch(e_test)
clfs = dict(sup=dict(), usup=dict())

# Straightforward use toeplitz lda
clfs["sup"]["toeplitz_lda"] = make_pipeline(
    EpochsVectorizer(
        select_ival=[0.05, 0.7],
    ),
    ToeplitzLDA(n_channels=nch),
)

# Straightforward use toeplitz lda with fortran solver
clfs["sup"]["toeplitz_lda_fortran"] = make_pipeline(
    EpochsVectorizer(
        select_ival=[0.05, 0.7],
    ),
    ToeplitzLDA(n_channels=nch, use_fortran_solver=True),
)

# Can also be used manually using our SLDA implementation
clfs["sup"]["slda_with_toeplitz"] = make_pipeline(
    EpochsVectorizer(
        select_ival=[0.05, 0.7],
    ),
    ShrinkageLinearDiscriminantAnalysis(
        n_channels=nch, enforce_toeplitz=True, tapering=linear_taper
    ),
)

# Normal SLDA our implementation
clfs["sup"]["our_s_lda"] = make_pipeline(
    EpochsVectorizer(
        select_ival=[0.05, 0.7],
    ),
    ShrinkageLinearDiscriminantAnalysis(n_channels=nch),
)

# Compare with plain sklearn
clfs["sup"]["skl_lda_shrinkage_auto"] = make_pipeline(
    EpochsVectorizer(
        select_ival=[0.05, 0.7],
    ),
    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
)

# Use provided covariance estimator to improve sklearn lda
clfs["sup"]["skl_lda_toep"] = make_pipeline(
    EpochsVectorizer(
        select_ival=[0.05, 0.7],
    ),
    LinearDiscriminantAnalysis(
        solver="lsqr", covariance_estimator=ToepTapLW(n_times="infer", n_channels=nch)
    ),
)

# Currently not yet implemented in upstream sklearn
clfs["sup"]["skl_lda_toep_pooled"] = make_pipeline(
    EpochsVectorizer(
        select_ival=[0.05, 0.7],
    ),
    LinearDiscriminantAnalysis(
        solver="lsqr",
        covariance_estimator=ToepTapLW(n_times="infer", n_channels=nch),
        pooled_cov=True,
    ),
)

# Unsupervised classifiers, i.e., make use of label proportions, not true labels
# LLP with proposed block-Toeplitz structure
clfs["usup"]["llp_toep"] = make_pipeline(
    EpochsVectorizer(
        select_ival=[0.05, 0.7],
    ),
    LearningFromLabelProportions(
        toeplitz_time=True, taper_time=linear_taper, n_times="infer", n_channels=nch
    ),
)

# LLP with plain sLDA
clfs["usup"]["llp_slda"] = make_pipeline(
    EpochsVectorizer(
        select_ival=[0.05, 0.7],
    ),
    LearningFromLabelProportions(n_times="infer", n_channels=nch),
)

print("Fitting supervised...")
for k in clfs["sup"]:
    print(f" {k}")
    clfs["sup"][k].fit(e_train, y=labels_train)

print("Fitting unsupervised...")
for k in clfs["usup"]:
    print(f" {k}")
    clfs["usup"][k].fit(e_train, y=llp_train)

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
        roc_aucs[cat][k] = roc_auc_score(
            labels_test, clfs[cat][k].decision_function(e_test)
        )
        bal_accs[cat][k] = balanced_accuracy_score(
            labels_test, clfs[cat][k].predict(e_test)
        )

print("\nScores\n======")
print("\n AUCs\n")
for cat in clfs:
    for k in clfs[cat]:
        print(f" {cat}.{k}: {roc_aucs[cat][k]:.4f}")

print(
    "\n!!! NOTE: we do not use skl method of choosing maximal class-dimension but instead "
    "traditional bias selection. This leads to bad balanced accuracy with toeplitz classifiers !!!"
)
print("\n Balanced acc\n")
for cat in clfs:
    for k in clfs[cat]:
        print(f" {cat}.{k}: {bal_accs[cat][k]:.4f}")
