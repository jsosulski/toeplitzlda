from pathlib import Path

try:
    import mne
except ImportError:
    print(
        "You need to install toeplitzlda with neuro extras to run examples "
        "with real EEG data, i.e. pip install toeplitzlda[neuro]"
    )
    exit(1)

from blockmatrix import linear_taper
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline

from toeplitzlda.classification import (
    EpochsVectorizer,
    ShrinkageLinearDiscriminantAnalysis,
    ToeplitzLDA,
)
from toeplitzlda.classification.covariance import ToepTapLW
from toeplitzlda.usup_replay.llp import LearningFromLabelProportions
from toeplitzlda.usup_replay.visual_speller import (
    VisualMatrixSpellerLLPDataset,
    seq_labels_from_epoch,
)


mne.set_log_level("ERROR")

# Load only first block of some subject
sub = 6
block = 1
dataset = VisualMatrixSpellerLLPDataset()
dataset.subject_list = [sub]
tmp_f = Path("/tmp/cache")
tmp_f.mkdir(exist_ok=True)
cache_f = tmp_f / f"epos_s{sub}_b{block}-epo.fif"
# if not cache_f.exists():
#     print("Preprocessing Data.")
epochs, _ = dataset.load_epochs(block_nrs=[block], fband=[0.5, 8], sampling_rate=40)
print(f"before drop: {len(epochs)}")
print(f"after drop: {len(epochs)}")
nch = len(epochs.ch_names)
epos_per_letter = 68
feature_ival = [0.05, 0.7]
e_train = epochs[0 : (epos_per_letter * 7)]
llp_train, labels_train = seq_labels_from_epoch(e_train)
e_test = epochs[(epos_per_letter * 7) :]
_, labels_test = seq_labels_from_epoch(e_test)
clfs = dict(sup=dict(), usup=dict())

# Straightforward use toeplitz lda
clfs["sup"]["toeplitz_lda"] = make_pipeline(
    EpochsVectorizer(
        select_ival=feature_ival,
    ),
    ToeplitzLDA(n_channels=nch),
)

# Use provided covariance estimator to improve sklearn lda
clfs["sup"]["skl_lda_toep"] = make_pipeline(
    EpochsVectorizer(
        select_ival=feature_ival,
    ),
    LinearDiscriminantAnalysis(solver="lsqr", covariance_estimator=ToepTapLW(n_channels=nch)),
)

clfs["sup"]["skl_lda_kron"] = make_pipeline(
    EpochsVectorizer(
        select_ival=feature_ival,
    ),
    LinearDiscriminantAnalysis(
        solver="lsqr", covariance_estimator=ToepTapLW(n_channels=nch, force_kronecker=True)
    ),
)

# Compare with plain sklearn
clfs["sup"]["skl_slda"] = make_pipeline(
    EpochsVectorizer(
        select_ival=feature_ival,
    ),
    # Use Ledoit-Wolf Shrinkage
    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
)

# # Currently not yet implemented in upstream sklearn
# clfs["sup"]["skl_lda_toep_pooled"] = make_pipeline(
#     EpochsVectorizer(
#         select_ival=feature_ival,
#     ),
#     LinearDiscriminantAnalysis(
#         solver="lsqr",
#         covariance_estimator=ToepTapLW(n_channels=nch),
#         pooled_cov=True,
#     ),
# )

# Unsupervised classifiers, i.e., make use of label proportions, not true labels
# LLP with proposed block-Toeplitz structure
clfs["usup"]["llp_toep"] = make_pipeline(
    EpochsVectorizer(
        select_ival=feature_ival,
    ),
    LearningFromLabelProportions(
        toeplitz_time=True, taper_time=linear_taper, n_times="infer", n_channels=nch
    ),
)

# LLP with plain sLDA
clfs["usup"]["llp_slda"] = make_pipeline(
    EpochsVectorizer(
        select_ival=feature_ival,
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
        y_df = clfs[cat][k].decision_function(e_test)
        roc_aucs[cat][k] = roc_auc_score(labels_test, y_df)
        y_pred = clfs[cat][k].predict(e_test)
        bal_accs[cat][k] = balanced_accuracy_score(labels_test, y_pred)

print("\nScores\n======")
print("\n AUCs\n")
for cat in clfs:
    for k in clfs[cat]:
        print(f" {cat}.{k}: {roc_aucs[cat][k]:.4f}")

print("\n Balanced acc\n")
for cat in clfs:
    for k in clfs[cat]:
        print(f" {cat}.{k}: {bal_accs[cat][k]:.4f}")
