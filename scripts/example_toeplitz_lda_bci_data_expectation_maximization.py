from pathlib import Path

from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky

from toeplitzlda.classification.unsupervised import ExternalLDA

import numpy as np
from blockmatrix import SpatioTemporalMatrix

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
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

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
    NUMBER_TO_LETTER_DICT,
    VisualMatrixSpellerMixDataset,
)


mne.set_log_level("ERROR")

sub = 3
block = 1
dataset = VisualMatrixSpellerMixDataset()
dataset.subject_list = [sub]
tmp_f = Path("/tmp/cache")
tmp_f.mkdir(exist_ok=True)
cache_f = tmp_f / f"epos_s{sub}_b{block}-epo.fif"
if not cache_f.exists():
    print("Preprocessing Data.")
    epochs, _ = dataset.load_epochs(block_nrs=[block], fband=[0.5, 8], sampling_rate=40)
    epochs.save(cache_f)
else:
    print("Loading cached")
    epochs = mne.read_epochs(cache_f)
nch = len(epochs.ch_names)
epos_per_letter = 68
# %%
e_train = epochs[0 : (epos_per_letter * 1)]
llp_train, labels_train = seq_labels_from_epoch(e_train)
# Test on letters 31-63
e_test = epochs[(epos_per_letter * 30) :]
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
try:
    import toeplitz

    clfs["sup"]["toeplitz_lda_fortran"] = make_pipeline(
        EpochsVectorizer(
            select_ival=[0.05, 0.7],
        ),
        ToeplitzLDA(n_channels=nch, use_fortran_solver=True),
    )
except:
    print("Skipping LDA with fortran solver, as it is not installed.")

# Can also be used manually using our SLDA implementation
clfs["sup"]["our_slda_with_toeplitz"] = make_pipeline(
    EpochsVectorizer(
        select_ival=[0.05, 0.7],
    ),
    ShrinkageLinearDiscriminantAnalysis(
        n_channels=nch, enforce_toeplitz=True, tapering=linear_taper
    ),
)

# Normal SLDA our implementation
clfs["sup"]["our_slda"] = make_pipeline(
    EpochsVectorizer(
        select_ival=[0.05, 0.7],
    ),
    ShrinkageLinearDiscriminantAnalysis(n_channels=nch),
)

# Use provided covariance estimator to improve sklearn lda
clfs["sup"]["skl_lda_toep"] = make_pipeline(
    EpochsVectorizer(
        select_ival=[0.05, 0.7],
    ),
    LinearDiscriminantAnalysis(
        solver="lsqr", covariance_estimator=ToepTapLW(n_channels=nch)
    ),
)

# Compare with plain sklearn
clfs["sup"]["skl_slda"] = make_pipeline(
    EpochsVectorizer(
        select_ival=[0.05, 0.7],
    ),
    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
)

# # Currently not yet implemented in upstream sklearn
# clfs["sup"]["skl_lda_toep_pooled"] = make_pipeline(
#     EpochsVectorizer(
#         select_ival=[0.05, 0.7],
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

# %%

# clmean_true = clfs["sup"]["our_slda"][-1].stored_cl_mean
# clmean_llp = clfs["usup"]["llp_slda"][-1].stored_cl_mean
#
# pip = make_pipeline(
#     EpochsVectorizer(
#         select_ival=[0.05, 0.7],
#     ),
#     GaussianMixture(weights_init=[52/68, 16/68], n_components=2, covariance_type="tied",
#                     means_init=np.zeros_like(clmean_llp.T)),
# )

# pip.fit(e_train)

# clmean_gmm = pip[-1].means_.T
# stm_true = clfs["sup"]["our_slda"][-1].stored_stm
# stm_glob = clfs["usup"]["llp_slda"][-1].stored_stm
# stm_gmm = SpatioTemporalMatrix(pip[-1].covariances_, 31, 27)
# # stm_gmm.taper_offdiagonals()
# # stm_gmm.force_toeplitz_offdiagonals()
#
#
# error_llp = np.sum((clmean_true - clmean_llp)**2)
# error_gmm = np.sum((clmean_true - clmean_gmm)**2)
#
# print(f"{error_llp=}\n{error_gmm=}")
#
# elda = ExternalLDA(means=clmean_gmm, cov=stm_gmm.mat)
# elda.calc()
#
# y_df = elda.decision_function(evec.transform(e_test))
# roc_auc = roc_auc_score(labels_test, y_df)
# y_pred = elda.predict(evec.transform(e_test))
# bal_acc = balanced_accuracy_score(labels_test, y_pred)
#
# print(f"{roc_auc=}")
# %% Poor man's expectation maximization

evec = EpochsVectorizer(select_ival=[0.05, 0.7])
gmm = GaussianMixture(
    weights_init=[52 / 68, 16 / 68],
    n_components=2,
    covariance_type="tied",
    means_init=np.zeros((2, evec.transform(e_train[0]).shape[1])),
)

spellable = [s for s in NUMBER_TO_LETTER_DICT.values() if s != "#"]


def idx_except(e, letter):
    idx = []
    for i in range(len(e)):
        if letter not in e[i].event_id:
            idx.append(i)
    return idx


def true_letter(e):
    evs = list(e["Target"].event_id.keys())
    t_letters = [ev.split("/")[3:] for ev in evs]
    t_let = t_letters[0]
    for tl in t_letters[1:]:
        for l in t_let:
            if l not in tl:
                t_let.remove(l)
        if len(t_let) == 1:
            break
    return t_let[0]


tmp_f = Path("/tmp/cache")
mne.set_log_level("ERROR")
tmp_f.mkdir(exist_ok=True)
dcode = "Mix"
# TODO FIX THIS SCRIPT
import pandas as pd

df = pd.DataFrame()
entry = dict()
for use_toep in [True, False]:
    for use_agg_mean in [True, False]:
        entry["toep"] = use_toep
        entry["agg_mean"] = use_agg_mean
        for sub in range(1, 13):
            entry["sub"] = sub
            dataset = (
                VisualMatrixSpellerLLPDataset()
                if dcode == "LLP"
                else VisualMatrixSpellerMixDataset()
            )
            dataset.subject_list = [sub]
            for block in range(1, 4):
                agg_mean = np.zeros((2, 837))
                entry["block"] = block
                cache_f = tmp_f / f"{dcode}_s{sub}_b{block}-epo.fif"
                print(f"Subject {sub}, block {block}")
                sentence = ""
                true_sentence = ""
                if not cache_f.exists():
                    print("Preprocessing Data.")
                    epochs, _ = dataset.load_epochs(
                        block_nrs=[block], fband=[0.5, 8], sampling_rate=40
                    )
                    epochs.save(cache_f)
                else:
                    print("Loading cached")
                    epochs = mne.read_epochs(cache_f)
                    epochs.reset_drop_log_selection()
                for let_i in range(35):
                    entry["num_let"] = let_i + 1
                    e_train = epochs[f"Letter_{let_i+1}"]
                    if len(e_train) != 68:
                        print(f"WARNING, EPO NOT FULL ({let_i+1}: {len(e_train)})")
                    true_sentence += true_letter(e_train)
                    e_train_cum = epochs[0 : np.max(e_train.selection)]
                    cov_model = (
                        ToepTapLW(n_channels=31)
                        if use_toep
                        else ToepTapLW(n_channels=31, only_lw=True)
                    )
                    # cov_model = ToepTapLW(n_channels=31, only_lw=True)
                    cov_model.fit(evec.transform(e_train))
                    gmm.covariances_ = cov_model.covariance_
                    gmm.precisions_cholesky_ = _compute_precision_cholesky(
                        gmm.covariances_, "tied"
                    )

                    e_train.reset_drop_log_selection()
                    X = evec.transform(e_train)

                    letter_likelihoods = np.empty(len(spellable)) + np.nan
                    prev_logprob_max = -np.inf
                    best_means = None
                    for i, s in enumerate(spellable):
                        try:
                            t_epo = e_train[s]
                        except:
                            print(f"Could not select for symbol {s} -> does not exist")
                            continue
                        nt_idx = [
                            i for i in e_train.selection if i not in t_epo.selection
                        ]
                        nt_epo = e_train[nt_idx]
                        non_target = evec.transform(nt_epo).mean(0)[None, :]
                        target = evec.transform(t_epo).mean(0)[None, :]
                        trial_means = np.vstack([non_target, target])
                        clmeans = (agg_mean * let_i + trial_means) / (let_i + 1)
                        gmm.means_ = trial_means if not use_agg_mean else clmeans
                        logprob = gmm._estimate_log_prob(X)
                        logprob_diff = logprob[:, 1] - logprob[:, 0]
                        top_16 = logprob_diff[np.argsort(logprob_diff)[-16:]]
                        logprob_max = np.sum(top_16)
                        if logprob_max > prev_logprob_max:
                            prev_logprob_max = logprob_max
                            # best_means = clmeans
                            best_means = trial_means
                        letter_likelihoods[i] = logprob_max
                        # print(f"{s}: {logprob_max}")
                    most_likely_idx = np.argmax(letter_likelihoods)
                    agg_mean = (agg_mean * let_i + best_means) / (let_i + 1)
                    sentence += spellable[most_likely_idx]
                    entry["correct"] = (
                        spellable[most_likely_idx] == true_sentence[let_i]
                    )
                    entry["decoded"] = spellable[most_likely_idx]
                    entry["true"] = true_sentence[let_i]
                    df = df.append(entry, ignore_index=True)

                correct_letters = [d == t for d, t in zip(sentence, true_sentence)]
                print(
                    f" Decoded: '{sentence}' ({np.sum(correct_letters)}/{len(correct_letters)})"
                )
                print(f" True:    '{true_sentence}'")

            df["cumulated_cov"] = False
            df.to_csv(f"/home/jan/results_em_no_cumu.csv")

            # gmm.mea

            # %%
            # sent = dict()
            # sent['slda'] = "FRANZL JAGT IA KOMPKET. !EAWAHRLOSTEN TJAI LUIROGURCH FRHIBDTG."
            # sent['slda_cum'] = "FRANZQ JAGT DR KOMPPEST VJRWAHRLMSNEN TAXI LUER.JULCH FXEIBURG."
            # sent['toep_lda'] = "FRANZY JAGT IM KOMPLETT VERWAHRLOSTEN TAXI QUER<DURCH FREIBURG."
            # sent['toep_lda_cum'] = "FRANZY JAGT IM KOMPLETT VERWAHRLMSTEN TAXI QUER<DURCH FREIBURG."
            #
            # for s in sent:
            #     print(f"{s:>20}: {sent[s]}")
