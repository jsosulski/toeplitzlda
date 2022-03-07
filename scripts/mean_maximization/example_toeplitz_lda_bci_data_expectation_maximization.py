from copy import deepcopy
import time
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky

import numpy as np
from blockmatrix import SpatioTemporalData

try:
    import mne
except ImportError:
    print(
        "You need to install toeplitzlda with neuro extras to run examples "
        "with real EEG data, i.e. pip install toeplitzlda[neuro]"
    )
    exit(1)

from sklearn.mixture import GaussianMixture
import pandas as pd

from toeplitzlda.classification import EpochsVectorizer
from toeplitzlda.classification.covariance import ToepTapLW
from toeplitzlda.usup_replay.visual_speller import (
    VisualMatrixSpellerLLPDataset,
    NUMBER_TO_LETTER_DICT,
    VisualMatrixSpellerMixDataset,
)

mne.set_log_level("ERROR")
# %%
""" Poor man's expectation maximization

This approach does not require *any* label/proportion information. Required information:
- Which symbols can be spelled?
- For each epoch: which symbols have been highlighted?

Assumptions:
- Similar to LDA: Both classes have the same covariance
- Underlying are only 2 different normal distributions, i.e. Target/NonTarget
- Each letter highlighted the same number of times

Notable non-assumptions:
- Linear separation (? not really or? Likelihood threshold is linear?...)
"""


def get_initialized_gmm(feature_dim=None):
    gmm_model = GaussianMixture(
        weights_init=[52 / 68, 16 / 68],
        n_components=2,
        covariance_type="tied",
        means_init=np.zeros((2, feature_dim)),
    )
    return gmm_model


def get_true_letter(e):
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


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


# Extending stderr of variance to covariance tapering yields equal performance on small subset
def stderr_of_variance(n):
    return np.sqrt(2 / (n - 1))


def stderr_taper_factory(n_epos):
    def stderr_taper(d, dmax):
        return stderr_of_variance(n_epos * dmax) / stderr_of_variance(n_epos * (dmax - d))

    return stderr_taper


# Setup/initialize basic variables
evec = EpochsVectorizer(select_ival=[0.05, 0.7])
spellable = [s for s in NUMBER_TO_LETTER_DICT.values() if s != "#"]
n_times = None  # This is set automatically, once data is loaded
n_channels = None  # This is set automatically, once data is loaded
tmp_f = Path("/tmp/cache")
tmp_f.mkdir(exist_ok=True)
df = pd.DataFrame()
row = dict()
rows = list()
blocks = [1, 2, 3]
np.random.seed(123)

# Parameters to change
dcode = "LLP"
plot_erp_after_every_letter = False
debug_prints = True
# Idea: instead of using Sigma_T use (Sigma_T - (Mu^T * Mu)) = (Sigma_T - Sigma_B) = Sigma_W ?
# This should be a better estimate of Within-Scatter. But: not necesarrily full rank!
# FIXME: Not implemented yet.
cov_clmean_correction = True
# Sanity checks
sanity_check_use_random_mean_for_first_n_letters = 0
sanity_check_letter_for_data_based_random_means = "G"  # only relevant if previous > 0
sanity_check_random_letter_order = False

if sanity_check_use_random_mean_for_first_n_letters > 0:
    spellable.remove(sanity_check_letter_for_data_based_random_means)
    spellable.insert(0, sanity_check_letter_for_data_based_random_means)

# Dataset specifics
if dcode == "Mix":
    n_letters = 35
    # Note that Letters 36-70 in MIX are not used, as free spelling was done.
    # Ground truth for these spelled letters is not conclusively available. (?)
    true_sentence = "FRANZY JAGT IM TAXI QUER DURCH DAS "
    subjects = list(range(1, 13))
    dataset_class = VisualMatrixSpellerMixDataset
elif dcode == "LLP":
    n_letters = 63
    true_sentence = "FRANZY JAGT IM KOMPLETT VERWAHRLOSTEN TAXI QUER DURCH FREIBURG."
    subjects = list(range(1, 14))
    dataset_class = VisualMatrixSpellerLLPDataset
else:
    raise ValueError(f"Dataset code: {dcode} is not valid")

# DEBUG OVERWRITE
n_letters = 14
subjects = [11]
blocks = [3]

parameter_combinations = [[True, True], [False, True], [True, False], [False, False]]

for hyp_i, (use_toeplitz_covariance, use_aggregated_mean) in enumerate(parameter_combinations):
    row["toeplitz_covariance"] = use_toeplitz_covariance
    row["aggregated_mean"] = use_aggregated_mean
    print(f"Evaluating setting: {dcode=} {use_toeplitz_covariance=} {use_aggregated_mean=}")
    for sub in subjects:
        row["subject"] = sub
        dataset = dataset_class()
        dataset.subject_list = [sub]
        for block in blocks:
            row["block"] = block
            cache_f = tmp_f / f"{dcode}_s{sub}_b{block}-epo.fif"
            print(f"Subject {sub}, block {block}")
            try:
                if not cache_f.exists():
                    # Load raw data and preprocess
                    print("Preprocessing Data.")
                    all_epochs, _ = dataset.load_epochs(
                        block_nrs=[block], fband=[0.5, 8], sampling_rate=40
                    )
                    all_epochs.save(cache_f)
                else:
                    # Use already preprocessed data
                    print("Loading cached")
                    all_epochs = mne.read_epochs(cache_f)
            except:
                print("Could not load epochs. Skipping this block")
                continue
            print("Starting letter processing")
            all_epochs.reset_drop_log_selection()
            # Need to obtain feature dimensions from loaded data
            evec.transform(all_epochs)
            n_times = len(evec.times_)
            n_channels = len(all_epochs.ch_names)
            # Reset variables
            decoded_sentence = ""
            aggregated_clmeans = np.zeros((2, n_times * n_channels))
            for let_i in range(n_letters):
                letter_evaluation_start_time = time.time()
                if not debug_prints:
                    print("!" if (let_i + 1) % 10 == 0 else ".", end="")
                else:
                    print(
                        f" Current letter: {let_i+1} ({dcode=} {use_toeplitz_covariance=} {use_aggregated_mean=})"
                    )
                row["nth_letter"] = let_i + 1
                # Select epochs only from current letter/trial
                epo_current_trial = all_epochs[f"Letter_{let_i + 1}"]
                if len(epo_current_trial) < 68:
                    print(
                        f"WARNING, EPO NOT FULL (Letter {let_i+1} has only {len(epo_current_trial)}/68)"
                    )
                # Has never occured
                elif len(epo_current_trial) > 68:
                    print("!!!!!!!! More epochs than allowed. Aborting block. !!!!!!!!")
                    break

                # Select all epochs from current and past letters/trials
                selected_letters = [f"Letter_{li + 1}" for li in range(let_i + 1)]
                epo_cumulated_trials = all_epochs[selected_letters]

                if debug_prints:
                    print(f"  Fitting covariance on {len(epo_cumulated_trials)} epochs")
                cov_model = ToepTapLW(n_channels=n_channels, only_lw=(not use_toeplitz_covariance))
                cov_model.fit(evec.transform(epo_cumulated_trials))
                # Store total covariance/scatter. Needed for implementation of only within cov
                # But that has rank issues for now.
                total_cov = cov_model.covariance_
                total_prec = np.linalg.pinv(total_cov)

                # GMM is only used for likelihood helper functions
                # Actual EM is NOT run (fit call)
                gmm = get_initialized_gmm(feature_dim=n_times * n_channels)
                gmm.covariances_ = total_cov
                cholesky_time = time.time()
                gmm.precisions_cholesky_ = _compute_precision_cholesky(gmm.covariances_, "tied")
                cholesky_duration = time.time() - cholesky_time
                if debug_prints:
                    print(f"  Cholesky decomposition took {cholesky_duration:1.3f} seconds")

                # Set up variables
                # Reset epoch indexing to 0..67
                epo_current_trial.reset_drop_log_selection()
                spellable_likelihoods = np.empty(len(spellable)) + np.nan
                spellable_num_targets = np.empty(len(spellable))
                spellable_curr_sqdist = np.empty(len(spellable))
                spellable_agg_sqdist = np.empty(len(spellable))
                prev_logprob_score = -np.inf
                best_mean_estimate = None
                current_trial_X = evec.transform(epo_current_trial)

                # Only needed for sanity check
                random_means = None
                # Iterate over all spellable symbols and choose the one producing maximal
                # likelihoods given the data of the current trial
                # Potential speedups after first few letters. Only check first X most likely
                # target letters for new mean assignments etc.
                for si, s in enumerate(spellable):
                    try:
                        t_epo = epo_current_trial[s]
                    except:
                        print(f"Could not select for symbol {s} as it does not exist in epos")
                        continue
                    # Everything not in t_epo is nontarget
                    nt_idx = np.setxor1d(t_epo.selection, epo_current_trial.selection)
                    nt_epo = epo_current_trial[nt_idx]
                    non_target_mean = np.mean(evec.transform(nt_epo), axis=0)[np.newaxis, :]
                    target_mean = np.mean(evec.transform(t_epo), axis=0)[np.newaxis, :]
                    # SANITY CHECK: OVERRIDE MEAN VECTORS WITH RANDOM VECTORS
                    if let_i < sanity_check_use_random_mean_for_first_n_letters:
                        if random_means is None:
                            if sanity_check_letter_for_data_based_random_means is not None:
                                random_means = [
                                    non_target_mean.repeat(len(spellable), axis=0),
                                    target_mean.repeat(len(spellable), axis=0),
                                ]
                            # Gaussian random stuff
                            else:
                                random_non_target_mean = np.random.multivariate_normal(
                                    np.zeros_like(non_target_mean).squeeze() - 1,
                                    total_cov,
                                    size=len(spellable),
                                )
                                random_target_mean = np.random.multivariate_normal(
                                    np.zeros_like(target_mean).squeeze() + 1,
                                    total_cov,
                                    size=len(spellable),
                                )
                                random_means = [random_non_target_mean, random_target_mean]
                        non_target_mean = random_means[0][si, :]
                        target_mean = random_means[1][si, :]

                    current_trial_means = np.vstack([non_target_mean, target_mean])
                    aggregated_trial_means = (aggregated_clmeans * let_i + current_trial_means) / (
                        let_i + 1
                    )
                    if use_aggregated_mean:
                        gmm.means_ = aggregated_trial_means
                    else:
                        gmm.means_ = current_trial_means
                    # Potentially apply EM?
                    # gmm.fit(current_trial_X)

                    # For each datapoint, obtain likelihood of belonging to class 0 or 1
                    curr_mean_diff = (current_trial_means[1, :] - current_trial_means[0, :])[
                        :, None
                    ]
                    agg_mean_diff = (aggregated_trial_means[1, :] - aggregated_trial_means[0, :])[
                        :, None
                    ]
                    sq_curr_mean_dist = (curr_mean_diff.T @ total_prec @ curr_mean_diff)[0, 0]
                    sq_agg_mean_dist = (agg_mean_diff.T @ total_prec @ agg_mean_diff)[0, 0]

                    logprob_time = time.time()
                    logprob = gmm._estimate_log_prob(current_trial_X)
                    logprob_duration = time.time() - logprob_time
                    # if debug_prints:
                    #     print(f" Logprob calculation took {logprob_duration:1.3f} seconds")
                    logprob_diff = logprob[:, 1] - logprob[:, 0]
                    # Targets were flashed 16 times: get 16 highest likelihood diffs
                    # TODO: sometimes epochs are not completely loaded due to marker issues. Still take top16?
                    top_16 = np.sort(logprob_diff)[-16:]
                    logprob_score = np.sum(top_16)

                    # We want to update the aggregated means with the best estimate for the current
                    # Letter/Trial
                    if logprob_score > prev_logprob_score:
                        prev_logprob_score = logprob_score
                        best_mean_estimate = current_trial_means
                    spellable_likelihoods[si] = logprob_score
                    spellable_agg_sqdist[si] = sq_agg_mean_dist
                    spellable_curr_sqdist[si] = sq_curr_mean_dist
                    # print(f"{s}: {logprob_max}")
                # Update the current class mean aggregate with best trial estimate
                aggregated_clmeans = (aggregated_clmeans * let_i + best_mean_estimate) / (let_i + 1)
                # Most likely spellable produces the highest likelihood
                most_likely_idx = np.argmax(spellable_likelihoods)
                decoded_sentence += spellable[most_likely_idx]
                letter_evaluation_total_time = time.time() - letter_evaluation_start_time
                correct = spellable[most_likely_idx] == true_sentence[let_i]
                # Softmax as approximation of how sure we are, FOR NOW only used as metric.
                # But: this could be used to inform mean_aggregation, e.g.,
                # how to weight each trial mean estimate...
                softmax_best_2 = np.sort(softmax(spellable_likelihoods))[-2:]
                softmax_best_5 = np.sort(softmax(spellable_likelihoods))[::-1][0:5]
                softmax_best_5_str = " ".join(map("{:.4f}".format, softmax_best_5))
                spellable_order = [spellable[i] for i in np.argsort(spellable_likelihoods)][::-1]
                distance_to_true_letter = spellable_order.index(true_sentence[let_i])
                print(f" Decoded/Actual:  {spellable[most_likely_idx]}/{true_sentence[let_i]}")
                print(f" According to curr_dist:  {spellable[np.argmax(spellable_curr_sqdist)]}")
                print(f" According to agg_dist:  {spellable[np.argmax(spellable_agg_sqdist)]}")
                print(f"  Softmax top5:                {softmax_best_5_str}")
                print(f"  Best 5 letters (descending): {spellable_order[:5]}\n")
                softmax_logratio = np.log10(softmax_best_2[1]) - np.log10(softmax_best_2[0])
                row["correct"] = correct
                row["decoded_letter"] = spellable[most_likely_idx]
                row["true_letter"] = true_sentence[let_i]
                row["softmax_logratio_to_second"] = softmax_logratio
                row["evaluation_time"] = letter_evaluation_total_time
                row["num_epos"] = len(epo_current_trial)
                row["distance_to_true_letter"] = distance_to_true_letter
                rows.append(deepcopy(row))
                if debug_prints:
                    print(f" Letter took {letter_evaluation_total_time:.3f} seconds to evaluate")

                # Print erp mean estimates
                if plot_erp_after_every_letter and hyp_i == 0:
                    plot_me_channels = ["Pz", "O2"]
                    f, ax = plt.subplots(
                        2,
                        len(plot_me_channels),
                        figsize=(10, 3.5 * len(plot_me_channels)),
                        sharey="all",
                    )
                    for chi, ch in enumerate(plot_me_channels):
                        idx_ch = all_epochs.ch_names.index(ch)
                        for axi, (description, mean) in enumerate(
                            [
                                ("current trial", best_mean_estimate),
                                ("aggregated trials", aggregated_clmeans),
                            ]
                        ):
                            ch_mean = np.array(
                                [
                                    SpatioTemporalData.from_stacked_channel_prime(
                                        mean[0, :], n_chans=31
                                    ).get_channel_vec(idx_ch),
                                    SpatioTemporalData.from_stacked_channel_prime(
                                        mean[1, :], n_chans=31
                                    ).get_channel_vec(idx_ch),
                                ]
                            )
                            ax[axi, chi].plot(evec.times_, ch_mean.T)
                            ax[axi, chi].set_title(f"{ch} {description} mean")
                    f.suptitle(
                        f"Mean estimates after {let_i+1} letter(s), Sub: {sub}, Block: {block}"
                    )
                    plt.show()

            correct_letters = [d == t for d, t in zip(decoded_sentence, true_sentence)]
            print("\nResults====")
            print(
                f" Decoded: '{decoded_sentence}' ({np.sum(correct_letters)}/{len(correct_letters)})"
            )
            print(f" True:    '{true_sentence}'\n")

            # Add all rows to dataframe and reset rows buffer
            df = pd.concat([df, pd.DataFrame.from_records(rows)], ignore_index=True)
            rows = list()
            df.to_csv(f"/home/jan/results_em_{dcode.lower()}.csv")
