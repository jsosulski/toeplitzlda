import os
import warnings
from pathlib import Path


try:
    import mne
except ImportError:
    print(
        "You need to install toeplitzlda with neuro extras to run examples "
        "with real EEG data, i.e. pip install toeplitzlda[neuro]"
    )
    exit(1)
import numpy as np
import pandas as pd
from blockmatrix import linear_taper
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

from toeplitzlda.classification.toeplitzlda import EpochsVectorizer
from toeplitzlda.usup_replay.llp import LearningFromLabelProportions
from toeplitzlda.usup_replay.visual_speller import (
    VisualMatrixSpellerLLPDataset,
    VisualMatrixSpellerMixDataset,
    seq_labels_from_epoch,
)

mne.set_log_level("INFO")

np.seterr(divide="ignore")  # this does nothing for some reason
# We get a division by 0 warning for calculating theoretical classifier sureness,
# but we do not care as nan is a valid result
warnings.filterwarnings("ignore", category=RuntimeWarning)
# Pandas deprecation of append, but the proposed concat does not work...?
warnings.filterwarnings("ignore", category=FutureWarning)

EVENT_ID_TO_LETTER_DICT = {
    1: "A",
    2: "B",
    4: "C",
    6: "D",
    7: "E",
    9: "F",
    10: "G",
    11: "H",
    12: "I",
    13: "J",
    15: "K",
    16: "L",
    17: "M",
    19: "N",
    20: "O",
    21: "P",
    22: "Q",
    24: "R",
    25: "S",
    26: "T",
    27: "U",
    29: "V",
    30: "W",
    31: "X",
    32: "Y",
    34: "Z",
    35: " ",
    37: ".",
    39: ",",
    40: "!",
    41: "?",
    42: "<",
}
# This was the sentence each subject was instructed to spell in each run
GT_LLP = "FRANZY JAGT IM KOMPLETT VERWAHRLOSTEN TAXI QUER DURCH FREIBURG."
GT_MIX = "FRANZY JAGT IM TAXI QUER DURCH DAS "

spellable = list(EVENT_ID_TO_LETTER_DICT.values())


def predicted_letter(epo, pred, ax=None, gt=None, agg=np.mean):
    epo.selection = np.array(range(len(epo)))
    cum_scores = np.zeros(len(spellable))
    for i, l in enumerate(spellable):
        scores = pred[epo[l].selection]
        cum_scores[i] = agg(scores)
    high_score = np.argmax(cum_scores)
    if ax is not None:
        bars = ax.bar(spellable, cum_scores - np.mean(cum_scores))
        bars[high_score].set_facecolor("red")
        if gt is not None:
            gt_x = spellable.index(gt)
            ax.axvspan(gt_x - 0.5, gt_x + 0.5, alpha=0.3, color="green")

    return spellable[high_score], cum_scores


def predicted_letter_over_trial(epo, pred, ax=None, gt=None):
    epo.selection = np.array(range(len(epo)))
    if len(epo) < 15:
        print("WARNING: not enough epochs? Invalid epochs")
        return "#", np.nan, np.inf, np.nan, np.nan
    cum_score_history = np.zeros((len(spellable), len(epo)))
    highlight_history = np.zeros((len(spellable), len(epo)))
    letter_sequencing = np.zeros_like(highlight_history)

    for ei in range(len(epo)):
        evs = list(epo[ei].event_id.keys())[0].split("/")  # [3:]
        letters = [e for e in evs[3:] if e != "#"]
        indices = [spellable.index(l) for l in letters]
        letter_sequencing[indices, ei] = 1
        if ei > 0:
            highlight_history[:, ei] = highlight_history[:, ei - 1]
            cum_score_history[:, ei] = cum_score_history[:, ei - 1]
        highlight_history[indices, ei] += 1
        cum_score_history[indices, ei] += pred[ei]

    high_score_history = np.argmax(cum_score_history / highlight_history, axis=0)
    high_score_history[np.any(highlight_history == 0, axis=0)] = -1
    first_valid = np.where(high_score_history >= 0)[0][0]
    if ax is not None:
        alphas = [0.2, 1]
        for si, s in enumerate(spellable):
            a = alphas[1] if s == gt else alphas[0]

            ax.plot(
                range(len(epo)),
                cum_score_history[si, :] / highlight_history[si, :],
                label=s,
                alpha=a,
            )
        ax.axvline(first_valid, c="k", linestyle=":")
    pred_letter = spellable[high_score_history[-1]]

    correct = pred_letter == gt
    t_let_seq = letter_sequencing[spellable.index(gt)]
    if correct:
        earliest_correct = (
            1 + len(epo) - np.where(~(high_score_history == high_score_history[-1])[::-1])[0][0]
        )
        num_target_flashes = int(t_let_seq[:earliest_correct].sum())
    else:
        earliest_correct = np.nan
        num_target_flashes = np.nan
    mandatory_target_flashes = int(t_let_seq[:first_valid].sum())

    return (
        pred_letter,
        earliest_correct,
        first_valid,
        num_target_flashes,
        mandatory_target_flashes,
    )


# %%
# Choose dataset here
ds = "LLP"

if ds == "LLP":
    dataset = VisualMatrixSpellerLLPDataset()
    n_subs = 13
    GT = GT_LLP
elif ds == "Mix":
    dataset = VisualMatrixSpellerMixDataset()
    n_subs = 12
    GT = GT_MIX
else:
    raise ValueError("invalid ds code")

df = pd.DataFrame(
    columns=[
        "subject",
        "block",
        "clf",
        "nth_letter",
        "correct",
        "correct_sofar",
        "auc",
        "earliest_correct",
        "first_valid",
        "num_target_flashes",
        "mandatory_target_flashes",
    ]
)

num_letters = 63 if ds == "LLP" else 35  # Maximum number of letters is 63
letter_memory = np.inf  # Keep this many letters in memory for training
sr = 40
lowpass = 8

use_base = False
use_chdrop = False
use_jump = False
use_neutral_jump = False
use_each_best = False

enable_early_stopping_simulation = False
enable_calculate_postfix = False

if use_jump:
    ntimes = 6
elif sr == 20:
    ntimes = 14
elif sr == 100:
    ntimes = 66
elif sr == 8:
    ntimes = 6
elif sr == 40:
    ntimes = 27
elif sr == 200:
    ntimes = 131
elif sr == 1000:
    ntimes = 651
else:
    raise ValueError("invalid sampling rate")

epo_cache_root = Path("/") / "tmp" / ds
os.makedirs(epo_cache_root, exist_ok=True)

suffix = "_jump" if use_jump else ""
suffix += "_base" if use_base else ""
suffix += "_chdrop" if use_chdrop else ""

basedir = Path.home() / f"results_usup" / f"{lowpass}hz_lowpass_{sr}Hz_sr_{ntimes}tD{suffix}"
os.makedirs(
    basedir,
    exist_ok=True,
)


def get_llp_epochs(sub, block, use_cache=True):
    dataset.subject_list = [sub]
    epo_file = epo_cache_root / f"sub_{sub}_block_{block}-epo.fif"
    if use_cache and epo_file.is_file():
        print("WARNING: Loading cached data.")
        return mne.epochs.read_epochs(epo_file)
    else:
        print("Preprocessing data.")
        try:
            # ATTENTION: TODO Unify Lowpass handling -> also in select_ival and n_times param
            epochs = dataset.load_epochs(block_nrs=[block], fband=[0.5, lowpass], sampling_rate=sr)
            if use_base:
                epochs.apply_baseline((-0.2, 0))
            if use_chdrop:
                epochs.drop_channels("Fp1")
                epochs.drop_channels("Fp2")
            if use_cache:
                epochs.save(epo_file)
            return epochs
        except Exception as e:  # TODO specify what to catch
            raise e


for sub in range(1, 1 + n_subs):
    for block in range(1, 4):
        print(f"Subject {sub}, block {block}")
        print(f" Loading Data")
        epochs, _ = get_llp_epochs(sub, block, use_cache=False)
        if epochs is None:
            continue
        print(f" Starting evaluation")
        # These are the original time intervals used for averaging
        jm = [
            [0.05, 0.12],
            [0.12, 0.20],
            [0.20, 0.28],
            [0.28, 0.38],
            [0.38, 0.53],
            [0.53, 0.70],
        ]
        vec_args = dict(jumping_mean_ivals=jm) if use_jump else dict(select_ival=[0.05, 0.70])
        if use_each_best:
            vec_args_slda = dict(jumping_mean_ivals=jm)
            vec_args_toep = dict(select_ival=[0.05, 0.70])
        else:
            vec_args_slda = vec_args
            vec_args_toep = vec_args

        clfs = dict(
            slda=make_pipeline(
                EpochsVectorizer(
                    mne_scaler=mne.decoding.Scaler(epochs.info, scalings="mean"),
                    **vec_args_slda,
                ),
                LearningFromLabelProportions(
                    n_channels=len(epochs.ch_names),
                    n_times=6 if use_each_best or use_jump else ntimes,
                ),
            ),
            toep_lda=make_pipeline(
                EpochsVectorizer(
                    mne_scaler=mne.decoding.Scaler(epochs.info, scalings="mean"),
                    **vec_args_toep,
                ),
                LearningFromLabelProportions(
                    n_times=ntimes,
                    n_channels=len(epochs.ch_names),
                    toeplitz_time=True,
                    taper_time=linear_taper,
                ),
            ),
        )
        correct_letters = {k: 0 for k in clfs}
        aucs = {k: list() for k in clfs}
        clf_state = {k: None for k in clfs}
        for let_i in range(1, 1 + num_letters):
            gt_letter = GT[let_i - 1]
            beg, end = max(1, let_i - letter_memory + 1), let_i
            letters = [f"Letter_{i}" for i in range(beg, end + 1)]

            epo_all = epochs[letters]
            if let_i > 1 and enable_early_stopping_simulation:
                epo_train = epochs[letters[:-1]]
            else:
                epo_train = epo_all
            s, l = seq_labels_from_epoch(epo_train)

            X = epo_train
            cur_epo = epochs[f"Letter_{let_i}"]
            cur_s, cur_l = seq_labels_from_epoch(cur_epo)
            cur_n_epos = len(cur_epo)

            pred_letter = dict()
            pred_earliest = dict()
            for cli, ckey in enumerate(clfs):
                clf = clfs[ckey]
                if enable_calculate_postfix:
                    if clf_state[ckey] is None:
                        print("Training classifier on all epochs.")
                        X = epochs
                        seq, l = seq_labels_from_epoch(X)
                        clf.fit(epochs, seq)
                        clf_state[ckey] = clf
                    clf = clf_state[ckey]
                else:
                    clf.fit(X, s)
                cur_X = cur_epo
                pred = clf.decision_function(cur_X)
                (
                    pred_letter[ckey],
                    earliest_correct,
                    first_valid,
                    num_target_flashes,
                    mandatory_target_flashes,
                ) = predicted_letter_over_trial(cur_X, pred, gt=gt_letter)
                if let_i == 1:
                    earliest_correct = np.min([68, earliest_correct])
                if pred_letter[ckey] == gt_letter:
                    correct_letters[ckey] += 1

                auc = roc_auc_score(cur_l, pred)
                aucs[ckey].append(auc)
                row = dict(
                    subject=sub,
                    block=block,
                    clf=ckey,
                    nth_letter=let_i,
                    correct=pred_letter[ckey] == gt_letter,
                    correct_sofar=correct_letters[ckey],
                    earliest_correct=earliest_correct,
                    first_valid=first_valid,
                    num_target_flashes=num_target_flashes,
                    mandatory_target_flashes=mandatory_target_flashes,
                    auc=auc,
                )
                # df = pd.concat([df, row], ignore_index=True)
                # This cause FutureWarning
                df = df.append(row, ignore_index=True)

            if not np.all(np.array(list(pred_letter.values())) == gt_letter):
                print(f'Using letters {beg}-{end} (target: "{gt_letter}"):')
                for k in clfs:
                    print(f' {k.rjust(25)} predicted: "{pred_letter[k]}"')
                print("----------")

df["sample_rate"] = sr
df["ntime_features"] = ntimes
df["letter_memory"] = letter_memory
df["lowpass"] = lowpass
df["early_stop_sim"] = enable_early_stopping_simulation
df["postfix_sim"] = enable_calculate_postfix
csv_name = f"{basedir}{ds}_usup_toeplitz.csv"
df.to_csv(csv_name)
