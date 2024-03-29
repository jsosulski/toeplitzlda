import glob
import os
import re
import zipfile
from abc import ABC
from datetime import datetime, timezone

import numpy as np
import mne
from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset

VSPELL_BASE_URL = "https://zenodo.org/record/"
VISUAL_SPELLER_LLP_URL = VSPELL_BASE_URL + "5831826/files/"
VISUAL_SPELLER_MIX_URL = VSPELL_BASE_URL + "5831879/files/"
OPTICAL_MARKER_CODE = 500
VALID_LETTER_NUMBERS = []
VALID_LETTER_NUMBERS.extend([i for i in range(201, 242 + 1)])
VALID_LETTER_NUMBERS.extend([i for i in range(101, 142 + 1)])

NUMBER_TO_LETTER_DICT = {
    1: "A",
    2: "B",
    3: "#",
    4: "C",
    5: "#",
    6: "D",
    7: "E",
    8: "#",
    9: "F",
    10: "G",
    11: "H",
    12: "I",
    13: "J",
    14: "#",
    15: "K",
    16: "L",
    17: "M",
    18: "#",
    19: "N",
    20: "O",
    21: "P",
    22: "Q",
    23: "#",
    24: "R",
    25: "S",
    26: "T",
    27: "U",
    28: "#",
    29: "V",
    30: "W",
    31: "X",
    32: "Y",
    33: "#",
    34: "Z",
    35: " ",
    36: "#",
    37: ".",
    38: "#",
    39: ",",
    40: "!",
    41: "?",
    42: "<",
}


def seq_labels_from_epoch(epo):
    s1 = epo["Sequence_1"].events
    s1[:, 2] = 1
    s2 = epo["Sequence_2"].events
    s2[:, 2] = 2
    s = np.vstack([s1, s2])
    l1 = epo["Target"].events
    l1[:, 2] = 1
    l0 = epo["NonTarget"].events
    l0[:, 2] = 0
    l = np.vstack([l0, l1])
    s = np.array(sorted(s, key=lambda x: x[0]))
    l = np.array(sorted(l, key=lambda x: x[0]))
    return s[:, 2], l[:, 2]


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


class _BaseVisualMatrixSpellerDataset(BaseDataset, ABC):
    def __init__(
        self,
        n_subjects,
        run_split_n,
        raw_slice_offset,
        data_path=None,
        src_url=None,
        **kwargs,
    ):
        if run_split_n is not None and (run_split_n > 7 or run_split_n < 1):
            raise ValueError(f"Split must be in interval [1, 7] or None. But was {run_split_n}.")

        self.n_channels = 31  # all channels except 5 times x_* CH and EOGvu
        if kwargs["interval"] is None:
            # "Epochs were windowed to [−200, 700] ms relative to the stimulus onset [...]."
            kwargs["interval"] = [-0.2, 0.7]

        super().__init__(
            events=dict(Target=10002, NonTarget=10001),
            paradigm="p300",
            subjects=(np.arange(n_subjects) + 1).tolist(),
            **kwargs,
        )

        self.run_split_n = run_split_n
        self.raw_slice_offset = 2_000 if raw_slice_offset is None else raw_slice_offset
        self._src_url = src_url
        self._data_path = data_path

    @staticmethod
    def _filename_trial_info_extraction(vhdr_file_path):
        vhdr_file_name = os.path.basename(vhdr_file_path)
        run_file_pattern = "^matrixSpeller_Block([0-9]+)_Run([0-9]+)\\.vhdr$"
        vhdr_file_patter_match = re.match(run_file_pattern, vhdr_file_name)

        if not vhdr_file_patter_match:
            # TODO: raise a wild exception?
            print(vhdr_file_path)

        session_name = os.path.basename(os.path.dirname(vhdr_file_path))
        # session_name = re.sub("[^0-9]", "", session_name)
        block_idx = vhdr_file_patter_match.group(1)
        run_idx = vhdr_file_patter_match.group(2)
        return session_name, block_idx, run_idx

    def _get_single_subject_data(self, subject):
        subject_data_vhdr_files = self.data_path(subject)
        sessions = dict()

        for file_idx, subject_data_vhdr_file in enumerate(subject_data_vhdr_files):
            (
                session_name,
                block_idx,
                run_idx,
            ) = VisualMatrixSpellerLLPDataset._filename_trial_info_extraction(
                subject_data_vhdr_file
            )
            raw_bvr_list = _read_raw_llp_study_data(
                vhdr_fname=subject_data_vhdr_file,
                run_split_n=self.run_split_n,
                raw_slice_offset=self.raw_slice_offset,
                run_idx=run_idx,
                verbose=None,
            )

            session_name = f"{session_name}_block_{block_idx}"
            session_name = "".join(c for c in session_name if c.isdigit())
            if self.run_split_n is not None:
                for split_idx, raw_split in enumerate(raw_bvr_list):
                    session_name = session_name + f"_run_{run_idx}_split_{split_idx}"
                    sessions[session_name] = {"0": raw_split}
            else:
                if session_name not in sessions.keys():
                    sessions[session_name] = dict()
                sessions[session_name][run_idx] = raw_bvr_list[0]

        return sessions

    def data_path(self, subject, path=None, force_update=False, update_path=None, verbose=None):

        url = f"{self._src_url}subject{subject:02d}.zip"
        data_archive_path = dl.data_path(url, "llp")
        data_dir_extracted_path = os.path.dirname(data_archive_path)
        # else:
        #     raise ValueError(f'URL or data path must be given but both are None.')

        subject_dir_path = os.path.join(data_dir_extracted_path, f"subject{subject:02d}")

        data_extracted = os.path.isdir(subject_dir_path)
        if not data_extracted:
            # print('unzip', path_to_data_archive)  # TODO logging? check verbose
            zipfile_path = glob.glob(
                os.path.join(data_dir_extracted_path, data_archive_path, "*.zip")
            )[0]
            _BaseVisualMatrixSpellerDataset._extract_data(data_dir_extracted_path, zipfile_path)

        run_glob_pattern = os.path.join(
            data_dir_extracted_path,
            f"subject{subject:02d}",
            "matrixSpeller_Block*_Run*.vhdr",
        )
        subject_paths = glob.glob(run_glob_pattern)

        return natural_sort(subject_paths)

    @staticmethod
    def _extract_data(data_dir_extracted_path, data_archive_path):
        zip_ref = zipfile.ZipFile(data_archive_path, "r")
        zip_ref.extractall(data_dir_extracted_path)

    def load_epochs(self, block_nrs=(1, 2, 3), fband=(0.5, 16), sampling_rate=100):
        raws = list()
        event_list = list()
        block_ids = list()
        for subj_id, subj_data in self.get_data().items():
            for session_id, session_data in subj_data.items():
                for run_id, run_data in session_data.items():
                    bnr = int(run_data.filenames[0].split("Block")[1][0])
                    if bnr in block_nrs:
                        raws.append(run_data)
                        events, _ = mne.events_from_annotations(run_data)
                        event_list.append(events)
                        block_ids.extend([bnr] * len(events))

        DUMMY_DATESTR = "20100101120150667372"
        meas_date = datetime.strptime(DUMMY_DATESTR, "%Y%m%d%H%M%S%f")
        meas_date = meas_date.replace(tzinfo=timezone.utc)
        [r.set_meas_date(meas_date) for r in raws]
        raws_epochs, _ = mne.concatenate_raws(raws, events_list=event_list, preload=True)
        raws_epochs = raws_epochs.filter(*fband)
        events, events_dict = mne.events_from_annotations(raws_epochs)

        epochs = mne.Epochs(
            raw=raws_epochs,
            events=events,
            event_id=events_dict,
            picks=["eeg"],
            preload=True,
            tmin=self.tmin,
            tmax=self.tmax,
            baseline=None,
        )
        epochs = epochs.resample(sampling_rate)
        # symbols = pd.unique(list(NUMBER_TO_LETTER_DICT.values()))
        # cols = ["block_nr", "letter_nr", *symbols]
        # md = pd.DataFrame(columns=cols)
        # for i in range(len(epochs)):
        #     print(str(i))
        #     e = epochs[i]
        #     ev = list(e.event_id.keys())[0].split("/")
        #     row = pd.Series(index=cols)
        #     row[:] = False
        #     highlights = [e for e in ev if len(e) == 1]
        #     row.loc[highlights] = True
        #     row.block_nr = block_ids[i]
        #     row.letter_nr = int(ev[0].split("_")[-1])
        #     md = md.append(row, ignore_index=True)
        #
        # epochs.metadata = md

        return epochs, raws_epochs


class VisualMatrixSpellerLLPDataset(_BaseVisualMatrixSpellerDataset):
    """
    Learning from label proportions for a visual matrix speller (ERP) dataset from Hübner et al 2017 [1]_.


    **Dataset description**

    The subjects were asked to spell the sentence: “Franzy jagt im komplett verwahrlosten Taxi quer durch Freiburg”.
    The sentence was chosen because it contains each letter used in German at least once. Each subject spelled this
    sentence three times. The stimulus onset asynchrony (SOA) was 250 ms (corresponding to 15 frames on the LCD screen
    utilized) while the stimulus duration was 100 ms (corresponding to 6 frames on the LCD screen utilized). For each
    character, 68 highlighting events occurred and a total of 63 characters were spelled three times. This resulted in
    a total of 68 ⋅ 63 ⋅ 3 = 12852 EEG epochs per subject. Spelling one character took around 25 s including 4 s for
    cueing the current symbol, 17 s for highlighting and 4 s to provide feedback to the user. Assuming a perfect
    decoding, these timing constraints would allow for a maximum spelling speed of 2.4 characters per minute. Fig 1
    shows the complete experimental structure and how LLP is used to reconstruct average target and non-target ERP
    responses.

    Subjects were placed in a chair at 80 cm distance from a 24-inch flat screen. EEG signals from 31 passive Ag/AgCl
    electrodes (EasyCap) were recorded, which were placed approximately equidistantly according to the extended
    10–20 system, and whose impedances were kept below 20 kΩ. All channels were referenced against the nose and the
    ground was at FCz. The signals were registered by multichannel EEG amplifiers (BrainAmp DC, Brain Products) at a
    sampling rate of 1 kHz. To control for vertical ocular movements and eye blinks, we recorded with an EOG electrode
    placed below the right eye and referenced against the EEG channel Fp2 above the eye. In addition, pulse and
    breathing activity were recorded.

    Parameters
    ----------
    run_split_n: None, int
        Define how to split the runs. If None load whole EEG sessions. If integer load splits of run_split_n runs.
        That is to say runs are loaded individually if run_split_n is set to one. If run_split_n is set to 2 then the
        data is split into two splits of three trials a 68 epochs. The seventh trial is discarded. Default is set to
        None.
        Note: the data can be split into at most seven splits.
    interval: array_like
        range/interval in milliseconds in which the brain response/activity relative to an event/stimulus onset lies in.
        Default is set to [-.2, .7].
    raw_slice_offset: int, None
        defines the crop offset in milliseconds before the first and after the last event (target or non-targeet) onset.
        Default None which crops with an offset 2,000 ms.

    References
    ----------
    .. [1] Hübner, D., Verhoeven, T., Schmid, K., Müller, K. R., Tangermann, M., & Kindermans, P. J. (2017)
           Learning from label proportions in brain-computer interfaces: Online unsupervised learning with guarantees.
           PLOS ONE 12(4): e0175856.
           https://doi.org/10.1371/journal.pone.0175856
    """

    def __init__(
        self,
        sessions_per_subject=1,
        data_path=None,
        run_split_n=None,
        interval=None,
        raw_slice_offset=None,
    ):
        llp_speller_paper_doi = "10.1371/journal.pone.0175856"
        self.tmin = -0.2
        self.tmax = 0.8
        self.downsample_freq = 100
        super().__init__(
            src_url=VISUAL_SPELLER_LLP_URL,
            data_path=data_path,
            run_split_n=run_split_n,
            raw_slice_offset=raw_slice_offset,
            n_subjects=13,
            sessions_per_subject=sessions_per_subject,  # if varying, take minimum
            code="VisualSpellerLLP",
            interval=interval,
            doi=llp_speller_paper_doi,
        )


class VisualMatrixSpellerMixDataset(_BaseVisualMatrixSpellerDataset):
    """
    Mixtue of LLP and EM for a visual matrix speller (ERP) dataset from Hübner et al 2018 [1]_.

    Within a single session, a subject was asked to spell the beginning of a sentence in each of three blocks.The text
    consists of the 35 symbols “Franzy jagt im Taxi quer durch das”. Each block, one of the three decoding
    algorithms (EM, LLP, MIX) was used in order to guess the attended symbol. The order of the blocks was
    pseudo-randomized over subjects, such that each possible order of the three decoding algorithms was used twice.
    This randomization should reduce systematic biases by order effects or temporal effects, e.g., due to fatigue or
    task-learning.

    A trial describes the process of spelling one character. Each of the 35 trials per block contained 68 highlighting
    events. The stimulus onset asynchrony (SOA) was 250 ms and the stimulus duration was 100 ms leading to an
    interstimulus interval (ISI) of 150 ms.

    Parameters
    ----------
    run_split_n: None, int
        Define how to split the runs. If None load whole EEG sessions. If integer load splits of run_split_n runs.
        That is to say runs are loaded individually if run_split_n is set to one. If run_split_n is set to 2 then the
        data is split into two splits of three trials a 68 epochs. The seventh trial is discarded. Default is set to
        None.
        Note: the data can be split into at most seven splits.
    interval: array_like
        range/interval in milliseconds in which the brain response/activity relative to an event/stimulus onset lies in.
        Default is set to [-.2, .7].
    raw_slice_offset: int, None
        defines the crop offset in milliseconds before the first and after the last event (target or non-targeet) onset.
        Default None which crops with an offset 2,000 ms.

    References
    ----------
    .. [1] Huebner, D., Verhoeven, T., Mueller, K. R., Kindermans, P. J., & Tangermann, M. (2018).
           Unsupervised learning for brain-computer interfaces based on event-related potentials: Review and online
           comparison [research frontier].
           IEEE Computational Intelligence Magazine, 13(2), 66-77.
           https://doi.org/10.1109/MCI.2018.2807039
    """

    def __init__(self, run_split_n=None, interval=None, raw_slice_offset=None):
        mix_speller_paper_doi = "10.1109/MCI.2018.2807039"
        self.tmin = -0.2
        self.tmax = 0.8
        self.downsample_freq = 100
        super().__init__(
            src_url=VISUAL_SPELLER_MIX_URL,
            run_split_n=run_split_n,
            raw_slice_offset=raw_slice_offset,
            n_subjects=12,
            sessions_per_subject=1,  # if varying, take minimum
            code="VisualSpellerMIX",
            interval=interval,
            doi=mix_speller_paper_doi,
        )


def _read_raw_llp_study_data(vhdr_fname, run_split_n, raw_slice_offset, run_idx, verbose=None):
    """
    Read LLP BVR recordings file. Ignore the different sequence lengths. Just tag event as target or non-target if it
    contains a target or does not contain a target.

    Parameters
    ----------
    vhdr_fname: str
        Path to the EEG header file.
    n_chars: number of characters of the spelled sentence
    verbose : bool, int, None
        specify the loglevel.

    Returns
    -------
    raw_object: mne.io.Raw
        the loaded BVR raw object.
    """
    non_scalp_channels = ["EOGvu", "x_EMGl", "x_GSR", "x_Respi", "x_Pulse", "x_Optic"]
    raw_bvr = mne.io.read_raw_brainvision(
        vhdr_fname=vhdr_fname,  # eog='EOGvu',
        misc=non_scalp_channels,
        preload=True,
        verbose=verbose,
    )  # type: mne.io.Raw
    raw_bvr.set_montage("standard_1020")
    # TODO: MNE/MOABB do not like anonymized data, therefore we need to set a dummy date
    # DUMMY_DATESTR = '20100101120150667372'
    # meas_date = datetime.strptime(DUMMY_DATESTR, '%Y%m%d%H%M%S%f')
    # meas_date = meas_date.replace(tzinfo=timezone.utc)
    # raw_bvr.set_meas_date(meas_date)

    events = _parse_events(raw_bvr)

    (
        onset_arr_list,
        marker_arr_list,
        sequence_arr_list,
        letter_arr_list,
    ) = _extract_target_non_target_description(events, run_split_n)

    has_multiple_splits = run_split_n is not None and run_split_n > 1

    def annotate_and_crop_raw(onset_arr, marker_arr, sequence_arr):
        raw = raw_bvr.copy() if has_multiple_splits else raw_bvr
        raw_annotated = raw.set_annotations(
            _create_annotations_from(
                marker_arr, onset_arr, sequence_arr, letter_arr_list, raw, run_idx
            )
        )

        tmin = max((onset_arr[0] - raw_slice_offset) / 1e3, 0)
        tmax = min((onset_arr[-1] + raw_slice_offset) / 1e3, raw.times[-1])
        return raw_annotated.crop(tmin=tmin, tmax=tmax, include_tmax=True)

    return list(map(annotate_and_crop_raw, onset_arr_list, marker_arr_list, sequence_arr_list))


def _create_annotations_from(marker_arr, onset_arr, sequence_arr, letter_arr, raw_bvr, run_idx):
    default_bvr_marker_duration = raw_bvr.annotations[0]["duration"]

    onset = onset_arr / 1e3  # convert onset in seconds to ms
    durations = np.repeat(default_bvr_marker_duration, len(marker_arr))
    description = _create_description(marker_arr, sequence_arr, letter_arr, run_idx)

    orig_time = raw_bvr.annotations[0]["orig_time"]
    return mne.Annotations(
        onset=onset, duration=durations, description=description, orig_time=orig_time
    )


def _create_description(marker_arr, sequence_arr, letter_arr, run_idx):
    desc = list()
    epochs_per_trial = 68
    trials_per_run = 7
    letter_nr = (trials_per_run * (int(run_idx) - 1)) + 1
    for i in range(0, trials_per_run):
        start_idx = i * epochs_per_trial
        end_idx = (i + 1) * epochs_per_trial

        for label, seq, letters in zip(
            marker_arr[start_idx:end_idx],
            sequence_arr[start_idx:end_idx],
            letter_arr[0][start_idx:end_idx],
        ):
            single_desc = f"Letter_{letter_nr}/"
            if seq == 21:
                single_desc += "Sequence_1/"
            elif seq == 22:
                single_desc += "Sequence_2/"

            if label == 0:
                single_desc += "NonTarget/"
            elif label == 1:
                single_desc += "Target/"
            for letter in letters:
                letter = int(str(letter)[-2:])
                if letter != -1:
                    single_desc += NUMBER_TO_LETTER_DICT[letter] + "/"
            desc.append(single_desc[:-1])
        letter_nr += 1
    return desc


def _parse_events(raw_bvr):
    stimulus_pattern = re.compile("(Stimulus/S|Optic/O) *([0-9]+)")

    def parse_marker(desc):
        match = stimulus_pattern.match(desc)
        if match is None:
            return None
        if match.group(1) == "Optic/O":
            return OPTICAL_MARKER_CODE
        return int(match.group(2))

    events, _ = mne.events_from_annotations(raw=raw_bvr, event_id=parse_marker, verbose=None)
    return events


def _find_single_trial_start_end_idx(events):
    trial_start_end_markers = [21, 22, 10]
    return np.where(np.isin(events[:, 2], trial_start_end_markers))[0]


def _extract_target_non_target_description(events, run_split_n):
    single_trial_start_end_idx = _find_single_trial_start_end_idx(events)

    n_events = single_trial_start_end_idx.size - 1

    onset_arr = np.empty((n_events,), dtype=np.int64)
    marker_arr = np.empty((n_events,), dtype=np.int64)
    sequence_arr = np.empty((n_events,), dtype=np.int64)
    letter_arr = np.empty((n_events, 12), dtype=np.int64)

    broken_events_idx = list()
    for epoch_idx in range(n_events):
        epoch_start_idx = single_trial_start_end_idx[epoch_idx]
        epoch_end_idx = single_trial_start_end_idx[epoch_idx + 1]

        epoch_events = events[epoch_start_idx:epoch_end_idx]
        onset_ms = _find_epoch_onset(epoch_events)
        if onset_ms == -1:
            broken_events_idx.append(epoch_idx)
            continue

        onset_arr[epoch_idx] = onset_ms
        marker_arr[epoch_idx] = int(
            _single_trial_contains_target(epoch_events)
        )  # 1/true if single trial has target

        sequence_arr[epoch_idx] = _single_trail_sequence_type(epoch_events)
        letter_arr[epoch_idx] = _single_trail_letter_numbers(epoch_events)
    if run_split_n is None or run_split_n == 1:
        return (
            [np.delete(onset_arr, broken_events_idx)],
            [np.delete(marker_arr, broken_events_idx)],
            [np.delete(sequence_arr, broken_events_idx)],
            [np.delete(letter_arr, broken_events_idx, axis=0)],
        )
    else:
        return _split_run_into_n_splits_of_trials(
            broken_events_idx,
            marker_arr,
            onset_arr,
            sequence_arr,
            letter_arr,
            run_split_n,
        )


def _split_run_into_n_splits_of_trials(
    broken_events, marker_arr, onset_arr, sequence_arr, letter_arr, run_split_n
):
    epochs_per_trial = 68
    trials_per_run = 7

    trials_per_split = int(trials_per_run / run_split_n)
    trial_split_idx = (
        np.arange(start=trials_per_split, stop=trials_per_run, step=trials_per_split)
        * epochs_per_trial
    )
    epochs_per_split = trials_per_split * epochs_per_trial

    broken_events = np.array(broken_events)
    has_broken = len(broken_events) > 0

    def broken_idx_for_split(split_idx):
        if not has_broken:
            return []

        broke_int_split_i = broken_events - split_idx * epochs_per_split
        idx_broken_events_in_split = (broke_int_split_i >= 0) & (
            broke_int_split_i < epochs_per_split
        )
        return broke_int_split_i[idx_broken_events_in_split]

    def split_data(data_arr):
        splits = np.split(data_arr, trial_split_idx)
        del splits[run_split_n:]

        return [np.delete(split, broken_idx_for_split(i)) for i, split in enumerate(splits)]

    onset_arr_list = split_data(onset_arr)
    marker_arr_list = split_data(marker_arr)
    sequence_arr_list = split_data(sequence_arr)
    letter_arr_list = split_data(letter_arr)
    return onset_arr_list, marker_arr_list, sequence_arr_list, letter_arr_list


def _find_epoch_onset(epoch_events):
    optical_idx = epoch_events[:, 2] == OPTICAL_MARKER_CODE
    stimulus_onset_time = epoch_events[optical_idx, 0]

    def second_optical_is_feedback():
        if stimulus_onset_time.size != 2:
            return False

        stimulus_prior_second_optical_marker = epoch_events[np.where(optical_idx)[0][1] - 1, 2]
        return stimulus_prior_second_optical_marker in [50, 51, 11]

    if stimulus_onset_time.size == 1 or second_optical_is_feedback():
        return stimulus_onset_time[0]

    # broken epoch: no true onset found..
    return -1


def _single_trial_contains_target(trial_events):
    trial_markers = trial_events[:, 2]
    return np.any((trial_markers > 100) & (trial_markers <= 142))


def _single_trail_sequence_type(trial_events):
    sequence_marker = trial_events[:, 2][0]
    return sequence_marker


def _single_trail_letter_numbers(trial_events):
    sequence_marker = trial_events[:, 2]
    sequence_letter_numbers = []
    for marker in sequence_marker:
        if marker in VALID_LETTER_NUMBERS:
            sequence_letter_numbers.append(marker)

    return np.array(sequence_letter_numbers)
