import argparse
from pathlib import Path
from typing import Tuple

import mne
import numpy as np
import torch

import const as C


def preprocess(
    infile_path: Path,
    *,
    filter: Tuple[float, float],
    resample: float,
    tfr: bool,
) -> torch.Tensor:
    raw = mne.io.read_raw_bdf(infile_path, preload=True)

    # Filter
    raw.notch_filter(
        50, method="iir", iir_params={"order": 3, "ftype": "butter", "output": "sos"}
    )
    raw.filter(
        l_freq=filter[0],
        h_freq=filter[1],
        method="iir",
        iir_params={"order": 3, "ftype": "butter", "output": "sos"},
    )
    mne.set_eeg_reference(raw, copy=False)

    # Extract quiet segments
    events = mne.find_events(
        raw,
        stim_channel="Status",
        min_duration=1 / raw.info["sfreq"],
        shortest_event=1,
        initial_event=True,
    )
    # quiet_segments = []
    # for time, code in events[np.isin(events[:, 2], [C.EVENT_QUIET_START, C.EVENT_QUIET_STOP])][:, (0, 2)]:
    #     if code == C.EVENT_QUIET_START:
    #         start_time = time
    #     elif code == C.EVENT_QUIET_STOP and start_time is not None:
    #         tmin = start_time / raw.info["sfreq"] - 0.2
    #         tmax = time / raw.info["sfreq"]
    #         segment = raw.copy().crop(tmin, tmax)
    #         segment.resample(resample)
    #         quiet_segments.append(segment.get_data())
    #         start_time = None
    # assert len(quiet_segments) == 30
    # Quiet segments are between 9.41 and 11.68 s long
    epochs = mne.Epochs(
        raw, events, C.EVENT_QUIET_START, tmin=-0.2, tmax=10.0, preload=True
    )
    assert len(epochs) == 30
    epochs.reorder_channels(C.CHANNELS)

    if tfr:
        start_freq = max(filter[0], 1.0)
        end_freq = filter[1]
        freqs = (end_freq - start_freq + 1) ** (np.arange(C.TFR_RESOLUTION) / C.TFR_RESOLUTION) + start_freq - 1
        tfr_epochs = mne.time_frequency.tfr_morlet(epochs, freqs, 5, average=False, return_itc=False)
        array = tfr_epochs.data
    else:
        array = epochs.get_data()

    array = mne.filter.resample(array, resample, epochs.info["sfreq"])
    return torch.from_numpy(array)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--filter", type=float, nargs=2, default=(0.1, 45.0))
    parser.add_argument("--resample", type=float, default=128.0)
    parser.add_argument("--tfr", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    failed = []
    subject_ids = args.subjects or C.SUBJECT_IDS
    for subject_id in subject_ids:
        infile_path = (
            C.RAW_EEG_DATA_PATH
            / subject_id
            / "EEG1"
            / f"{subject_id}_Termin1_CogTAiL.bdf"
        )
        if args.tfr:
            outfile_path = C.EEG_DATA_PATH / f"{subject_id}.tfr.pt"
        else:
            outfile_path = C.EEG_DATA_PATH / f"{subject_id}.pt"
        try:
            data = preprocess(
                infile_path, filter=args.filter, resample=args.resample, tfr=args.tfr
            )
            torch.save(data, outfile_path)
        except Exception as e:
            print(f"Preprocessing subject {subject_id} failed:\n{e}")
            failed.append(subject_id)
            continue
    if len(failed) > 0:
        print(f"Subjects failed: {', '.join(failed)}")
