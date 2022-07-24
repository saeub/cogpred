import argparse
from pathlib import Path
from typing import Tuple

import mne
import torch

import const as C


def preprocess(
    infile_path: Path,
    *,
    filter: Tuple[float, float],
    resample: float,
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
    epochs.resample(resample)
    epochs.reorder_channels(C.CHANNELS)

    return torch.from_numpy(epochs.get_data())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("subjects", nargs="+")
    parser.add_argument("--filter", nargs=2, default=(0.1, 45.0))
    parser.add_argument("--resample", default=128)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for subject_id in args.subjects:
        infile_path = (
            C.RAW_EEG_DATA_PATH
            / subject_id
            / "EEG1"
            / f"{subject_id}_Termin1_CogTAiL.bdf"
        )
        outfile_path = C.EEG_DATA_PATH / f"{subject_id}.pt"
        data = preprocess(infile_path, filter=args.filter, resample=args.resample)
        torch.save(data, outfile_path)
