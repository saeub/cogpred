from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

import const as C


@dataclass
class Subject:
    id: str
    eeg: np.ndarray
    pta: float

    @staticmethod
    def load(subject_id: str) -> "Subject":
        # TODO: Load directly to GPU
        eeg = np.load(C.EEG_DATA_PATH / f"{subject_id}.npy")
        pta = pd.read_csv(C.PTA_DATA_PATH, index_col="pbn_code")["PTA"][subject_id]
        return Subject(subject_id, eeg, pta)

    @staticmethod
    def ids() -> List[str]:
        all_ids = [path.stem for path in C.EEG_DATA_PATH.iterdir()]
        return [id for id in all_ids if id not in ["4a71bn", "57yhxc"]]


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, subjects: List[Subject]):
        self.X = np.concatenate([subject.eeg for subject in subjects])
        self.y = np.concatenate(
            [
                [subject.pta]
                for subject in subjects
                for trial in range(subject.eeg.shape[0])
            ]
        )
        assert self.X.shape[0] == self.y.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X = self.X[index, :]
        y = self.y[index]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )

    def __len__(self) -> int:
        return self.X.shape[0]
