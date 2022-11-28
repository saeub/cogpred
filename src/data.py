from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

import const as C


@dataclass
class Subject:
    id: str
    eeg: torch.Tensor
    pta: float

    @staticmethod
    def load(subject_id: str, tfr: bool = False, device: str = "cpu") -> "Subject":
        filename = f"{subject_id}.tfr.pt" if tfr else f"{subject_id}.pt"
        eeg = torch.load(C.EEG_DATA_PATH / filename, device).type(torch.float32)
        pta = pd.read_csv(C.PTA_DATA_PATH, index_col="pbn_code")["PTA"][subject_id]
        return Subject(subject_id, eeg, pta)


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, subjects: Sequence[Subject]):
        self.X = torch.cat([subject.eeg for subject in subjects])
        self.y = torch.cat(
            [
                torch.tensor([subject.pta], dtype=torch.float32)
                for subject in subjects
                for trial in range(subject.eeg.size(0))
            ]
        )
        assert self.X.shape[0] == self.y.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X = self.X[index, :]
        y = self.y[index]
        return X, y

    def __len__(self) -> int:
        return self.X.shape[0]
