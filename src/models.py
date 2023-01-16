from abc import ABC, abstractmethod
import copy
from typing import Dict, Optional, Sequence

import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import const as C
from data import EEGDataset, Subject


class Model(ABC):
    @abstractmethod
    def train(
        self,
        train_subjects: Sequence[Subject],
        validate_subjects: Sequence[Subject],
    ):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, subject: Subject) -> float:
        raise NotImplementedError()

    def evaluate(self, test_subjects: Sequence[Subject], verbose: bool = False) -> Dict[str, float]:
        pred = []
        true = []
        for subject in test_subjects:
            p = self.predict(subject)
            t = subject.pta > C.HEARING_LOSS_THRESHOLD
            pred.append(p)
            true.append(t)
            if verbose:
                print(f"Subject {subject.id}: true={t:d}, pred={p:.2f}")
        pred_binary = [p > 0.5 for p in pred]
        return {
            "accuracy": sklearn.metrics.accuracy_score(true, pred_binary),
            "precision": sklearn.metrics.precision_score(true, pred_binary, zero_division=0),
            "recall": sklearn.metrics.recall_score(true, pred_binary),
            "f1": sklearn.metrics.f1_score(true, pred_binary),
            "auc": sklearn.metrics.roc_auc_score(true, pred),
        }


class CNN(Model):
    class _Module(nn.Module):
        def __init__(self, input_channels: int):
            super().__init__()
            self.layers = nn.ModuleList(
                [
                    nn.BatchNorm1d(input_channels),

                    nn.Conv1d(input_channels, 64, kernel_size=32),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.MaxPool1d(4),

                    nn.Conv1d(64, 32, kernel_size=32),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.MaxPool1d(4),

                    nn.Conv1d(32, 32, kernel_size=32),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.MaxPool1d(4),

                    nn.Flatten(),
                    nn.Linear(320, 50),
                    nn.ReLU(),
                    nn.Linear(50, 1),
                    nn.Sigmoid(),
                ]
            )

        def forward(self, X: torch.Tensor):
            y = X
            for layer in self.layers:
                y = layer(y)
            assert y.size(1) == 1
            return y

    def __init__(self, *, input_channels: int, max_epochs: int = 20, batch_size: int = 32, optimize_metric: Optional[str] = None, patience: Optional[int] = None, upsample: bool = False, device: str = "cpu"):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.optimize_metric = optimize_metric
        self.patience = patience
        self.upsample = upsample
        self.device = device
        self.module = self._Module(input_channels).to(self.device)

    def _standardize_label(self, label: torch.Tensor) -> torch.Tensor:
        return (label - self.label_mean) / self.label_std

    def _unstandardize_label(self, label: torch.Tensor) -> torch.Tensor:
        return label * self.label_std + self.label_mean

    def _upsample(self, subjects: Sequence[Subject]) -> Sequence[Subject]:
        num_hearing_loss = sum(subject.pta > C.HEARING_LOSS_THRESHOLD for subject in subjects)
        factor = round(num_hearing_loss / (len(subjects) - num_hearing_loss))
        upsampled_subjects = []
        for subject in subjects:
            if subject.pta <= C.HEARING_LOSS_THRESHOLD:
                upsampled_subjects.extend([subject] * factor)
            else:
                upsampled_subjects.append(subject)
        print(f"Upsampled {num_hearing_loss}/{len(subjects)} -> {sum(subject.pta > C.HEARING_LOSS_THRESHOLD for subject in upsampled_subjects)}/{len(upsampled_subjects)}")
        return upsampled_subjects

    def train(
        self,
        train_subjects: Sequence[Subject],
        validate_subjects: Sequence[Subject],
    ):
        if self.upsample:
            train_subjects = self._upsample(train_subjects)

        self.label_mean = np.mean([subject.pta for subject in train_subjects])
        self.label_std = np.std([subject.pta for subject in train_subjects])

        self.module.train()
        optimizer = optim.Adam(self.module.parameters())
        dataset = EEGDataset(train_subjects)

        # Early stopping
        best_checkpoint = None
        best_metric = None
        epochs_since_best = 0

        for epoch in range(self.max_epochs):
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True
            )
            epoch_loss = 0
            for X, y in loader:
                X = X.to(self.device)
                y = y.to(self.device)
                ce_weight = (y - C.HEARING_LOSS_THRESHOLD) ** 2 + 10
                ce_weight = ce_weight / ce_weight.sum()
                y = (y > C.HEARING_LOSS_THRESHOLD).float()
                optimizer.zero_grad()
                y_pred = self.module(X)
                loss = F.binary_cross_entropy(y_pred.squeeze(), y.squeeze())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            train_metrics = self.evaluate(train_subjects)
            val_metrics = self.evaluate(validate_subjects, verbose=True)
            if self.optimize_metric is not None:
                if best_checkpoint is None or val_metrics[self.optimize_metric] > best_metric:
                    best_checkpoint = copy.deepcopy(self.module.state_dict())
                    best_metric = val_metrics[self.optimize_metric]
                    epochs_since_best = 0
                else:
                    epochs_since_best += 1
            print(f"Epoch {epoch + 1}: loss={epoch_loss}\n  train_metrics: {train_metrics}\n  val_metrics: {val_metrics}")
            if self.patience is not None and epochs_since_best > self.patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break
        if best_checkpoint is not None:
            print(f"Restoring best checkpoint with {self.optimize_metric}={best_metric}")
            self.module.load_state_dict(best_checkpoint)

    def predict(self, subject: Subject) -> float:
        # TODO: Use configurable batch size
        self.module.eval()
        X = subject.eeg
        X = X.to(self.device)
        y_pred = self.module(X)
        y_pred = torch.mean(y_pred).item()
        return y_pred


class TFRCNN(CNN):
    class _Module(CNN._Module):
        def __init__(self, input_channels: int):
            super().__init__(input_channels)
            self.layers = nn.ModuleList(
                [
                    nn.Flatten(1, 2),  # Combine channel and frequency dimensions
                    nn.BatchNorm1d(input_channels * C.TFR_RESOLUTION),

                    nn.Conv1d(input_channels * C.TFR_RESOLUTION, 64, kernel_size=16),
                    nn.Conv1d(250, 64, kernel_size=16),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),

                    nn.Conv1d(64, 64, kernel_size=16),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.MaxPool1d(4),

                    nn.Flatten(),
                    nn.Linear(128, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                    nn.Sigmoid(),
                ]
            )
