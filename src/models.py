from abc import ABC, abstractmethod
from typing import Collection, Dict

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
        train_subjects: Collection[Subject],
        validate_subjects: Collection[Subject],
    ):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, subject: Subject) -> float:
        raise NotImplementedError()

    def evaluate(self, test_subjects: Collection[Subject]) -> Dict[str, float]:
        pred_ptas = []
        true_ptas = []
        pred_hls = []
        true_hls = []
        for subject in test_subjects:
            pred_pta = self.predict(subject)
            pred_ptas.append(pred_pta)
            true_ptas.append(subject.pta)
            pred_hls.append(pred_pta > C.HEARING_LOSS_THRESHOLD)
            true_hls.append(subject.pta > C.HEARING_LOSS_THRESHOLD)
        # mse = np.mean((np.array(true_ptas) - np.array(pred_ptas)) ** 2)
        mse = sklearn.metrics.mean_squared_error(true_ptas, pred_ptas)
        accuracy = sklearn.metrics.accuracy_score(true_hls, pred_hls)
        precision = sklearn.metrics.precision_score(true_hls, pred_hls)
        recall = sklearn.metrics.recall_score(true_hls, pred_hls)
        f1 = sklearn.metrics.f1_score(true_hls, pred_hls)
        return {
            "mse": mse,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


class MeanDummy(Model):
    def train(
        self,
        train_subjects: Collection[Subject],
        validate_subjects: Collection[Subject],
    ):
        labels = [subject.pta for subject in train_subjects]
        self.mean_label = np.mean(labels)

    def predict(self, subject: Subject) -> float:
        return self.mean_label


class CNN(Model):
    class _Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(
                [
                    nn.LayerNorm([128, 1306]),
                    nn.Conv1d(128, 64, kernel_size=32),
                    nn.MaxPool1d(4),
                    nn.LayerNorm([64, 318]),
                    nn.Conv1d(64, 32, kernel_size=16),
                    nn.MaxPool1d(4),
                    nn.LayerNorm([32, 75]),
                    nn.Conv1d(32, 16, kernel_size=8),
                    nn.MaxPool1d(4),
                    nn.LayerNorm([16, 17]),
                    nn.Flatten(),
                    nn.Linear(272, 16),
                    nn.Linear(16, 1),
                ]
            )

        def forward(self, X: torch.Tensor):
            y = X
            for layer in self.layers:
                y = layer(y)
            assert y.size(1) == 1
            return y

    def __init__(self, *, epochs: int = 20, batch_size: int = 8, device: str = "cpu"):
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.module = self._Module().to(self.device)

    def _standardize_label(self, label: float) -> float:
        return (label - self.label_mean) / self.label_std

    def _unstandardize_label(self, label: float) -> float:
        return label * self.label_std + self.label_mean

    def train(
        self,
        train_subjects: Collection[Subject],
        validate_subjects: Collection[Subject],
    ):
        self.label_mean = np.mean([subject.pta for subject in train_subjects])
        self.label_std = np.std([subject.pta for subject in train_subjects])

        self.module.train()
        optimizer = optim.Adam(self.module.parameters())
        dataset = EEGDataset(train_subjects)
        for epoch in range(self.epochs):
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True
            )
            epoch_loss = 0
            for X, y in loader:
                X = X.to(self.device)
                y = y.to(self.device)
                y = self._standardize_label(y)
                optimizer.zero_grad()
                y_pred = self.module(X)
                loss = F.mse_loss(y_pred.squeeze(), y.squeeze())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            val_metrics = self.evaluate(validate_subjects)
            print(f"Epoch {epoch}: loss={epoch_loss}, val_metrics={val_metrics}")

    def predict(self, subject: Subject) -> float:
        # TODO: Use configurable batch size
        self.module.eval()
        X = subject.eeg
        X = X.to(self.device)
        y_pred = self.module(X)
        y_pred = torch.mean(y_pred).item()
        y_pred = self._unstandardize_label(y_pred)
        return y_pred


class TFRCNN(CNN):
    class _Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(
                [
                    nn.Flatten(1, 2),  # Combine channel and frequency dimensions
                    nn.LayerNorm([6400, 41]),
                    nn.Conv1d(6400, 64, kernel_size=4),
                    nn.MaxPool1d(2),
                    nn.LayerNorm([64, 19]),
                    nn.Conv1d(64, 32, kernel_size=4),
                    nn.MaxPool1d(2),
                    nn.LayerNorm([32, 8]),
                    nn.Flatten(),
                    nn.Linear(256, 16),
                    nn.Linear(16, 1),
                ]
            )

        def forward(self, X: torch.Tensor):
            y = X
            for layer in self.layers:
                y = layer(y)
            assert y.size(1) == 1
            return y
