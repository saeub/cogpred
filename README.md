# CogPred

Experiments in predicting hearing loss (based on pure tone average (PTA)) from EEG recordings.

## Setup

- `pip install -r requirements.txt`

## Directory structure

- `src/`: source code
- `data/`
    - `raw/EEG/${subject_id}/`: raw EEG data (1 folder per subject)
        - `EEG1/${subject_id}_Termin1_CogTAiL.bdf`: raw EEG data file
    - `preproccessed`
        - `eeg`: preprocessed EEG data (PyTorch)
        - `measurements`: auditory and cognitive measurements
            - `PTA.csv`: auditory threshold measurements
            - `cognitive_measures.csv`: cognitive measurements

## EEG preprocessing

We only perform minimal automatic preprocessing (cropping/aligning, bandpass filtering, and optional time-frequency analysis).

1. Make sure that all the raw data files are linked/mounted in the directory structure described above.
2. Run `python src/preprocess.py` with subject IDs as arguments. To preprocess all: `ls data/raw/EEG/ | xargs python src/preprocess.py`  
   Use `--tfr` to generate the time-frequency representation instead of the raw time-domain signal. This adds a frequency dimension and reduces time resolution).

## Models

The model architecture is a CNN with three convolution and max-pooling layers (two in the case of TFR input, due to the already reduced dimensionality), batch normalization and optional dropout, followed by two dense layers. See [`models.py`](https://github.com/saeub/cogpred/blob/main/src/models.py) for details.

## Training and evaluation

`python src/main.py` will run ten-fold cross-validation with default hyperparameters. `python src/main.py --help` shows configurable hyperparameters.

## Results

In the table below, accuracy, F1, and area under the ROC curve (AUC) are the means of those values across all ten test folds. The models were trained for a maximum of 40 epochs, and the checkpoint with the best validation AUC score was used for evaluation.

| Preprocessing | Channel grouping | Dropout | Accuracy | F1    | AUC   |
| ------------- | ---------------- | ------- | -------- | ----- | ----- |
| raw           | no               | 0.0     | 0.615    | 0.641 | 0.508 |
| raw           | no               | 5.0     | 0.585    | 0.640 | 0.466 |
| raw           | yes              | 0.0     | 0.508    | 0.470 | 0.577 |
| raw           | yes              | 5.0     | 0.531    | 0.485 | 0.536 |
| tfr           | no               | 0.0     | 0.654    | 0.673 | 0.485 |
| tfr           | no               | 5.0     | 0.592    | 0.515 | 0.457 |
| tfr           | yes              | 0.0     | 0.562    | 0.504 | 0.550 |
| tfr           | yes              | 5.0     | 0.431    | 0.255 | 0.493 |

## Other things I tried

- Regression instead of classification (didn't learn anything)
- Not upsampling the minority class (didn't learn anything or very slowly)
- Predicting n-back performance instead of PTA (wasn't any easier)
