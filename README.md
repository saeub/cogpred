# CogPred

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

1. Make sure that all the raw data files are linked/mounted in the directory structure described above.
2. Run `python src/preprocess.py` with subject IDs as arguments. To preprocess all: `ls data/raw/EEG/ | xargs python src/preprocess.py`
