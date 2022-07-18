from typing import Any, Dict, Sequence, Type

import numpy as np

from data import Subject
from models import Model


def crossvalidate(
    model_class: Type[Model],
    subjects: Sequence[Subject],
    num_folds: int,
    model_kwargs: Dict[str, Any],
):
    fold_size = len(subjects) // num_folds
    fold_scores = {}
    for i in range(num_folds):
        validate_start = i * fold_size
        validate_end = (i + 1) * fold_size
        test_start = (i + 1) % num_folds * fold_size
        test_end = ((i + 1) % num_folds + 1) * fold_size
        train_subjects = subjects[:validate_start] + subjects[test_end:]
        validate_subjects = subjects[validate_start:validate_end]
        test_subjects = subjects[test_start:test_end]
        model = model_class(**model_kwargs)
        model.train(train_subjects, validate_subjects)
        scores = model.evaluate(test_subjects)
        for metric, score in scores.items():
            if metric not in fold_scores:
                fold_scores[metric] = []
            fold_scores[metric].append(score)
    return {metric: np.mean(scores) for metric, scores in fold_scores.items()}
