import random

import const as C

random.seed(C.RANDOM_SEED)

import argparse
import logging
from typing import Optional, Sequence

import models
from crossvalidation import crossvalidate
from data import Subject

logger = logging.Logger(__name__)


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    device = "cpu" if args.gpu is None else f"cuda:{args.gpu}"

    subject_ids = C.SUBJECT_IDS[:]
    random.shuffle(subject_ids)

    logger.info("Loading...")
    subjects = [Subject.load(subject_id, tfr=False) for subject_id in subject_ids]

    logger.info("Training...")
    mean_scores = crossvalidate(models.CNN, subjects, 10, {"epochs": 20}, device=device)
    print(mean_scores)


if __name__ == "__main__":
    main()
