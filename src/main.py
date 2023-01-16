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

    group_channels = False
    tfr = False

    logger.info("Loading...")
    subjects = [Subject.load(subject_id, tfr=tfr, log=False, group_channels=group_channels) for subject_id in subject_ids]

    logger.info("Training...")
    input_channels = len(C.CHANNEL_GROUPS) if group_channels else len(C.CHANNELS)
    model_class = models.TFRCNN if tfr else models.CNN
    mean_scores = crossvalidate(model_class, subjects, 10, {"input_channels": input_channels, "max_epochs": 40, "upsample": True, "device": device, "optimize_metric": "f1"})
    print(mean_scores)


if __name__ == "__main__":
    main()
