import random

import const as C

random.seed(C.RANDOM_SEED)

import argparse
from typing import Optional, Sequence

import models
from crossvalidation import crossvalidate
from data import Subject


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--group-channels", action="store_true")
    parser.add_argument("--tfr", action="store_true")
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--majority-vote", action="store_true")
    parser.add_argument("--gpu", type=int)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    device = "cpu" if args.gpu is None else f"cuda:{args.gpu}"

    subject_ids = C.SUBJECT_IDS[:]
    random.shuffle(subject_ids)

    print("Loading...")
    subjects = [Subject.load(subject_id, tfr=args.tfr, log=False, group_channels=args.group_channels) for subject_id in subject_ids]

    print("Training...")
    input_channels = len(C.CHANNEL_GROUPS) if args.group_channels else len(C.CHANNELS)
    model_class = models.TFRCNN if args.tfr else models.CNN
    mean_scores = crossvalidate(model_class, subjects, 10, {"input_channels": input_channels, "dropout": args.dropout, "max_epochs": 40, "upsample": args.upsample, "majority_vote": args.majority_vote, "device": device, "optimize_metric": "auc"})
    print(mean_scores)


if __name__ == "__main__":
    main()
