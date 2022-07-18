import random

import const as C

random.seed(C.RANDOM_SEED)

import models
from crossvalidation import crossvalidate
from data import Subject


def main():
    subject_ids = Subject.ids()
    random.shuffle(subject_ids)

    print("Loading...")
    subjects = [Subject.load(subject_id) for subject_id in subject_ids]

    print("Training...")
    mean_scores = crossvalidate(models.CNN, subjects, 10)
    print(mean_scores)


if __name__ == "__main__":
    main()
