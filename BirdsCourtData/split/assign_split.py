"""Builds BirdsCourtData/Train and BirdsCourtData/Test from a flat pool.

Pool layout expected (one folder per modality, same basenames):
    pool/Image/*.jpg  pool/Annotation/*.json  pool/Thumbnell/*.jpg  pool/VGGT Outputs/*.json

Usage (from anywhere):
    python3 assign_split.py path/to/pool

Copies (does not move) each file into Train/ or Test/ according to
train_test_split.json. Images in neither list are ignored.
"""
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(HERE)
SPLIT = os.path.join(HERE, "train_test_split.json")
SUBDIRS = ["Image", "Annotation", "Thumbnell", "VGGT Outputs"]


def main():
    pool = sys.argv[1]
    split = json.load(open(SPLIT))
    groups = {"Train": split["train_set"], "Test": split["test_set"]}
    for group, bases in groups.items():
        for sub in SUBDIRS:
            os.makedirs(os.path.join(DATA, group, sub), exist_ok=True)
            n = 0
            for base in bases:
                for f in os.listdir(os.path.join(pool, sub)):
                    if os.path.splitext(f)[0] == base:
                        shutil.copy2(os.path.join(pool, sub, f), os.path.join(DATA, group, sub, f))
                        n += 1
            print(f"{group}/{sub}: {n}")


if __name__ == "__main__":
    main()
