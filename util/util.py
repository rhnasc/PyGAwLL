import os
import numpy as np
from enum import Enum

BASE_DIR = os.path.dirname(__file__)
DATASETS = {
    "zoo": os.path.join(BASE_DIR, "datasets", "zoo.dat"),
    "housing": os.path.join(BASE_DIR, "datasets", "housing.dat"),
}


class DatasetType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class Util:
    def read_dataset(dataset_name):
        try:
            with open(DATASETS[dataset_name], "r") as fin:
                lines = fin.readlines()
        except FileNotFoundError:
            print("file name error")
            return

        prob_type = None
        chrom_size = None
        n_examples = None
        X_dataset = None
        d_dataset = None
        d_dataset_regression = None
        perc_train = 0.70
        n_classes = None

        for line in lines:
            tokens = line.split()
            if not tokens:
                continue
            keyword = tokens[0][0:-1]

            if keyword == "TYPE":
                try:
                    prob_type = int(tokens[1])
                except ValueError:
                    print("TYPE error")
                    return
            elif keyword == "N_ATTRIBUTES":
                try:
                    chrom_size = int(tokens[1])
                except ValueError:
                    print("N_ATTRIBUTES error")
                    return
            elif keyword == "N_EXAMPLES":
                try:
                    n_examples = int(tokens[1])
                    X_dataset = np.zeros((n_examples, chrom_size))
                    if prob_type == 1:
                        d_dataset = np.zeros(n_examples, dtype=int)
                    else:
                        d_dataset_regression = np.zeros(n_examples)
                except ValueError:
                    print("N_EXAMPLES error")
                    return
            elif keyword == "N_CLASSES":
                try:
                    n_classes = int(tokens[1])
                except ValueError:
                    print("N_CLASSES error")
                    return
            elif keyword == "DATASET" and n_examples is not None:
                dataset_lines = lines[
                    lines.index(line) + 1 : lines.index(line) + 1 + n_examples
                ]
                for i, data_line in enumerate(dataset_lines):
                    data_tokens = data_line.split()
                    for j in range(chrom_size):
                        X_dataset[i][j] = float(data_tokens[j])
                    if prob_type == 1:
                        d_dataset[i] = int(data_tokens[chrom_size])
                    else:
                        d_dataset_regression[i] = float(data_tokens[chrom_size])
                break

        # print("Splitting the dataset in two datasets (training and testing)")

        n_examples_train = int(perc_train * n_examples)
        n_examples_test = n_examples - n_examples_train

        # print(f"n_examples_train: {n_examples_train}")
        # print(f"n_examples_test: {n_examples_test}")

        X_trainset = X_dataset[:n_examples_train]
        X_testset = X_dataset[n_examples_train:]

        if prob_type == 1:
            dataset_type = DatasetType.CLASSIFICATION
            d_trainset = d_dataset[:n_examples_train]
            d_testset = d_dataset[n_examples_train:]
        else:
            dataset_type = DatasetType.REGRESSION
            d_trainset_regression = d_dataset_regression[:n_examples_train]
            d_testset_regression = d_dataset_regression[n_examples_train:]

        # print("Dataset successfully read and split.")

        return (
            dataset_type,
            chrom_size,
            X_trainset,
            d_trainset if prob_type == 1 else d_trainset_regression,
            X_testset,
            d_testset if prob_type == 1 else d_testset_regression,
        )
