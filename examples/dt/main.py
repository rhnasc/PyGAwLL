import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../gawll"))
)
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../util"))
)

from gawll import GAwLL
from util import Util, DatasetType
from enum import Enum

import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error


class DT:
    def __init__(
        self,
        random_state=42,
        min_samples_split=8,
        dataset_type=DatasetType.CLASSIFICATION,
    ):
        self.random_state = random_state
        self.min_samples_split = min_samples_split
        self._dataset_type = dataset_type

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, dimensions=None):
        if self._dataset_type == DatasetType.CLASSIFICATION:
            model = DecisionTreeClassifier(
                random_state=self.random_state, min_samples_split=self.min_samples_split
            )
        elif self._dataset_type == DatasetType.REGRESSION:
            model = DecisionTreeRegressor(
                random_state=self.random_state, min_samples_split=self.min_samples_split
            )

        if dimensions is not None:
            X_train = self.X_train[:, dimensions]
            X = X[:, dimensions]
        else:
            X_train = self.X_train

        model.fit(X_train, self.y_train)
        return model.predict(X)

    def evaluate(self, X_test, y_test, dimensions=None):
        if not np.any(dimensions):
            return 0.0

        y_pred = self.predict(X_test, dimensions)

        if self._dataset_type == DatasetType.CLASSIFICATION:
            accuracy = accuracy_score(y_test, y_pred)
            return accuracy
        elif self._dataset_type == DatasetType.REGRESSION:
            mse = 1 - mean_squared_error(y_test, y_pred)
            return mse


def main():
    # ds = "zoo"
    ds = "housing"

    min_samples_split = {
        "zoo": 5,
        "housing": 16,
    }[ds]

    (dataset_type, chrom_size, X_trainset, d_trainset, X_testset, d_testset) = (
        Util.read_dataset(ds)
    )

    # Total number of runs
    total_runs = 10
    # Mutation Rate
    mutation_probability = 1.0 / chrom_size
    # Maximum number of generations
    max_generations = 2000

    dt = DT(dataset_type=dataset_type, min_samples_split=min_samples_split)
    dt.fit(X_trainset, d_trainset)

    fitness_function = lambda chromosome: 0.98 * dt.evaluate(
        X_testset, d_testset, dimensions=chromosome
    ) + 0.02 * (1 - sum(chromosome) / len(chromosome))

    instance = GAwLL(
        fitness_function=fitness_function,
        chrom_size=chrom_size,
        mutation_probability=mutation_probability,
        max_generations=max_generations,
    )

    for i in range(total_runs):
        instance.run(i + 1)

    Util.save_statistics(ds, "dt", instance.statistics)


if __name__ == "__main__":
    main()
