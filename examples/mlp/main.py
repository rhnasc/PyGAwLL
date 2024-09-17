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
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

import warnings
from sklearn.exceptions import ConvergenceWarning


class MLP:
    def __init__(
        self,
        random_state=42,
        dataset_type=DatasetType.CLASSIFICATION,
        hidden_layer_sizes=(16, 8),
        learning_rate_init=0.01,
        max_iter=500,
    ):
        self.random_state = random_state
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter

        self._dataset_type = dataset_type

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, dimensions=None):
        if self._dataset_type == DatasetType.CLASSIFICATION:
            model = MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                learning_rate_init=self.learning_rate_init,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
        elif self._dataset_type == DatasetType.REGRESSION:
            model = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                learning_rate_init=self.learning_rate_init,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )

        if dimensions is not None:
            X_train = self.X_train[:, dimensions]
            X = X[:, dimensions]
        else:
            X_train = self.X_train

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            # warnings.filterwarnings("error", category=UserWarning)
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
    ds = "zoo"
    # ds = "housing"

    hidden_layer_sizes = {
        "zoo": (16, 8),
        "housing": (64, 32, 16),
    }[ds]

    learning_rate_init = {
        "zoo": 0.01,
        "housing": 0.001,
    }[ds]

    max_iter = {
        "zoo": 100,
        "housing": 50,
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

    mlp = MLP(
        dataset_type=dataset_type,
        hidden_layer_sizes=hidden_layer_sizes,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
    )
    mlp.fit(X_trainset, d_trainset)

    fitness_function = lambda chromosome: 0.98 * mlp.evaluate(
        X_testset, d_testset, dimensions=chromosome
    ) + 0.02 * (1 - sum(chromosome) / len(chromosome))

    # print(mlp.evaluate(X_testset, d_testset, dimensions=[True] * chrom_size))
    # return

    instance = GAwLL(
        fitness_function=fitness_function,
        chrom_size=chrom_size,
        mutation_probability=mutation_probability,
        max_generations=max_generations,
    )

    for i in range(total_runs):
        instance.run(i + 1)

    Util.save_statistics(ds, "mlp", instance.statistics)


if __name__ == "__main__":
    main()
