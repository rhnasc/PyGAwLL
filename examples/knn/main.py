import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../gawll"))
)
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../util"))
)

from gawll import GAwLL
from util import Util

import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, dimensions=None):
        if dimensions is not None:
            # Select dimensions both in train and test sets
            X = X[:, dimensions]
            X_train = self.X_train[:, dimensions]
        else:
            X_train = self.X_train

        # Predict every sample in the test set individually
        predictions = [self._predict(x, X_train) for x in X]
        return np.array(predictions)

    def _predict(self, x, X_train):
        # Calculate euclidian norm on vector difference
        distances = [np.linalg.norm(x - x_train) for x_train in X_train]

        # Select K closest labels
        k_indices = np.argsort(distances)[: self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        counter = Counter(k_nearest_labels)

        # When no label has a majority, select the numerically smallest label
        # This behavior mimics the KNN implemented on https://github.com/rtinos/GAwLL
        # and allows for a closer comparison with this re-implementation
        most_common_elements = counter.most_common()
        max_count = most_common_elements[0][1]
        result = min(elem for elem, count in most_common_elements if count == max_count)

        return result

    def evaluate(self, X_test, y_test, dimensions=None):
        y_pred = self.predict(X_test, dimensions)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy


def main():

    (chrom_size, X_trainset, d_trainset, X_testset, d_testset) = Util.read_dataset(
        "zoo"
    )

    # Total number of runs
    total_runs = 1
    # Mutation Rate
    mutation_probability = 1.0 / chrom_size
    # Maximum number of generations
    max_generations = 100

    knn = KNN()
    knn.fit(X_trainset, d_trainset)

    fitness_function = lambda chromosome: 0.98 * knn.evaluate(
        X_testset, d_testset, dimensions=chromosome
    ) + 0.02 * (sum(chromosome) / len(chromosome))

    instance = GAwLL(
        fitness_function=fitness_function,
        chrom_size=chrom_size,
        mutation_probability=mutation_probability,
        max_generations=max_generations,
    )

    for i in range(total_runs):
        instance.run(i + 1)


if __name__ == "__main__":
    main()
