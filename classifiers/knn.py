import numpy as np

class KNearestNeighbor(object):
    def __init__(self) -> None:
        pass

    def train(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray, k: int = 1,
                num_loops: int = 0) -> np.ndarray:
        if num_loops == 0:
            dists = self.compute_distance_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distance_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distance_two_loops(X)
        else:
            raise ValueError(f'Invalid value {num_loops} for num_loops.')

        return self.predict_labels(dists, k)

    def compute_distance_two_loops(self, X: np.ndarray) -> np.ndarray:
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i][j] = np.sqrt(np.sum(np.square(X[i] - self.X_train[j])))
        return dists

    def compute_distance_one_loop(self, X: np.ndarray) -> np.ndarray:
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i] = np.sqrt(np.sum(np.square(self.X_train - X[i]), axis=1))
        return dists

    def compute_distance_no_loops(self, X: np.ndarray) -> np.ndarray:
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        X_norm_squared = np.sum(np.square(X), axis=1).reshape(num_test, 1)
        X_train_norm_squared = np.sum(np.square(self.X_train), axis=1).reshape(1, num_train)
        dists = np.sqrt(X_norm_squared + X_train_norm_squared - 2 * X.dot(self.X_train.T))
        return dists

    def predict_labels(self, dists: np.ndarray, k: int) -> np.ndarray:
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = self.y_train[np.argsort(dists[i])][:k]
            y_pred[i] = np.argmax(np.bincount(closest_y))
        return y_pred

