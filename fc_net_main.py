import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Dict
from datasets.data_utils import load_cifar10
from classifiers.fc_net import FullyConnectedNet
from utils.gradient_check import eval_numerical_gradient
from solver import Solver


def get_CIFAR10_data(
    num_training=49000, num_validation=1000, num_test=1000, subtract_mean=True
) -> Dict[str, np.ndarray]:
    
    cifar10_dir = os.path.join(
        os.path.dirname(__file__), "datasets/cifar-10-batches-py"
    )
    X_train, y_train, X_test, y_test = load_cifar10(cifar10_dir)

    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def main():
    data = get_CIFAR10_data()
    for k, v in list(data.items()):
        print(f'{k}: {v.shape}')

    np.random.seed(231)
    N, D, H1, H2, C = 2, 15, 20, 30, 10
    X = np.random.randn(N, D)
    y = np.random.randint(C, size=(N,))

    for reg in [0, 3.14]:
        print(f'Running check with reg = {reg}')
        model = FullyConnectedNet(
            [H1, H2],
            input_dim=D,
            num_classes=C,
            reg=reg,
            weight_scale=5e-2,
            dtype=np.float64
        )

        loss, grads = model.loss(X, y)
        print(f'Initial loss: {loss}')

        for name in sorted(grads):
            f = lambda _: model.loss(X, y)[0]
            grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
            print(f'{name} relative error: {rel_error(grad_num, grads[name])}')


    num_train = 50
    small_data = {
      "X_train": data["X_train"][:num_train],
      "y_train": data["y_train"][:num_train],
      "X_val": data["X_val"],
      "y_val": data["y_val"],
    }

    weight_scale = 1e-2
    learning_rate = 1e-4
    model = FullyConnectedNet(
        [100, 100],
        weight_scale=weight_scale,
        dtype=np.float64
    )
    solver = Solver(
        model,
        small_data,
        print_every=10,
        num_epochs=20,
        batch_size=25,
        update_rule="sgd",
        optim_config={"learning_rate": learning_rate},
    )
    solver.train()

    plt.plot(solver.loss_history)
    plt.title("Training loss history")
    plt.xlabel("Iteration")
    plt.ylabel("Training loss")
    plt.grid(linestyle='--', linewidth=0.5)
    plt.show()


if __name__ == '__main__':
    main()
