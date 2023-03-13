from typing import Dict, Tuple
import numpy as np
import os
import itertools
import matplotlib.pyplot as plt
from classifiers.fc_net import TwoLayerNet
from datasets.data_utils import load_cifar10
from solver import Solver
from utils.vis_utils import visualize_grid


def show_net_weights(net: TwoLayerNet):
    W1 = net.params['W1']
    W1 = W1.reshape(3, 32, 32, -1).transpose(3, 1, 2, 0)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()

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

def tune_hyperparams(data: Dict[str, np.ndarray]) -> Tuple[TwoLayerNet | None,
                                                           Solver | None]:
    results = {}
    best_val = -1
    best_model = None
    best_solver = None

    learning_rates = np.geomspace(3e-4, 3e-2, 3)
    regularization_strengths = np.geomspace(1e-6, 1e-2, 5)

    for lr, reg in itertools.product(learning_rates, regularization_strengths):
        model = TwoLayerNet(hidden_dim=128, reg=reg)
        solver = Solver(model, data, optim_config={'learning_rate': lr}, num_epochs=10, verbose=False)
        solver.train()
        
        results[(lr, reg)] = solver.best_val_acc

        if results[(lr, reg)] > best_val:
            best_val = results[(lr, reg)]
            best_model = model
            best_solver = solver

    for lr, reg in sorted(results):
        val_accuracy = results[(lr, reg)]
        print('lr %e reg %e val accuracy: %f' % (lr, reg, val_accuracy))
        
    print('best validation accuracy achieved during cross-validation: %f' % best_val)
    
    if best_model is not None:
        y_val_pred = np.argmax(best_model.loss(data['X_val']), axis=1) # pyright: ignore
        print('Validation set accuracy: ', (y_val_pred == data['y_val']).mean())

        y_test_pred = np.argmax(best_model.loss(data['X_test']), axis=1) # pyright: ignore
        print('Test set accuracy: ', (y_test_pred == data['y_test']).mean())

    return best_model, best_solver

def main():
    data = get_CIFAR10_data()
    for k, v in list(data.items()):
        print(f'{k}: {v.shape}')

    model = TwoLayerNet(hidden_dim=50)
    solver = Solver(model, data, optim_config={'learning_rate': 1e-3})
    solver.train()

    plt.subplot(2, 1, 1)
    plt.title('Training loss')
    plt.plot(solver.loss_history, 'o')
    plt.xlabel('Iteration')

    plt.subplot(2, 1, 2)
    plt.title('Accuracy')
    plt.plot(solver.train_acc_history, '-o', label='train')
    plt.plot(solver.val_acc_history, '-o', label='val')
    plt.plot([0.5] * len(solver.val_acc_history), 'k--')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.show()

    show_net_weights(model)

    best_model, best_solver = tune_hyperparams(data)

    if best_model is not None and best_solver is not None:
        plt.subplot(2, 1, 1)
        plt.title('Training loss')
        plt.plot(best_solver.loss_history, 'o')
        plt.xlabel('Iteration')

        plt.subplot(2, 1, 2)
        plt.title('Accuracy')
        plt.plot(best_solver.train_acc_history, '-o', label='train')
        plt.plot(best_solver.val_acc_history, '-o', label='val')
        plt.plot([0.5] * len(best_solver.val_acc_history), 'k--')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        plt.show()

        show_net_weights(best_model)


if __name__ == '__main__':
    main()
