import csv
from itertools import product
from pathlib import Path

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from model import DenseModel
from util import (
    confidence_interval,
    load_fashion_mnist,
    plot_confusion_matrix,
    plot_predictions,
    plot_saliency,
)

CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]


def main():
    train, val, test = load_fashion_mnist()
    x_train, y_train = train
    x_val, y_val = val
    x_test, y_test = test

    architectures = {'one_hidden_layer': (128,), 'two_hidden_layers': (128, 64)}
    learning_rates = [0.001, 0.0001]
    dropout_rates = [0.0, 0.3]
    results = []
    best_result = None
    best_weights = None

    # Two architectures x two learning rates x dropout off/on = eight runs.
    for (name, units), learning_rate, dropout_rate in product(
        architectures.items(), learning_rates, dropout_rates
    ):
        # Start each run with a fresh model and optimizer.
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)
        print(f'\n{name}: learning_rate={learning_rate}, dropout={dropout_rate}')
        model = DenseModel(
            hidden_layers=units, learning_rate=learning_rate, dropout_rate=dropout_rate
        )
        model.build((None, 28, 28))
        model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=10,
            batch_size=64,
            verbose=2,
        )

        # Select settings using validation data only.
        val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
        results.append({
            'architecture': name,
            'learning_rate': learning_rate,
            'dropout_rate': dropout_rate,
            'epochs': 10,
            'batch_size': 64,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
        })
        if best_result is None or val_loss < best_result['val_loss']:
            best_result = results[-1]
            best_weights = model.get_weights()
        print(f'Validation loss: {val_loss:.4f}, accuracy: {val_accuracy:.4f}')

    results_path = Path(__file__).with_name('results.csv')
    with results_path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f'\nSaved all eight validation results to {results_path}')

    # The search is complete. Evaluate the selected model on the test set once.
    tf.keras.backend.clear_session()
    best_model = DenseModel(
        hidden_layers=architectures[best_result['architecture']],
        learning_rate=best_result['learning_rate'],
        dropout_rate=best_result['dropout_rate'],
    )
    best_model.build((None, 28, 28))
    best_model.set_weights(best_weights)
    test_loss, test_accuracy = best_model.evaluate(x_test, y_test, verbose=0)
    test_predictions = best_model.predict(x_test, verbose=0)

    images_dir = Path(__file__).with_name('images')
    images_dir.mkdir(exist_ok=True)

    confusion_display = plot_confusion_matrix(
        y_test, test_predictions,
        title='Fashion-MNIST Test Confusion Matrix',
        save_path=images_dir / 'confusion_matrix.png',
    )
    plt.close(confusion_display.figure_)

    # 95% confidence interval on the test-set generalization error rate.
    error_rate, error_ci_lower, error_ci_upper = confidence_interval(y_test, test_predictions)

    final_result = dict(
        best_result,
        test_loss=test_loss,
        test_accuracy=test_accuracy,
        test_error_rate=error_rate,
        test_error_ci_lower=error_ci_lower,
        test_error_ci_upper=error_ci_upper,
    )
    with results_path.with_name('test_results.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=final_result.keys())
        writer.writeheader()
        writer.writerow(final_result)
    print(f'Selected configuration: {best_result}')
    print(f'Final test loss: {test_loss:.4f}, accuracy: {test_accuracy:.4f}')
    print(
        f'Test error rate: {error_rate:.4f} '
        f'(95% CI: [{error_ci_lower:.4f}, {error_ci_upper:.4f}])'
    )
    print(f'Confusion matrix saved to {images_dir / "confusion_matrix.png"}')

    visualize_predictions(best_model, x_test, y_test, test_predictions, images_dir)


def visualize_predictions(model, x_test, y_test, predictions, images_dir, n_correct=4, n_incorrect=4):
    """Save prediction and saliency figures for a sample of test examples.

    Saliency maps show, per pixel, how much the predicted class probability
    would change for a small change in that pixel (the absolute gradient of
    the output w.r.t. the input). Pixels with the brightest heatmap values
    are the ones most responsible for the model's prediction.
    """
    predicted_labels = predictions.argmax(axis=1)
    true_labels = y_test.argmax(axis=1)
    correct_indices = np.flatnonzero(predicted_labels == true_labels)
    incorrect_indices = np.flatnonzero(predicted_labels != true_labels)

    # Mix correct and incorrect examples so the saliency comparison is informative.
    rng = np.random.default_rng(42)
    sample_indices = np.concatenate([
        rng.choice(correct_indices, size=n_correct, replace=False),
        rng.choice(incorrect_indices, size=n_incorrect, replace=False),
    ])

    print('\nSaving prediction and saliency figures for sample test examples:')
    for rank, index in enumerate(sample_indices):
        prediction_fig = plot_predictions(
            x_test[index], y_test[index], predictions[index],
            save_path=images_dir / f'prediction_{rank}.png',
        )
        saliency_fig = plot_saliency(
            model, x_test[index],
            save_path=images_dir / f'saliency_{rank}.png',
        )
        plt.close(prediction_fig)
        plt.close(saliency_fig)

        true_class = CLASS_NAMES[true_labels[index]]
        predicted_class = CLASS_NAMES[predicted_labels[index]]
        outcome = 'correct' if true_class == predicted_class else 'incorrect'
        print(
            f'  Example {rank} (test index {index}): true={true_class}, '
            f'predicted={predicted_class} ({outcome})'
        )
    print(f'Figures saved to {images_dir}')


if __name__ == '__main__':
    main()
