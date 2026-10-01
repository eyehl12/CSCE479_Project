from statistics import NormalDist

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def load_fashion_mnist(val_split=0.2, random_state=42):
    #load data
    train_data, test_data = tfds.load('fashion_mnist', split=['train', 'test'], as_supervised=True, batch_size=-1)
    train_images, train_labels = tfds.as_numpy(train_data)
    test_images, test_labels = tfds.as_numpy(test_data)

    #reshape images to match the model input
    train_images = train_images.reshape(-1, 28, 28)
    test_images = test_images.reshape(-1, 28, 28)

    #split official training data into training and validation
    x_test, y_test = test_images, test_labels
    x_train, x_val, y_train, y_val = train_test_split(
        train_images, train_labels,
        test_size=val_split,
        random_state=random_state,
        stratify=train_labels
    )

    #convert class labels into one-hot vectors
    y_train = to_categorical(y_train, num_classes=10)
    y_val = to_categorical(y_val, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    #return train, validation, and test data as three separate tuples
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)



def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', save_path=None):
    #convert one-hot labels and model predictions into class IDs
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    #get position of the predicted class and true class via argmanx
    y_true = y_true.argmax(axis=1)
    y_pred = y_pred.argmax(axis=1)

    #class names
    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

    #normalize each true-class row into proportions
    matrix = confusion_matrix(
        y_true, y_pred, labels=range(10), normalize='true'
    )

    #build the figure
    fig, ax = plt.subplots(figsize=(10, 8))
    display = ConfusionMatrixDisplay(matrix, display_labels=class_names)
    display.plot(ax=ax, cmap='Blues', values_format='.2f', xticks_rotation=45)
    display.im_.set_clim(0, 1)
    ax.set_title(title)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return display



def confidence_interval(y_true, y_pred, confidence=0.95):
    #convert labels and predictions into class IDs
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    y_true = y_true.argmax(axis=1)
    y_pred = y_pred.argmax(axis=1)

    #calculate the fraction of incorrectly classified examples
    sample_count = len(y_true)
    error_rate = float(np.mean(y_true != y_pred))
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)

    #Caculate lower and upper bounds
    denominator = 1 + z**2 / sample_count
    center = (error_rate + z**2 / (2 * sample_count)) / denominator
    margin = z * np.sqrt(
        error_rate * (1 - error_rate) / sample_count
        + z**2 / (4 * sample_count**2)
    ) / denominator

    lower_bound = max(0.0, float(center - margin))
    upper_bound = min(1.0, float(center + margin))
    return error_rate, lower_bound, upper_bound



def plot_predictions(image, y_true, y_pred, save_path=None):
    #get the predicted class
    true_class = np.argmax(y_true)
    predicted_class = np.argmax(y_pred)

    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

    #create the figure
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    color = 'green' if true_class == predicted_class else 'red'
    ax.set_title(
        f'Actual: {class_names[true_class]}\n'
        f'Predicted: {class_names[predicted_class]}',
        color=color,
    )
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


'''
methods below need chaning

'''
def plot_training_history(history, save_path=None):
    """Plot training and validation loss/accuracy from model.fit() history."""
    epochs = np.asarray(history.epoch) + 1
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    #compare training and validation loss
    axes[0].plot(epochs, history.history['loss'], label='Training')
    axes[0].plot(epochs, history.history['val_loss'], label='Validation')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (includes any regularization penalty)')
    axes[0].legend()

    #compare training and validation accuracy
    axes[1].plot(epochs, history.history['accuracy'], label='Training')
    axes[1].plot(epochs, history.history['val_accuracy'], label='Validation')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_saliency(model, image, save_path=None):
    """Plot one raw 28x28 image and sensitivity of its predicted probability.

    Brighter pixels have larger absolute gradients. Values are scaled within
    this image; they show local sensitivity, not whether a pixel supports or
    opposes the prediction. Returns the figure.
    """
    #add a batch dimension; the model handles normalization
    inputs = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        predictions = model(inputs, training=False)
        predicted_class = tf.argmax(predictions[0])
        score = tf.gather(predictions[0], predicted_class)

    #measure how much the predicted probability changes with each pixel
    gradients = tape.gradient(score, inputs)
    saliency = np.abs(gradients[0].numpy())
    saliency = saliency / (saliency.max() + 1e-12)

    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original image')
    axes[0].axis('off')

    #overlay the heatmap on the original image
    axes[1].imshow(image, cmap='gray')
    heatmap = axes[1].imshow(saliency, cmap='hot', alpha=0.6, vmin=0, vmax=1)
    axes[1].set_title(f'Predicted: {class_names[int(predicted_class.numpy())]}')
    axes[1].axis('off')
    fig.colorbar(heatmap, ax=axes[1], label='Relative sensitivity')
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig
