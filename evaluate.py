"""
Evaluation and visualization module for Alzheimer's MRI classification.
Includes confusion matrix, classification report, accuracy/loss plots, and Grad-CAM.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

CLASS_NAMES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
LAST_CONV_LAYER = 'conv5_block3_out'


# ─── Training Curves ──────────────────────────────────────────────────────────

def plot_training_history(history):
    """Plot accuracy and loss curves from training history."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(history.history['acc'], label='Train Accuracy', marker='o')
    axes[0].plot(history.history['val_acc'], label='Val Accuracy', marker='o')
    axes[0].set_title('Training vs Validation Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history['loss'], label='Train Loss', marker='o')
    axes[1].plot(history.history['val_loss'], label='Val Loss', marker='o')
    axes[1].set_title('Training vs Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()


# ─── Test Set Evaluation ──────────────────────────────────────────────────────

def evaluate_model(model, test_ds):
    """
    Evaluate model on test set and print metrics.
    
    Returns:
        Tuple of (y_true, y_pred) numpy arrays
    """
    y_true = []
    y_pred = []

    for imgs, lbls in test_ds:
        preds = model.predict(imgs, verbose=0)
        y_true.extend(np.argmax(lbls.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    return y_true, y_pred


def plot_confusion_matrix(y_true, y_pred, normalize=False):
    """Plot confusion matrix (raw counts or normalized)."""
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized Confusion Matrix'
        fmt = '.2f'
    else:
        title = 'Confusion Matrix'
        fmt = 'd'

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(title)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()


# ─── Grad-CAM Explainability ──────────────────────────────────────────────────

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=LAST_CONV_LAYER, pred_index=None):
    """
    Generate a Grad-CAM heatmap for a given image.
    
    Args:
        img_array: Input image tensor with batch dimension
        model: Trained Keras model
        last_conv_layer_name: Name of the last convolutional layer
        pred_index: Optional class index to explain (defaults to top prediction)
    
    Returns:
        Normalized heatmap as numpy array
    """
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def apply_gradcam(img_numpy, heatmap, alpha=0.4):
    """
    Overlay a Grad-CAM heatmap on an image.
    
    Args:
        img_numpy: Original image as numpy array (H, W, 3)
        heatmap: Normalized heatmap from make_gradcam_heatmap
        alpha: Blending factor (0 = original, 1 = heatmap only)
    
    Returns:
        Superimposed image as PIL Image
    """
    img = img_numpy.astype('uint8')
    heatmap = np.uint8(255 * heatmap)

    jet = plt.get_cmap('jet')
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    return tf.keras.preprocessing.image.array_to_img(superimposed_img)


def visualize_gradcam_batch(model, images, labels, class_names=CLASS_NAMES, num_samples=4):
    """
    Show original images alongside their Grad-CAM explanations.
    
    Args:
        model: Trained Keras model
        images: Batch of images tensor
        labels: Corresponding one-hot labels
        class_names: List of class names
        num_samples: Number of images to visualize
    """
    num_samples = min(num_samples, len(images))
    plt.figure(figsize=(15, 10))

    for i in range(num_samples):
        img_tensor = tf.expand_dims(images[i], 0)
        heatmap = make_gradcam_heatmap(img_tensor, model)
        cam_img = apply_gradcam(images[i], heatmap)

        # Original
        plt.subplot(2, num_samples, i * 2 + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        true_idx = np.argmax(labels[i])
        plt.title(f'Original: {class_names[true_idx]}')
        plt.axis('off')

        # Grad-CAM
        plt.subplot(2, num_samples, i * 2 + 2)
        plt.imshow(cam_img)
        pred_idx = np.argmax(model.predict(img_tensor, verbose=0)[0])
        plt.title(f'Grad-CAM: {class_names[pred_idx]}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('gradcam_results.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_predictions(model, images, labels, class_names=CLASS_NAMES, num_samples=None):
    """Display prediction samples with true vs predicted labels."""
    if num_samples is None:
        num_samples = len(images)
    num_samples = min(num_samples, 8)

    plt.figure(figsize=(15, 10))
    for i in range(num_samples):
        plt.subplot(2, 4, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        pred_idx = np.argmax(model.predict(tf.expand_dims(images[i], 0), verbose=0))
        true_idx = np.argmax(labels[i])
        color = 'green' if pred_idx == true_idx else 'red'
        plt.title(f'True: {class_names[true_idx]}\nPred: {class_names[pred_idx]}', color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
