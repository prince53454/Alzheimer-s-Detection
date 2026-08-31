"""
Model definition and training module for Alzheimer's MRI classification.
Uses ResNet50 transfer learning with class-weighted training.
"""

import os
import json
import h5py
import shutil
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications.resnet50 import ResNet50

# ─── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE = (176, 208)
NUM_CLASSES = 4
MODEL_PATH = 'resnet_normalized.h5'
LAST_CONV_LAYER = 'conv5_block3_out'


def build_resnet50():
    """
    Build a ResNet50 model with frozen base layers and a custom classification head.
    
    Returns:
        Compiled Keras Model
    """
    base = ResNet50(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        weights='imagenet',
        include_top=False
    )
    base.trainable = False

    x = layers.Flatten()(base.output)
    prediction = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base.input, outputs=prediction)
    return model


def compile_model(model):
    """Compile the model with Adam optimizer and categorical crossentropy loss."""
    model.compile(
        optimizer='adam',
        loss=tf.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.AUC(name='auc'), 'acc']
    )
    return model


def train_model(model, train_ds, val_ds, class_weights=None, epochs=10):
    """
    Train the model with optional class weights for handling imbalanced data.
    
    Args:
        model: Compiled Keras model
        train_ds: Training dataset (prefetched, one-hot labels)
        val_ds: Validation dataset (prefetched, one-hot labels)
        class_weights: Dictionary mapping class index to weight
        epochs: Number of training epochs
    
    Returns:
        Training history object
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]

    fit_kwargs = {
        'x': train_ds,
        'epochs': epochs,
        'validation_data': val_ds,
        'callbacks': callbacks,
    }
    if class_weights:
        fit_kwargs['class_weight'] = class_weights

    history = model.fit(**fit_kwargs)
    return history


def save_model(model, path=MODEL_PATH):
    """Save the trained model to disk."""
    model.save(path)
    print(f"Model saved to {path}")


def _patch_h5_compat(path):
    """
    Patch .h5 model files saved with newer Keras that include
    'quantization_config' which older Keras doesn't recognize.
    """
    with h5py.File(path, 'r+') as f:
        config = json.loads(f.attrs.get('model_config', '{}'))
        has_compat_issue = False

        def strip_quantization(cfg):
            nonlocal has_compat_issue
            if isinstance(cfg, dict):
                if 'quantization_config' in cfg:
                    cfg.pop('quantization_config')
                    has_compat_issue = True
                for v in cfg.values():
                    strip_quantization(v)
            elif isinstance(cfg, list):
                for item in cfg:
                    strip_quantization(item)

        strip_quantization(config)
        if has_compat_issue:
            shutil.copy2(path, path + '.bak')
            f.attrs['model_config'] = json.dumps(config).encode('utf-8')
            print("Patched model config for compatibility.")


def load_model(path=MODEL_PATH):
    """Load a trained model from disk, patching compatibility issues if needed."""
    if os.path.exists(path):
        try:
            model = tf.keras.models.load_model(path)
        except (TypeError, ValueError):
            _patch_h5_compat(path)
            model = tf.keras.models.load_model(path)
    else:
        raise FileNotFoundError(f"Model not found: {path}")
    print(f"Model loaded from {path}")
    return model


def get_or_train_model(train_ds, val_ds, class_weights=None, epochs=10):
    """
    Load an existing model or train a new one if not found.
    
    Args:
        train_ds: Training dataset
        val_ds: Validation dataset
        class_weights: Optional class weights dictionary
        epochs: Training epochs (only used if training)
    
    Returns:
        Tuple of (model, history) where history is None if model was loaded
    """
    if os.path.exists(MODEL_PATH):
        model = load_model()
        return model, None
    
    print("No saved model found. Training new model...")
    model = build_resnet50()
    model = compile_model(model)
    history = train_model(model, train_ds, val_ds, class_weights, epochs)
    save_model(model)
    return model, history
