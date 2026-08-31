"""
Data loading and preparation module for Alzheimer's MRI classification.
Handles dataset extraction, loading, preprocessing, and class weight calculation.
"""

import os
import zipfile
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight

# ─── Global Constants ────────────────────────────────────────────────────────
IMG_SIZE = (176, 208)
BATCH_SIZE = 8
AUTOTUNE = tf.data.experimental.AUTOTUNE
SEED = 54
NUM_CLASSES = 4
CLASS_NAMES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']


def extract_dataset(zip_path, extract_dir="DATASEt"):
    """
    Extract dataset from a zip file.
    
    Args:
        zip_path: Path to the dataset zip file
        extract_dir: Directory to extract into
    
    Returns:
        Tuple of (TRAIN_DIR, TEST_DIR) paths
    """
    extract_base_path = extract_dir

    if not os.path.exists(extract_base_path) or not os.listdir(extract_base_path):
        print(f"Extracting dataset to {extract_base_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_base_path)
        print(f"Extraction complete.")
    else:
        print(f"Dataset already extracted at {extract_base_path}.")

    # Dynamically determine the actual root of the dataset
    actual_dataset_root = extract_base_path
    contents = os.listdir(extract_base_path)
    if len(contents) == 1 and os.path.isdir(os.path.join(extract_base_path, contents[0])):
        actual_dataset_root = os.path.join(extract_base_path, contents[0])
        print(f"Adjusted dataset root to: {actual_dataset_root}")

    train_dir = os.path.join(actual_dataset_root, "AugmentedAlzheimerDataset")
    test_dir = os.path.join(actual_dataset_root, "OriginalDataset")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    print(f"Training dir: {train_dir}")
    print(f"Test dir: {test_dir}")
    return train_dir, test_dir


def create_datasets(train_dir, test_dir, batch_size=BATCH_SIZE):
    """
    Create TensorFlow datasets from directory structure.
    
    Args:
        train_dir: Path to training images directory
        test_dir: Path to test images directory
        batch_size: Batch size for datasets
    
    Returns:
        Tuple of (train_ds, val_ds, test_ds) raw datasets
    """
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        verbose=False
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        verbose=False
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        verbose=False
    )

    train_ds.class_names = CLASS_NAMES
    val_ds.class_names = CLASS_NAMES
    test_ds.class_names = CLASS_NAMES

    return train_ds, val_ds, test_ds


def compute_class_weights(train_ds):
    """
    Compute balanced class weights to handle class imbalance.
    
    Args:
        train_ds: Training dataset
    
    Returns:
        Dictionary mapping class index to weight
    """
    all_labels = []
    for _, labels in train_ds:
        all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels)
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(all_labels),
        y=all_labels
    )
    class_weights = {i: w for i, w in enumerate(weights)}

    print("Class weights:")
    for idx, weight in class_weights.items():
        print(f"  {CLASS_NAMES[idx]}: {weight:.4f}")

    return class_weights


def prepare_datasets(train_ds, val_ds, test_ds):
    """
    Apply one-hot encoding and prefetching to datasets.
    
    Args:
        train_ds, val_ds, test_ds: Raw TensorFlow datasets
    
    Returns:
        Tuple of (train_prefetch, val_prefetch, test_prefetch, sample_images, sample_labels)
    """
    def one_hot_label(image, label):
        return image, tf.one_hot(label, NUM_CLASSES)

    train_pref = train_ds.map(one_hot_label, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
    val_pref = val_ds.map(one_hot_label, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
    test_pref = test_ds.map(one_hot_label, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

    # Get a sample batch for visualization
    sample_images, sample_labels = None, None
    for img, lbl in test_pref.take(1):
        sample_images = img
        sample_labels = lbl

    print(f"Datasets prepared. Train batches: {len(train_pref)}, Val batches: {len(val_pref)}, Test batches: {len(test_pref)}")
    return train_pref, val_pref, test_pref, sample_images, sample_labels
