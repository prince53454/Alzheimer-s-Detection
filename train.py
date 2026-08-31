"""
Main training pipeline for Alzheimer's MRI classification.
Run this script to train the ResNet50 model from scratch.

Usage:
    python train.py --zip_path /path/to/archive.zip
    python train.py --train_dir /path/to/AugmentedAlzheimerDataset --test_dir /path/to/OriginalDataset
"""

import os
import sys
import argparse
import warnings
import numpy as np
import tensorflow as tf

# Suppress warnings
warnings.simplefilter('ignore')
tf.random.set_seed(54)
np.random.seed(54)

from data import extract_dataset, create_datasets, compute_class_weights, prepare_datasets, CLASS_NAMES
from model import build_resnet50, compile_model, train_model, save_model
from evaluate import (plot_training_history, evaluate_model, plot_confusion_matrix,
                      visualize_gradcam_batch, visualize_predictions)


def main():
    parser = argparse.ArgumentParser(description='Train Alzheimer\'s MRI classifier')
    parser.add_argument('--zip_path', type=str, help='Path to dataset zip file')
    parser.add_argument('--train_dir', type=str, help='Direct path to training images directory')
    parser.add_argument('--test_dir', type=str, help='Direct path to test images directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--model_output', type=str, default='resnet_normalized.h5', help='Output model path')
    args = parser.parse_args()

    print("=" * 60)
    print("  ALZHEIMER'S MRI CLASSIFICATION - TRAINING PIPELINE")
    print("=" * 60)

    # ── Step 1: Get dataset paths ──
    if args.train_dir and args.test_dir:
        train_dir = args.train_dir
        test_dir = args.test_dir
    elif args.zip_path:
        print("\n[1/6] Extracting dataset...")
        train_dir, test_dir = extract_dataset(args.zip_path)
    else:
        # Try default relative paths
        default_train = os.path.join("DATASEt", "AugmentedAlzheimerDataset")
        default_test = os.path.join("DATASEt", "OriginalDataset")
        if os.path.exists(default_train) and os.path.exists(default_test):
            train_dir, test_dir = default_train, default_test
        else:
            print("Error: Provide --zip_path or --train_dir/--test_dir")
            sys.exit(1)

    # ── Step 2: Create datasets ──
    print("\n[2/6] Loading datasets...")
    train_ds, val_ds, test_ds = create_datasets(train_dir, test_dir, batch_size=args.batch_size)

    print("\nDataset class distribution:")
    for name in CLASS_NAMES:
        count = len([x for x in os.listdir(os.path.join(train_dir, name))])
        print(f"  {name}: {count} images")

    # ── Step 3: Compute class weights ──
    print("\n[3/6] Computing class weights...")
    class_weights = compute_class_weights(train_ds)

    # ── Step 4: Prepare datasets ──
    print("\n[4/6] Preparing datasets with one-hot encoding and prefetching...")
    train_pref, val_pref, test_pref, sample_images, sample_labels = prepare_datasets(train_ds, val_ds, test_ds)

    # ── Step 5: Build and train model ──
    print(f"\n[5/6] Training ResNet50 for {args.epochs} epochs...")
    model = build_resnet50()
    model = compile_model(model)
    model.summary()

    history = train_model(model, train_pref, val_pref, class_weights, epochs=args.epochs)
    save_model(model, args.model_output)

    # ── Step 6: Evaluate and visualize ──
    print("\n[6/6] Evaluating model...")
    
    if history:
        plot_training_history(history)

    y_true, y_pred = evaluate_model(model, test_pref)
    plot_confusion_matrix(y_true, y_pred, normalize=False)
    plot_confusion_matrix(y_true, y_pred, normalize=True)
    visualize_predictions(model, sample_images, sample_labels)
    visualize_gradcam_batch(model, sample_images, sample_labels)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  Model saved to: {args.model_output}")
    print("  Run 'python app.py' to launch the Gradio web interface")
    print("=" * 60)


if __name__ == '__main__':
    main()
