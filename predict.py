"""
Quick single-image prediction script.
Usage: python predict.py /path/to/mri_image.jpg
"""

import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from evaluate import make_gradcam_heatmap, apply_gradcam

CLASS_NAMES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
MODEL_PATH = 'resnet_normalized.h5'
IMG_SIZE = (176, 208)


def predict_single(image_path):
    """Predict the class of a single MRI image."""
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Load and preprocess
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_batch = tf.expand_dims(img_array, axis=0)
    
    # Predict
    preds = model.predict(img_batch, verbose=0)[0]
    pred_idx = np.argmax(preds)
    label = CLASS_NAMES[pred_idx]
    conf = preds[pred_idx] * 100
    
    print(f"\nPrediction: {label} ({conf:.2f}% confidence)")
    print("All probabilities:")
    for i, name in enumerate(CLASS_NAMES):
        bar = '#' * int(preds[i] * 40)
        print(f"  {name:25s} {preds[i]*100:6.2f}% {bar}")
    
    # Grad-CAM
    heatmap = make_gradcam_heatmap(img_batch, model)
    cam_img = apply_gradcam(img_array, heatmap)
    
    # Display
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img_array.astype('uint8'))
    axes[0].set_title('Original MRI')
    axes[0].axis('off')
    axes[1].imshow(cam_img)
    axes[1].set_title(f'Grad-CAM: {label} ({conf:.1f}%)')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py /path/to/mri_image.jpg")
        sys.exit(1)
    predict_single(sys.argv[1])
