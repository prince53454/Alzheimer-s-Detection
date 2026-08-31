"""
Alzheimer's Diagnostic AI — Advanced Web Interface
Professional medical-grade UI with Grad-CAM, image preprocessing, and detailed analysis.
"""

import os
import sys
import json
import shutil
import h5py
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
from datetime import datetime

# ─── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE = (176, 208)
CLASS_NAMES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
CLASS_DESCRIPTIONS = {
    'NonDemented': 'No signs of Alzheimer\'s disease detected. Brain structures appear normal.',
    'VeryMildDemented': 'Very early signs of cognitive impairment detected. Subtle changes in brain tissue.',
    'MildDemented': 'Mild Alzheimer\'s indicators present. Noticeable changes in brain structure.',
    'ModerateDemented': 'Moderate Alzheimer\'s disease detected. Significant brain tissue changes.'
}
CLASS_SEVERITY = {
    'NonDemented': 0,
    'VeryMildDemented': 1,
    'MildDemented': 2,
    'ModerateDemented': 3
}
CLASS_COLORS = {
    'NonDemented': '#27ae60',
    'VeryMildDemented': '#f39c12',
    'MildDemented': '#e67e22',
    'ModerateDemented': '#e74c3c'
}
LAST_CONV_LAYER = 'conv5_block3_out'

# ─── Setup ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
MODEL_PATH = os.path.join(SCRIPT_DIR, 'resnet_normalized.h5')

# ─── Model Loading ────────────────────────────────────────────────────────────
def _patch_h5_compat(path):
    with h5py.File(path, 'r+') as f:
        config = json.loads(f.attrs.get('model_config', '{}'))
        patched = False
        def strip_quantization(cfg):
            nonlocal patched
            if isinstance(cfg, dict):
                if 'quantization_config' in cfg:
                    cfg.pop('quantization_config')
                    patched = True
                for v in cfg.values():
                    strip_quantization(v)
            elif isinstance(cfg, list):
                for item in cfg:
                    strip_quantization(item)
        strip_quantization(config)
        if patched:
            shutil.copy2(path, path + '.bak')
            f.attrs['model_config'] = json.dumps(config).encode('utf-8')

print(f"Loading model from: {MODEL_PATH}")
loaded_model = None
if os.path.exists(MODEL_PATH):
    try:
        loaded_model = tf.keras.models.load_model(MODEL_PATH)
    except (TypeError, ValueError):
        _patch_h5_compat(MODEL_PATH)
        loaded_model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
else:
    print(f"Warning: Model not found at {MODEL_PATH}")


# ─── Grad-CAM ─────────────────────────────────────────────────────────────────
def make_gradcam_heatmap(img_array, model, last_conv_layer_name=LAST_CONV_LAYER):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = last_conv_layer_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def apply_gradcam(img_numpy, heatmap, alpha=0.45):
    heatmap = np.uint8(255 * heatmap)
    jet = plt.get_cmap('jet')
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_numpy.shape[1], img_numpy.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    superimposed = jet_heatmap * alpha + img_numpy
    return tf.keras.preprocessing.image.array_to_img(superimposed)


# ─── Image Preprocessing Visualization ────────────────────────────────────────
def create_preprocessing_preview(image):
    """Create a side-by-side view: Original -> Resized -> Normalized."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor('#1a1a2e')

    # Original
    axes[0].imshow(image.astype('uint8'))
    axes[0].set_title('Original', color='white', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    axes[0].set_facecolor('#1a1a2e')

    # Resized
    resized = np.array(Image.fromarray(image.astype('uint8')).resize((IMG_SIZE[1], IMG_SIZE[0])))
    axes[1].imshow(resized)
    axes[1].set_title(f'Resized {IMG_SIZE}', color='white', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    axes[1].set_facecolor('#1a1a2e')

    # Normalized histogram
    normalized = resized.astype(np.float32) / 255.0
    axes[2].hist(normalized.flatten(), bins=50, color='#00d4ff', alpha=0.7, edgecolor='none')
    axes[2].set_title('Pixel Distribution', color='white', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Pixel Value (0-1)', color='#aaa')
    axes[2].set_ylabel('Frequency', color='#aaa')
    axes[2].set_facecolor('#1a1a2e')
    axes[2].tick_params(colors='#aaa')
    for spine in axes[2].spines.values():
        spine.set_color('#333')

    plt.tight_layout()
    buf = plt.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def create_confidence_chart(pred_probs):
    """Create a horizontal bar chart of class probabilities."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    y_pos = np.arange(len(CLASS_NAMES))
    colors = [CLASS_COLORS[name] for name in CLASS_NAMES]
    probs = [pred_probs[i] * 100 for i in range(len(CLASS_NAMES))]

    bars = ax.barh(y_pos, probs, color=colors, height=0.6, edgecolor='none')

    # Add percentage labels
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f'{prob:.1f}%', va='center', color='white', fontsize=11, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(CLASS_NAMES, color='white', fontsize=11)
    ax.set_xlabel('Probability (%)', color='#aaa', fontsize=11)
    ax.set_xlim(0, 110)
    ax.tick_params(axis='x', colors='#aaa')
    for spine in ax.spines.values():
        spine.set_color('#333')
    ax.invert_yaxis()

    plt.tight_layout()
    buf = plt.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def create_severity_gauge(prediction_idx):
    """Create a visual severity gauge."""
    fig, ax = plt.subplots(figsize=(6, 1.5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    # Background bar
    ax.barh(0, 100, height=0.5, color='#2a2a4a', edgecolor='none')

    # Colored segments
    segments = [
        (0, 25, '#27ae60', 'Normal'),
        (25, 50, '#f39c12', 'Very Mild'),
        (50, 75, '#e67e22', 'Mild'),
        (75, 100, '#e74c3c', 'Moderate'),
    ]
    for start, end, color, label in segments:
        ax.barh(0, end - start, left=start, height=0.5, color=color, alpha=0.3, edgecolor='none')

    # Indicator
    severity = CLASS_SEVERITY[CLASS_NAMES[prediction_idx]]
    position = (severity + 0.5) * 25
    ax.plot(position, 0, 'v', color='white', markersize=15, zorder=5)
    ax.text(position, 0.55, '▼', ha='center', va='bottom', color='white', fontsize=8)

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 1.0)
    ax.axis('off')

    # Labels
    for i, (name, color) in enumerate(CLASS_COLORS.items()):
        ax.text(i * 25 + 12.5, -0.4, name.replace('Demented', '').replace('Non', 'Non-\n'),
                ha='center', va='top', color='#aaa', fontsize=8)

    plt.tight_layout()
    buf = plt.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# ─── Full Analysis ────────────────────────────────────────────────────────────
def full_analysis(image):
    if image is None:
        return None, None, "No image uploaded.", None, None, None
    if loaded_model is None:
        return None, None, "### Error: Model not loaded.", None, None, None

    # Preprocess
    img_resized = tf.image.resize(image, IMG_SIZE)
    img_batch = tf.expand_dims(img_resized, axis=0)

    # Predict
    preds = loaded_model.predict(img_batch, verbose=0)
    idx = np.argmax(preds[0])
    label = CLASS_NAMES[idx]
    conf = preds[0][idx] * 100

    # Grad-CAM
    heatmap = make_gradcam_heatmap(img_batch, loaded_model)
    cam_image = apply_gradcam(image, heatmap)

    # Preprocessing preview
    preprocess_img = create_preprocessing_preview(image)

    # Confidence chart
    confidence_chart = create_confidence_chart(preds[0])

    # Severity gauge
    severity_img = create_severity_gauge(idx)

    # Result text
    severity_text = ['Normal', 'Very Mild', 'Mild', 'Moderate'][CLASS_SEVERITY[label]]
    color = CLASS_COLORS[label]
    result_markdown = f"""
### Diagnosis: **{label}**
**Severity Level:** {severity_text}
**AI Confidence:** {conf:.2f}%
**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
{CLASS_DESCRIPTIONS[label]}
"""
    return cam_image, preprocess_img, result_markdown, confidence_chart, severity_img, label


# ─── Batch Analysis ───────────────────────────────────────────────────────────
def batch_analysis(images):
    if not images:
        return "No images uploaded."
    results = []
    for i, img in enumerate(images):
        if img is None:
            continue
        img_resized = tf.image.resize(img, IMG_SIZE)
        img_batch = tf.expand_dims(img_resized, axis=0)
        preds = loaded_model.predict(img_batch, verbose=0)
        idx = np.argmax(preds[0])
        label = CLASS_NAMES[idx]
        conf = preds[0][idx] * 100
        results.append(f"**Image {i+1}:** {label} ({conf:.1f}%)")

    summary = f"## Batch Analysis Results ({len(results)} images)\n\n" + "\n\n".join(results)

    # Count predictions
    counts = {name: 0 for name in CLASS_NAMES}
    for r in results:
        for name in CLASS_NAMES:
            if name in r:
                counts[name] += 1
    summary += "\n\n---\n### Distribution\n"
    for name, count in counts.items():
        summary += f"- {name}: {count}\n"

    return summary


# ─── Gradio Interface ─────────────────────────────────────────────────────────

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.gradio-container {
    font-family: 'Inter', sans-serif !important;
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%) !important;
}
.main-header {
    text-align: center;
    padding: 30px 0;
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border-radius: 15px;
    margin-bottom: 20px;
    border: 1px solid rgba(0, 212, 255, 0.2);
}
.main-header h1 {
    background: linear-gradient(135deg, #00d4ff, #8b5cf6, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5em;
    font-weight: 700;
    margin: 0;
}
.main-header p {
    color: #8892b0;
    font-size: 1.1em;
    margin-top: 8px;
}
.feature-card {
    background: rgba(26, 26, 46, 0.8);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
}
.analysis-btn {
    background: linear-gradient(135deg, #00d4ff, #8b5cf6) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1.1em !important;
    padding: 12px 24px !important;
    border: none !important;
    border-radius: 10px !important;
    transition: all 0.3s !important;
}
.analysis-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0, 212, 255, 0.3) !important;
}
.result-box {
    background: rgba(26, 26, 46, 0.9);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 12px;
    padding: 20px;
    color: #e8eaff;
}
.info-card {
    background: rgba(0, 212, 255, 0.05);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 10px;
    padding: 15px;
    margin: 5px 0;
}
.footer-text {
    text-align: center;
    color: #5a6080;
    font-size: 0.9em;
    padding: 20px;
}
"""

with gr.Blocks(css=custom_css, title="Alzheimer's Diagnostic AI", theme=gr.themes.Soft()) as iface:

    # Header
    gr.HTML("""
    <div class="main-header">
        <h1>🧠 Alzheimer's Diagnostic AI Portal</h1>
        <p>Advanced Brain MRI Analysis with Deep Learning & Grad-CAM Explainability</p>
    </div>
    """)

    # Tab layout
    with gr.Tabs():
        # ─── Tab 1: Single Image Analysis ───
        with gr.Tab("🔬 Single Image Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Upload Brain MRI")
                    gr.Markdown("Upload a brain MRI scan for AI-powered Alzheimer's stage classification.")
                    image_input = gr.Image(type='numpy', label='Brain MRI Image', height=300)
                    analyze_btn = gr.Button('🚀 Run Full Analysis', variant='primary', elem_classes=['analysis-btn'])

                    gr.Markdown("### Model Info")
                    gr.HTML("""
                    <div class="info-card">
                        <strong>Architecture:</strong> ResNet50 (Transfer Learning)<br>
                        <strong>Input Size:</strong> 176 × 208 pixels<br>
                        <strong>Classes:</strong> 4 Alzheimer's stages<br>
                        <strong>Explainability:</strong> Grad-CAM
                    </div>
                    """)

                with gr.Column(scale=2):
                    gr.Markdown("### Analysis Results")
                    with gr.Row():
                        gradcam_output = gr.Image(label="Grad-CAM Attention Map", height=300)
                        preprocess_output = gr.Image(label="Preprocessing Pipeline", height=300)

                    result_text = gr.Markdown(value="*Upload an image and click Analyze...*", elem_classes=['result-box'])

                    with gr.Row():
                        confidence_output = gr.Image(label="Class Probabilities", height=250)
                        severity_output = gr.Image(label="Severity Assessment", height=250)

            analyze_btn.click(
                fn=full_analysis,
                inputs=image_input,
                outputs=[gradcam_output, preprocess_output, result_text, confidence_output, severity_output, gr.Textbox(visible=False)]
            )

        # ─── Tab 2: Batch Analysis ───
        with gr.Tab("📊 Batch Analysis"):
            gr.Markdown("### Upload Multiple MRI Scans")
            gr.Markdown("Analyze multiple images at once and get a summary report with distribution statistics.")
            batch_images = gr.File(file_count="multiple", file_types=["image"], label="Upload MRI Images")
            batch_btn = gr.Button('📊 Analyze Batch', variant='primary', elem_classes=['analysis-btn'])
            batch_results = gr.Markdown(value="*Upload images to begin batch analysis...*")

            batch_btn.click(fn=batch_analysis, inputs=batch_images, outputs=batch_results)

        # ─── Tab 3: About ───
        with gr.Tab("ℹ️ About This Model"):
            gr.Markdown("""
            ## About Alzheimer's Diagnostic AI

            ### How It Works
            This system uses **ResNet50**, a deep convolutional neural network pre-trained on ImageNet,
            fine-tuned to classify brain MRI scans into 4 Alzheimer's disease stages.

            ### Classes
            | Class | Description | Severity |
            |-------|-------------|----------|
            | 🟢 **NonDemented** | Healthy brain, no signs of Alzheimer's | None |
            | 🟡 **VeryMildDemented** | Very early cognitive impairment | Low |
            | 🟠 **MildDemented** | Mild Alzheimer's indicators | Medium |
            | 🔴 **ModerateDemented** | Moderate Alzheimer's disease | High |

            ### Grad-CAM Explainability
            **Grad-CAM** (Gradient-weighted Class Activation Mapping) visualizes which regions
            of the brain MRI the model focuses on when making predictions. The heatmap overlay
            shows:
            - **Red/warm regions**: Areas most influential to the prediction
            - **Blue/cool regions**: Less relevant areas

            ### Key Features
            - ✅ Transfer Learning with ResNet50
            - ✅ Class-weighted training for imbalanced data
            - ✅ Grad-CAM explainability
            - ✅ Real-time preprocessing visualization
            - ✅ Batch analysis support
            - ✅ Severity assessment gauge

            ### Technical Details
            - **Input**: Brain MRI (any resolution, resized to 176×208)
            - **Architecture**: ResNet50 (frozen base) + Dense(4, softmax)
            - **Training**: 10 epochs, Adam optimizer, class-weighted loss
            - **Dataset**: ~33,000 augmented + original brain MRI scans
            """)

    # Footer
    gr.HTML("""
    <div class="footer-text">
        <p>Alzheimer's Diagnostic AI • Built with TensorFlow, ResNet50 & Gradio</p>
        <p>⚠️ This tool is for research purposes only. Always consult a medical professional.</p>
    </div>
    """)


if __name__ == '__main__':
    # Auto-detect hosting environment
    port = int(os.environ.get('PORT', 8080))
    is_cloud = os.environ.get('SPACE_ID') is not None or os.environ.get('RENDER') or os.environ.get('PORT')
    if is_cloud:
        iface.launch(server_name='0.0.0.0', server_port=port)
    else:
        iface.launch(server_name='127.0.0.1', server_port=port, share=False)
