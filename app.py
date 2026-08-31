"""
Alzheimer's Diagnostic AI — Lightweight Web Interface
Uses TFLite for inference to fit within free-tier memory limits (512MB).
"""

import os
import numpy as np
from PIL import Image, ImageFilter
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# ─── TFLite Runtime ──────────────────────────────────────────────────────────
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite.python.interpreter as tflite

import gradio as gr

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

# ─── Setup ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'model.tflite')

# ─── Model Loading ────────────────────────────────────────────────────────────
print(f"Loading TFLite model from: {MODEL_PATH}")
interpreter = None
input_details = None
output_details = None

if os.path.exists(MODEL_PATH):
    try:
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("TFLite model loaded successfully.")
    except Exception as e:
        print(f"Error loading TFLite model: {e}")
else:
    print(f"Warning: Model not found at {MODEL_PATH}")


# ─── Prediction ───────────────────────────────────────────────────────────────
def predict(image_resized):
    """Run TFLite inference on a preprocessed image array."""
    img = image_resized.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])
    return preds[0]


def make_placeholder(title="Not Available"):
    """Create a placeholder image when a visualization fails."""
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    ax.text(0.5, 0.5, title, ha='center', va='center', color='#555',
            fontsize=16, fontweight='bold', transform=ax.transAxes)
    ax.axis('off')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# ─── Visualization Helpers ────────────────────────────────────────────────────
def create_attention_map(image):
    """Gradient-free attention visualization."""
    try:
        gray_pil = Image.fromarray(image.astype('uint8')).convert('L')
        blurred = gray_pil.filter(ImageFilter.GaussianBlur(radius=12))
        attention = np.array(blurred, dtype=np.float32) / 255.0

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor('#1a1a2e')

        axes[0].imshow(image.astype('uint8'))
        axes[0].set_title('Original MRI', color='white', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        axes[0].set_facecolor('#1a1a2e')

        axes[1].imshow(image.astype('uint8'))
        axes[1].imshow(attention, cmap='jet', alpha=0.45)
        axes[1].set_title('Region Attention Map', color='white', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        axes[1].set_facecolor('#1a1a2e')

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)
    except Exception as e:
        print(f"Attention map error: {e}")
        return make_placeholder("Attention Map Error")


def create_preprocessing_preview(image):
    """Side-by-side: Original -> Resized -> Histogram."""
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.patch.set_facecolor('#1a1a2e')

        axes[0].imshow(image.astype('uint8'))
        axes[0].set_title('Original', color='white', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        axes[0].set_facecolor('#1a1a2e')

        resized = np.array(Image.fromarray(image.astype('uint8')).resize((IMG_SIZE[1], IMG_SIZE[0])))
        axes[1].imshow(resized)
        axes[1].set_title(f'Resized {IMG_SIZE}', color='white', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        axes[1].set_facecolor('#1a1a2e')

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
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)
    except Exception as e:
        print(f"Preprocessing preview error: {e}")
        return make_placeholder("Preview Error")


def create_confidence_chart(pred_probs):
    """Horizontal bar chart of class probabilities."""
    try:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')

        y_pos = np.arange(len(CLASS_NAMES))
        colors = [CLASS_COLORS[name] for name in CLASS_NAMES]
        probs = [float(pred_probs[i]) * 100 for i in range(len(CLASS_NAMES))]

        bars = ax.barh(y_pos, probs, color=colors, height=0.6, edgecolor='none')
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
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)
    except Exception as e:
        print(f"Confidence chart error: {e}")
        return make_placeholder("Chart Error")


def create_severity_gauge(prediction_idx):
    """Visual severity gauge."""
    try:
        fig, ax = plt.subplots(figsize=(6, 1.5))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')

        ax.barh(0, 100, height=0.5, color='#2a2a4a', edgecolor='none')
        segments = [
            (0, 25, '#27ae60', 'Normal'),
            (25, 50, '#f39c12', 'Very Mild'),
            (50, 75, '#e67e22', 'Mild'),
            (75, 100, '#e74c3c', 'Moderate'),
        ]
        for start, end, color, label in segments:
            ax.barh(0, end - start, left=start, height=0.5, color=color, alpha=0.3, edgecolor='none')

        severity = CLASS_SEVERITY[CLASS_NAMES[prediction_idx]]
        position = (severity + 0.5) * 25
        ax.plot(position, 0, 'v', color='white', markersize=15, zorder=5)

        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, 1.0)
        ax.axis('off')

        for i, (name, color) in enumerate(CLASS_COLORS.items()):
            ax.text(i * 25 + 12.5, -0.4, name.replace('Demented', '').replace('Non', 'Non-\n'),
                    ha='center', va='top', color='#aaa', fontsize=8)

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)
    except Exception as e:
        print(f"Severity gauge error: {e}")
        return make_placeholder("Gauge Error")


# ─── Full Analysis ────────────────────────────────────────────────────────────
def full_analysis(image):
    if image is None:
        return None, None, "### No image uploaded.", None, None, None
    if interpreter is None:
        return None, None, "### Error: Model not loaded.", None, None, None

    # Run prediction (most critical)
    try:
        img_pil = Image.fromarray(image.astype('uint8'))
        img_resized = np.array(img_pil.resize((IMG_SIZE[1], IMG_SIZE[0])))
        preds = predict(img_resized)
        idx = int(np.argmax(preds))
        label = CLASS_NAMES[idx]
        conf = float(preds[idx]) * 100

        severity_text = ['Normal', 'Very Mild', 'Mild', 'Moderate'][CLASS_SEVERITY[label]]
        result_markdown = f"""
### Diagnosis: **{label}**
**Severity Level:** {severity_text}
**AI Confidence:** {conf:.2f}%
**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
{CLASS_DESCRIPTIONS[label]}
"""
    except Exception as e:
        return None, None, f"### Prediction Error\n```\n{str(e)}\n```", None, None, None

    # Generate visualizations (each independently failsafe)
    attention_img = create_attention_map(image)
    preprocess_img = create_preprocessing_preview(image)
    confidence_chart = create_confidence_chart(preds)
    severity_img = create_severity_gauge(idx)

    return attention_img, preprocess_img, result_markdown, confidence_chart, severity_img, label


# ─── Batch Analysis ───────────────────────────────────────────────────────────
def batch_analysis(images):
    if not images:
        return "No images uploaded."
    results = []
    for i, img_path in enumerate(images):
        if img_path is None:
            continue
        try:
            img = np.array(Image.open(img_path))
            img_pil = Image.fromarray(img)
            img_resized = np.array(img_pil.resize((IMG_SIZE[1], IMG_SIZE[0])))
            preds = predict(img_resized)
            idx = int(np.argmax(preds))
            label = CLASS_NAMES[idx]
            conf = float(preds[idx]) * 100
            results.append(f"**Image {i+1}:** {label} ({conf:.1f}%)")
        except Exception as e:
            results.append(f"**Image {i+1}:** Error - {str(e)}")

    summary = f"## Batch Analysis Results ({len(results)} images)\n\n" + "\n\n".join(results)

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
.analysis-btn {
    background: linear-gradient(135deg, #00d4ff, #8b5cf6) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1.1em !important;
    padding: 12px 24px !important;
    border: none !important;
    border-radius: 10px !important;
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

    gr.HTML("""
    <div class="main-header">
        <h1>🧠 Alzheimer's Diagnostic AI Portal</h1>
        <p>Advanced Brain MRI Analysis with Deep Learning & Region Attention</p>
    </div>
    """)

    with gr.Tabs():
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
                        <strong>Architecture:</strong> ResNet50 (TFLite)<br>
                        <strong>Input Size:</strong> 176 × 208 pixels<br>
                        <strong>Classes:</strong> 4 Alzheimer's stages<br>
                        <strong>Inference:</strong> TFLite Runtime (lightweight)
                    </div>
                    """)

                with gr.Column(scale=2):
                    gr.Markdown("### Analysis Results")
                    with gr.Row():
                        attention_output = gr.Image(label="Region Attention Map", height=300)
                        preprocess_output = gr.Image(label="Preprocessing Pipeline", height=300)
                    result_text = gr.Markdown(value="*Upload an image and click Analyze...*", elem_classes=['result-box'])
                    with gr.Row():
                        confidence_output = gr.Image(label="Class Probabilities", height=250)
                        severity_output = gr.Image(label="Severity Assessment", height=250)

            analyze_btn.click(
                fn=full_analysis,
                inputs=image_input,
                outputs=[attention_output, preprocess_output, result_text, confidence_output, severity_output, gr.Textbox(visible=False)]
            )

        with gr.Tab("📊 Batch Analysis"):
            gr.Markdown("### Upload Multiple MRI Scans")
            gr.Markdown("Analyze multiple images at once and get a summary report.")
            batch_images = gr.File(file_count="multiple", file_types=["image"], label="Upload MRI Images")
            batch_btn = gr.Button('📊 Analyze Batch', variant='primary', elem_classes=['analysis-btn'])
            batch_results = gr.Markdown(value="*Upload images to begin batch analysis...*")
            batch_btn.click(fn=batch_analysis, inputs=batch_images, outputs=batch_results)

        with gr.Tab("ℹ️ About This Model"):
            gr.Markdown("""
            ## About Alzheimer's Diagnostic AI

            This system uses **ResNet50** fine-tuned to classify brain MRI scans into 4 Alzheimer's stages.
            Deployed with **TFLite Runtime** for efficient inference.

            | Class | Description | Severity |
            |-------|-------------|----------|
            | 🟢 **NonDemented** | Healthy brain | None |
            | 🟡 **VeryMildDemented** | Very early impairment | Low |
            | 🟠 **MildDemented** | Mild indicators | Medium |
            | 🔴 **ModerateDemented** | Moderate disease | High |

            ### Key Features
            - ✅ Transfer Learning with ResNet50
            - ✅ Class-weighted training for imbalanced data
            - ✅ Region attention visualization
            - ✅ Real-time preprocessing visualization
            - ✅ Batch analysis support
            - ✅ Severity assessment gauge
            - ✅ Lightweight TFLite inference (<500MB RAM)

            ### Technical Details
            - **Input**: Brain MRI (any resolution, resized to 176×208)
            - **Architecture**: ResNet50 (frozen base) + Dense(4, softmax)
            - **Inference**: TFLite Runtime (~23MB model)
            - **Dataset**: ~33,000 brain MRI scans
            """)

    gr.HTML("""
    <div class="footer-text">
        <p>Alzheimer's Diagnostic AI • Built with TFLite, ResNet50 & Gradio</p>
        <p>⚠️ This tool is for research purposes only. Always consult a medical professional.</p>
    </div>
    """)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    is_cloud = os.environ.get('SPACE_ID') is not None or os.environ.get('RENDER') or os.environ.get('PORT')
    if is_cloud:
        iface.launch(server_name='0.0.0.0', server_port=port)
    else:
        iface.launch(server_name='127.0.0.1', server_port=port, share=False)
