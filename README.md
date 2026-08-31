# Alzheimer's MRI Classification

Deep learning model for classifying brain MRI scans into Alzheimer's disease stages using **ResNet50 transfer learning** with **Grad-CAM explainability**.

---

## 🔗 Live Detection App (Permanent Link)

### 👉 [Open Alzheimer's Diagnostic AI Portal](https://huggingface.co/spaces/your-username/alzheimer-detection)

> **Replace `your-username` with your actual Hugging Face username after deploying.**
> This link never expires — open it anytime to run detection without regenerating links.

### Quick Deploy to Get Your Permanent Link

1. **Install the Hugging Face CLI** (one time only):
   ```bash
   pip install huggingface_hub
   ```

2. **Login to Hugging Face**:
   ```bash
   huggingface-cli login
   ```
   *(Get your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens))*

3. **Create and deploy your Space**:
   ```bash
   # From this project directory:
   git lfs install
   git clone https://huggingface.co/spaces/YOUR_USERNAME/alzheimer-detection
   cp app.py resnet_normalized.h5 requirements.txt README.md alzheimer-detection/
   cd alzheimer-detection
   git add .
   git commit -m "Deploy Alzheimer's Detection App"
   git push
   ```

4. **Update the link above** in this README with your actual URL:
   `https://huggingface.co/spaces/YOUR_USERNAME/alzheimer-detection`

> 💡 **After deployment, the app is live 24/7 for free. No more expired links!**

---

## Classes
| Class | Description |
|-------|-------------|
| NonDemented | Healthy control |
| VeryMildDemented | Very mild cognitive impairment |
| MildDemented | Mild Alzheimer's |
| ModerateDemented | Moderate Alzheimer's |

## Project Structure
```
Alzheimer's detection/
├── data.py          # Dataset loading, preprocessing, class weights
├── model.py         # ResNet50 model definition and training
├── evaluate.py      # Evaluation metrics, Grad-CAM, visualizations
├── app.py           # Gradio web interface for inference
├── train.py         # Main training pipeline (entry point)
├── requirements.txt # Python dependencies
└── README.md        # This file
```

## Setup

```bash
pip install -r requirements.txt
```

## Training

### Option 1: From zip file
```bash
python train.py --zip_path /path/to/archive.zip
```

### Option 2: From extracted directories
```bash
python train.py --train_dir /path/to/AugmentedAlzheimerDataset --test_dir /path/to/OriginalDataset
```

### Option 3: Default paths (extract zip to DATASEt/ first)
```bash
python train.py
```

## Running the Web App Locally

After training (or placing `resnet_normalized.h5` in this directory):

```bash
python app.py
```

This launches a Gradio interface at **http://127.0.0.1:7870** where you can upload brain MRI images and get predictions with Grad-CAM heatmaps.

> 💡 For a **permanent, always-accessible link**, deploy to Hugging Face Spaces using the instructions above.

## Google Colab

Upload `train.ipynb` to Google Colab and run all cells. The notebook handles:
- Google Drive mounting and dataset extraction
- Model training with class-weighted loss
- Evaluation with confusion matrix and classification report
- Grad-CAM visualizations
- Gradio web app for inference

## Key Features
- **ResNet50 Transfer Learning**: Pre-trained on ImageNet, fine-tuned on brain MRIs
- **Class-Weighted Training**: Handles severe class imbalance (ModerateDemented is rare)
- **Grad-CAM Explainability**: Visual heatmaps showing which brain regions the model focuses on
- **Gradio Web Interface**: Easy-to-use browser-based prediction tool

## Results
- **Overall Accuracy**: ~94%
- **ModerateDemented Recall**: 1.00 (no missed severe cases)
- Key challenge: distinguishing VeryMildDemented vs NonDemented
