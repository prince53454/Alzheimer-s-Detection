# Alzheimer's MRI Classification

Deep learning model for classifying brain MRI scans into Alzheimer's disease stages using **ResNet50 transfer learning** with **Grad-CAM explainability**.

---

## 🔗 Live Detection App (Permanent Link)

### 👉 [Open Alzheimer's Diagnostic AI Portal](https://alzheimer-s-detection-ynt8.onrender.com)

> **Permanent link — open it anytime to run detection without regenerating links!**

### Quick Deploy to Get Your Permanent Link (Free)

1. **Go to [render.com](https://render.com)** and sign up (free)
2. **Click "New +" → "Web Service"**
3. **Connect your GitHub repo**: `https://github.com/prince53454/Alzheimer-s-Detection`
4. **Configure the service**:
   - **Name**: `alzheimer-detection`
   - **Runtime**: `Docker`
   - **Plan**: `Free`
5. **Click "Create Web Service"** and wait ~3-5 minutes — done!

> 💡 **The app is live 24/7 for free. No more expired links!**
> ⚠️ Free tier spins down after 15 min of inactivity — first load takes ~30 sec.

### Source Code
GitHub: https://github.com/prince53454/Alzheimer-s-Detection

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
