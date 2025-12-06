<!--
Project README updated to highlight project owner and Good First Contribution.
-->
# Text Recognition

Maintained by **Shalini Singh** (@itsshaliniS)

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

🚀 Quick path for new contributors: add a small fix or documentation update and open a PR titled `Good first contribution: <short-description>`.

**Owner / Maintainer**
- Shalini Singh — `itsshaliniS` on GitHub

**Good First Contribution**

If you're looking for an easy place to start, check the `issues/` folder in this repo or open the GitHub Issues page and filter by the `good-first-contribution` label. Small, well-scoped tasks include:

- Adding missing documentation (usage examples, screenshots)
- Fixing typos in docs
- Adding a LICENSE file (already added in this repo)

See `CONTRIBUTING.md` for how to submit a PR and the project's contribution guidelines.

## Quickstart

```powershell
git clone https://github.com/vannu07/Text_Recognition.git
cd Text_Recognition
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
python app.py
# open http://127.0.0.1:5000
```

## Where to start

- `docs/USAGE.md` — How to run and train locally
- `CONTRIBUTING.md` — Contribution workflow and PR checklist
- `.github/ISSUE_TEMPLATE/` — Templates to open issues
- `issues/` — Local curated list of small starter tasks

## License

This project uses the MIT License — see `LICENSE` for details.

## Contact

Maintainer: **Shalini Singh** — please open issues or PRs on GitHub for questions or help.
<<<<<<< HEAD
# 🔤 Handwritten Text Recognition (OCR) System

A production-grade, portfolio-worthy Handwritten Text Recognition system using deep learning. This project combines state-of-the-art computer vision techniques with a modern web interface for real-world text extraction from handwritten images.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌟 Features

### ✅ Deep Learning Model
- **CRNN Architecture**: Convolutional Recurrent Neural Network
- **ResNet18 Encoder**: Pre-trained CNN backbone for feature extraction
- **BiLSTM**: Bidirectional LSTM for sequence modeling
- **CTC Loss**: Connectionist Temporal Classification for alignment-free training
- **GPU Optimized**: Efficient training and inference on CUDA-enabled devices

### ✅ MLflow Integration
- **Experiment Tracking**: Automatic logging of metrics, parameters, and artifacts
- **DAGsHub Support**: Remote tracking and collaboration
- **Visualization**: Training curves, loss plots, and CER/WER metrics
- **Model Versioning**: Checkpoint management and best model selection

### ✅ Modern Web Interface
- **Responsive Design**: Clean, mobile-friendly UI
- **Image Upload**: Drag-and-drop or click to upload
- **Real-time Processing**: Instant text extraction
- **Result Management**: Copy, clear, and download results

### ✅ Production-Ready
- **Flask API**: RESTful backend for inference
- **Model Persistence**: Optimized .pkl format for deployment
- **Error Handling**: Robust exception management
- **Health Checks**: API endpoints for monitoring

---

## 📦 Project Structure

```
OCR-Project/
│
├── data/                      # Dataset directory
│   ├── words/                 # IAM word images
│   ├── train.txt             # Training annotations
│   └── val.txt               # Validation annotations
│
├── models/                    # Saved models
│   ├── best_model.pkl        # Best model (pickle format)
│   ├── best_model.pt         # Best model checkpoint
│   └── checkpoint_*.pt       # Epoch checkpoints
│
├── src/                       # Source code
│   ├── model.py              # CRNN architecture
│   ├── train.py              # Training script
│   ├── dataloader.py         # Dataset and data loading
│   ├── utils.py              # Utility functions (CER, WER, preprocessing)
│   └── mlflow_logger.py      # MLflow tracking
│
├── templates/                 # HTML templates
│   └── index.html            # Main web interface
│
├── static/                    # Static files
│   └── style.css             # CSS styling
│
├── app.py                    # Flask backend
├── requirements.txt          # Python dependencies
├── mlflow.yaml              # MLflow configuration
└── README.md                # This file
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM
- 10GB+ disk space

### Installation

1. **Clone or navigate to the project directory**
```bash
cd OCR-Project
```

2. **Create a virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download IAM Dataset** (Optional - for training)

Visit [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) and download the dataset.

Extract to `data/` directory:
```
data/
├── words/
│   ├── a01/
│   ├── a02/
│   └── ...
└── words.txt
```

---

## 🎓 Training the Model

### Basic Training

```bash
cd src
python train.py
```

### With Custom Configuration

Edit `src/train.py` to modify the configuration:

```python
config = {
    'num_epochs': 20,
    'batch_size': 32,
    'learning_rate': 0.001,
    'hidden_size': 256,
    'num_lstm_layers': 2,
    # ... more options
}
```

### Training Output

The training script will:
- ✓ Create model checkpoints in `models/`
- ✓ Log metrics to MLflow
- ✓ Save best model as `best_model.pkl`
- ✓ Generate training curves
- ✓ Calculate CER and WER metrics

---

## 🔬 MLflow & Experiment Tracking

### Local Tracking

MLflow automatically tracks experiments locally:

```bash
# View MLflow UI
mlflow ui
```

Navigate to `http://localhost:5000` to view experiments.

### DAGsHub Integration

1. **Create DAGsHub Account**
   - Visit [DAGsHub](https://dagshub.com)
   - Create a new repository

2. **Configure Credentials**

```bash
# Windows PowerShell
$env:MLFLOW_TRACKING_USERNAME="your_username"
$env:MLFLOW_TRACKING_PASSWORD="your_token"

# Linux/Mac
export MLFLOW_TRACKING_USERNAME=your_username
export MLFLOW_TRACKING_PASSWORD=your_token
```

3. **Update Training Config**

In `src/train.py`:
```python
config = {
    'dagshub_username': 'your_username',
    'dagshub_repo': 'ocr-project',
    # ...
}
```

4. **Run Training**
```bash
python src/train.py
```

View results at: `https://dagshub.com/your_username/ocr-project`

---

## 🌐 Running the Web Application

### Start Flask Server

```bash
python app.py
```

The application will start on `http://localhost:5000`

### Using the Web Interface

1. **Open Browser**: Navigate to `http://localhost:5000`
2. **Upload Image**: Click "Choose Image" or drag & drop
3. **Extract Text**: Click "Extract Text" button
4. **View Results**: Extracted text appears in the result box
5. **Copy/Clear**: Use buttons to manage results

### API Endpoints

#### Predict Text
```bash
POST /predict
Content-Type: multipart/form-data

Form Data:
- file: image file
```

#### Health Check
```bash
GET /health
```

#### Model Info
```bash
GET /info
```

---

## 🧪 Model Architecture

### CRNN Overview

```
Input Image (3, 32, 128)
         ↓
   ResNet18 Encoder
         ↓
   Feature Maps (512, H, W)
         ↓
   Adaptive Pooling
         ↓
   Sequence (W, Batch, 512)
         ↓
   BiLSTM (2 layers)
         ↓
   FC Layer
         ↓
   CTC Decoder
         ↓
   Output Text
```

### Key Components

1. **ResNet18**: Pre-trained CNN for robust feature extraction
2. **BiLSTM**: Captures temporal dependencies in sequences
3. **CTC Loss**: Handles variable-length sequences without alignment
4. **Greedy Decoder**: Fast inference with best-path decoding

### Model Statistics

- **Parameters**: ~13M
- **Input Size**: 32×128 (H×W)
- **Character Set**: 79 classes (letters, digits, punctuation)
- **Inference Time**: ~50ms per image (GPU)

---

## 📊 Evaluation Metrics

### Character Error Rate (CER)

```
CER = (Insertions + Deletions + Substitutions) / Total Characters
```

Lower is better. Target: < 10%

### Word Error Rate (WER)

```
WER = (Insertions + Deletions + Substitutions) / Total Words
```

Lower is better. Target: < 20%

---

## 🚀 Deployment

### Production Deployment with Gunicorn

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

Build and run:
```bash
docker build -t ocr-app .
docker run -p 5000:5000 ocr-app
```

### Cloud Deployment

- **AWS**: Deploy on EC2 with GPU instance
- **Google Cloud**: Use AI Platform or Compute Engine
- **Azure**: Deploy on Azure Machine Learning
- **Heroku**: Use Heroku with buildpacks

---

## 🎯 Use Cases

- ✓ **Document Digitization**: Convert handwritten notes to digital text
- ✓ **Form Processing**: Extract data from filled forms
- ✓ **Historical Archives**: Digitize historical manuscripts
- ✓ **Education**: Assist students with note-taking
- ✓ **Healthcare**: Process medical prescriptions
- ✓ **Banking**: Read handwritten checks and forms

---

## 🛠️ Development

### Testing Model

```bash
cd src
python model.py
```

### Testing Utilities

```bash
cd src
python utils.py
```

### Testing DataLoader

```bash
cd src
python dataloader.py
```

### Testing MLflow Logger

```bash
cd src
python mlflow_logger.py
```

---

## 📈 Performance Tips

### Training Optimization

1. **Batch Size**: Increase for faster training (if GPU memory allows)
2. **Learning Rate**: Use learning rate finder
3. **Data Augmentation**: Enable for better generalization
4. **Mixed Precision**: Use `torch.cuda.amp` for faster training

### Inference Optimization

1. **Model Quantization**: Reduce model size
2. **ONNX Export**: Convert to ONNX for faster inference
3. **Batch Inference**: Process multiple images together
4. **Model Pruning**: Remove unnecessary weights

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License. See `LICENSE` file for details.

---

## 🙏 Acknowledgments

- **IAM Dataset**: [FKI Handwriting Database](https://fki.tic.heia-fr.ch/)
- **PyTorch**: Deep learning framework
- **MLflow**: Experiment tracking
- **DAGsHub**: ML collaboration platform
- **ResNet**: He et al., "Deep Residual Learning"
- **CTC Loss**: Graves et al., "Connectionist Temporal Classification"

---

## 📞 Contact

For questions, suggestions, or collaborations:

- **GitHub Issues**: Open an issue
- **Email**: your.email@example.com
- **LinkedIn**: [Your Profile](https://linkedin.com/in/yourprofile)

---

## 🎓 Portfolio Note

This project demonstrates:

✅ **Deep Learning Expertise**: CRNN, ResNet, BiLSTM, CTC Loss  
✅ **ML Engineering**: Training pipelines, model optimization  
✅ **Experiment Tracking**: MLflow, DAGsHub integration  
✅ **Full-Stack Development**: Flask backend + Modern frontend  
✅ **Production Skills**: Model deployment, API design  
✅ **Code Quality**: Clean architecture, documentation  

Perfect for internships, hackathons, and job applications! 🚀

---

## 📚 References

1. Shi et al., "An End-to-End Trainable Neural Network for Image-based Sequence Recognition"
2. He et al., "Deep Residual Learning for Image Recognition"
3. Graves et al., "Connectionist Temporal Classification"
4. IAM Handwriting Database Documentation

---

**Made with ❤️ using PyTorch, Flask, and MLflow**

=======
Just created a file
>>>>>>> 377a5ab40d99a3735aaa013cada5de55a317511f
