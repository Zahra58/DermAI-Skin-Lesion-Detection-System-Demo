
# 🔬 DermAI - Skin Lesion Detection System

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-FF4B4B.svg)](https://streamlit.io/)
[![Live Demo](https://img.shields.io/badge/demo-live-success.svg)](https://huggingface.co/spaces/Zahra58/dermai-skin-lesion-detector)

**AI-powered educational web application for skin lesion classification | 80.18% validation accuracy on ISIC 2018 dataset**

<p align="center">
  <img src="https://img.shields.io/badge/Status-Production-success" alt="Status">
  <img src="https://img.shields.io/badge/ML-Computer%20Vision-blue" alt="ML">
  <img src="https://img.shields.io/badge/Deployment-HuggingFace-yellow" alt="Deployment">
</p>

-----

## Live Demo

**Try it now:** <https://huggingface.co/spaces/Zahra58/dermai-skin-lesion-detector>

Upload a dermatoscopic image and get instant AI-powered classification across 7 skin lesion categories.

> ⚠️ **Educational Use Only** - This is a demonstration project and not intended for clinical diagnosis. Always consult qualified healthcare professionals for medical advice.

-----

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Performance](#performance)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Technologies](#technologies)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

-----

## Overview

DermAI is an end-to-end machine learning project that demonstrates the complete ML lifecycle: from data preprocessing and model training to production deployment. The system classifies dermatoscopic images into 7 diagnostic categories using a custom Convolutional Neural Network.

### Key Highlights

- ✅ **80.18% validation accuracy** on ISIC 2018 medical imaging benchmark
- ✅ **Real-time inference** (<100ms per prediction)
- ✅ **Lightweight model** (0.56 MB) optimized for deployment
- ✅ **Privacy-first architecture** - no data storage
- ✅ **Production-ready** deployment on HuggingFace Spaces
- ✅ **Handles severe class imbalance** (67:1 ratio) using weighted sampling

-----

## Features

### Machine Learning

- Custom SimpleBNConv CNN architecture
- 5 convolutional blocks with batch normalization
- Weighted random sampling for class imbalance
- Data augmentation pipeline (rotation, flip, color jitter)
- Transfer learning ready architecture

### Web Application

- Real-time image upload and prediction
- Interactive probability visualization
- Per-class confidence scores with descriptions
- Mobile-responsive design
- Educational descriptions for each diagnosis

### Responsible AI

- Clear medical disclaimers
- Privacy-by-design (no data storage)
- Non-commercial open license
- Secure communication channels
- Transparent limitations

-----

## Model Architecture

### SimpleBNConv - Custom CNN

```
Input (224×224×3 RGB image)
    ↓
Block 1: Conv2d(3→8) → ReLU → BatchNorm2d → MaxPool2d
Block 2: Conv2d(8→16) → ReLU → BatchNorm2d → MaxPool2d
Block 3: Conv2d(16→32) → ReLU → BatchNorm2d → MaxPool2d
Block 4: Conv2d(32→64) → ReLU → BatchNorm2d → MaxPool2d
Block 5: Conv2d(64→128) → ReLU → BatchNorm2d → MaxPool2d
    ↓
Flatten
    ↓
Linear(128×7×7 → 7 classes)
    ↓
Softmax → Output probabilities
```

### Architecture Design Rationale

- **Lightweight**: Only 0.56 MB for fast loading and inference
- **Progressive Feature Learning**: Channel expansion (8→16→32→64→128)
- **Stability**: Batch normalization after each convolutional layer
- **Efficiency**: MaxPooling for spatial dimension reduction
- **Simplicity**: Single fully-connected classifier layer

-----

## Performance

### Overall Metrics

|Metric                         |Value     |
|-------------------------------|----------|
|**Validation Accuracy**        |**80.18%**|
|Training Accuracy              |83.26%    |
|UAR (Unweighted Average Recall)|46.97%    |
|Model Size                     |0.56 MB   |
|Inference Time                 |<100ms    |
|Total Parameters               |~560,000  |

### Per-Class Performance

|Class    |Full Name           |Type        |Recall |Support|
|---------|--------------------|------------|-------|-------|
|**NV**   |Melanocytic Nevus   |Benign      |92%    |6,705  |
|**VASC** |Vascular Lesion     |Benign      |82%    |142    |
|**BCC**  |Basal Cell Carcinoma|Malignant   |48%    |514    |
|**BKL**  |Benign Keratosis    |Benign      |45%    |1,099  |
|**AKIEC**|Actinic Keratosis   |Precancerous|37%    |327    |
|**DF**   |Dermatofibroma      |Benign      |26%    |115    |
|**MEL**  |Melanoma            |Malignant   |Limited|1,113  |

### Training Details

- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 64
- **Epochs**: 20
- **Data Augmentation**: Yes (rotation, flip, color jitter, affine)
- **Class Balancing**: WeightedRandomSampler
- **Regularization**: Batch normalization, Dropout (50%)

-----

## Installation

### Prerequisites

- Python 3.10+
- pip or conda package manager
- (Optional) CUDA-compatible GPU for training

### Quick Start

1. **Clone the repository**
   
   ```bash
   git clone https://github.com/Zahra58/DermAI-Skin-Lesion-Detection-System.git
   cd DermAI-Skin-Lesion-Detection-System
   ```
1. **Create virtual environment** (recommended)
   
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
1. **Install dependencies**
   
   ```bash
   pip install -r requirements.txt
   ```
1. **Download the trained model**
- The `best_model.pth` file (0.56 MB) is included in the repository
- Or train your own using the training notebook
1. **Run the application**
   
   ```bash
   streamlit run app.py
   ```
1. **Open in browser**
- Navigate to `http://localhost:8501`
- Upload an image and get predictions!

-----

## Usage

### Running Locally

```bash
# Start the Streamlit app
streamlit run app.py

# The app will open in your browser at http://localhost:8501
```

### Using the Web Interface

1. **Upload Image**: Click “Upload skin lesion image” button
1. **Select File**: Choose a `.jpg`, `.jpeg`, or `.png` file
1. **Analyze**: Click the “Analyze” button
1. **View Results**: See predicted class, confidence, and probability distribution

### Example Images

Test images from the ISIC 2018 dataset can be found in the `examples/` folder (if included).

### Programmatic Usage

```python
import torch
from PIL import Image
from torchvision import transforms

# Load model
model = SimpleBNConv(num_classes=7)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open('path/to/image.jpg').convert('RGB')
image_tensor = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probabilities, 1)

class_names = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
print(f"Prediction: {class_names[predicted.item()]}")
print(f"Confidence: {confidence.item():.2%}")
```

-----

## Dataset

### ISIC 2018 Challenge

**Dataset:** Skin Lesion Analysis Towards Melanoma Detection

- **Source**: [ISIC Archive](https://challenge2018.isic-archive.com/)
- **Total Images**: 10,015 dermatoscopic images
- **Resolution**: Variable (resized to 224×224)
- **Classes**: 7 diagnostic categories
- **Split**: ~80% train, ~20% validation
- **Annotations**: Expert-verified labels

### Class Distribution

|Class                     |Count|Percentage|Type        |
|--------------------------|-----|----------|------------|
|NV (Melanocytic Nevus)    |6,705|67.0%     |Benign      |
|MEL (Melanoma)            |1,113|11.1%     |Malignant   |
|BKL (Benign Keratosis)    |1,099|11.0%     |Benign      |
|BCC (Basal Cell Carcinoma)|514  |5.1%      |Malignant   |
|AKIEC (Actinic Keratosis) |327  |3.3%      |Precancerous|
|VASC (Vascular Lesion)    |142  |1.4%      |Benign      |
|DF (Dermatofibroma)       |115  |1.1%      |Benign      |

**Challenge**: Severe class imbalance (67:1 ratio)  
**Solution**: Implemented WeightedRandomSampler for balanced training

-----

## Project Structure

```
DermAI-Skin-Lesion-Detection-System/
│
├── app.py                      # Streamlit web application
├── best_model.pth             # Trained model weights (0.56 MB)
├── requirements.txt           # Python dependencies
├── LICENSE                    # CC BY-NC-SA 4.0 license
├── README.md                  # This file
│
├── training/                  # Training code (optional separate repo)
│   ├── models.py             # Model architecture definition
│   ├── train.py              # Training pipeline
│   ├── datasets.py           # Data loading and preprocessing
│   └── utils.py              # Helper functions
│
├── notebooks/                 # Jupyter notebooks
│   ├── exploration.ipynb     # Data exploration
│   ├── training.ipynb        # Model training
│   └── evaluation.ipynb      # Performance analysis
│
├── docs/                      # Documentation
│   ├── DEPLOYMENT.md         # Deployment guide
│   ├── ARCHITECTURE.md       # Technical architecture
│   └── CONTRIBUTING.md       # Contribution guidelines
│
└── assets/                    # Images and media
    ├── screenshots/          # App screenshots
    ├── diagrams/             # Architecture diagrams
    └── examples/             # Example predictions
```

-----

## Technologies

### Core ML Stack

- **[PyTorch 2.1.0](https://pytorch.org/)** - Deep learning framework
- **[torchvision 0.16.0](https://pytorch.org/vision/)** - Image preprocessing
- **[NumPy 1.24.3](https://numpy.org/)** - Numerical computing

### Web Application

- **[Streamlit 1.31.0](https://streamlit.io/)** - Web framework
- **[Pillow 10.2.0](https://python-pillow.org/)** - Image handling
- **[Plotly 5.18.0](https://plotly.com/)** - Interactive visualizations (future)

### Deployment

- **[HuggingFace Spaces](https://huggingface.co/spaces)** - Cloud hosting
- **Git** - Version control
- **GitHub Actions** - CI/CD (optional)

### Development Tools

- **Python 3.10** - Programming language
- **Jupyter Notebook** - Experimentation
- **Weights & Biases** - Experiment tracking (training)
- **VS Code** - IDE

-----

## Privacy & Security

### Data Protection

- ✅ **Zero data storage** - All images processed in memory only
- ✅ **No user tracking** - Session-based processing
- ✅ **No external API calls** - Self-contained inference
- ✅ **No analytics** - Minimal data collection
- ✅ **Automatic deletion** - Images discarded after prediction

### Security Features

- ✅ **HTTPS only** - Secure communication
- ✅ **No email exposure** - Contact via HuggingFace Discussions
- ✅ **Input validation** - File type and size checks
- ✅ **Error handling** - Graceful failure modes

-----

## 📄 License

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License** (CC BY-NC-SA 4.0).

### What This Means

#### ✅ You CAN:

- Use for educational purposes
- Use for academic research
- Use for personal projects
- Modify and adapt the code
- Share and redistribute (with attribution)

#### ❌ You CANNOT:

- Use for commercial purposes
- Use in clinical/medical practice without proper licensing
- Integrate into paid products or services
- Remove attribution or copyright notices

### Commercial Licensing

For commercial use, enterprise licenses, or custom implementations, please open a [Discussion](https://huggingface.co/spaces/Zahra58/dermai-skin-lesion-detector/discussions) on the HuggingFace Space.

**Full License:** <LICENSE>

-----

## Citation

If you use this project in your research or educational materials, please cite:

```bibtex
@software{etebari2024dermai,
  author = {Etebari, Zahra},
  title = {DermAI: Skin Lesion Detection System},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Zahra58/DermAI-Skin-Lesion-Detection-System},
  license = {CC BY-NC-SA 4.0}
}
```

### Dataset Citation

```bibtex
@article{tschandl2018ham10000,
  title={The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions},
  author={Tschandl, Philipp and Rosendahl, Cliff and Kittler, Harald},
  journal={Scientific data},
  volume={5},
  number={1},
  pages={1--9},
  year={2018},
  publisher={Nature Publishing Group}
}
```

-----

## Contributing

Contributions are welcome! This is an educational project, and improvements are encouraged.

### How to Contribute

1. **Fork the repository**
1. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
1. **Commit your changes** (`git commit -m 'Add amazing feature'`)
1. **Push to the branch** (`git push origin feature/amazing-feature`)
1. **Open a Pull Request**

### Contribution Ideas

- UI/UX improvements
- Additional visualizations
- Model architecture experiments
- Documentation enhancements
- Bug fixes
- New features

Please ensure your code follows the existing style and includes appropriate comments.

-----

## Known Issues & Limitations

### Current Limitations

1. **Class Imbalance Impact**
- Lower performance on minority classes (DF, VASC)
- Limited melanoma (MEL) samples affecting accuracy
1. **Medical Accuracy**
- Not validated for clinical use
- Should not replace professional diagnosis
- Educational demonstration only
1. **Image Quality Requirements**
- Best results with dermatoscopic images
- May not perform well on smartphone photos
- Requires clear, well-lit images

### Future Improvements

- [ ] Transfer learning with pre-trained models (ResNet, EfficientNet)
- [ ] Model ensembling for improved accuracy
- [ ] Explainable AI visualizations (Grad-CAM)
- [ ] Multi-language support
- [ ] Batch processing capability
- [ ] Confidence calibration
- [ ] Additional evaluation metrics
- [ ] Mobile app version

-----

## 📞 Contact

**Zahra Etebari**

- **Portfolio**: [Your Portfolio Website]
- **LinkedIn**: [Your LinkedIn Profile]
- **Live Demo**: [HuggingFace Space](https://huggingface.co/spaces/Zahra58/dermai-skin-lesion-detector)
-  **Discussions**: [HuggingFace Discussions](https://huggingface.co/spaces/Zahra58/dermai-skin-lesion-detector/discussions)

For commercial licensing inquiries, please open a Discussion on the HuggingFace Space.

-----

## Acknowledgments

- **ISIC Archive** - For providing the open-access dataset
- **PyTorch Team** - For the excellent deep learning framework
- **Streamlit** - For the intuitive web framework
- **HuggingFace** - For free model hosting and community
- **Medical Professionals** - For creating annotated datasets for research

-----

## Star History

If you find this project useful, please consider giving it a star! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=Zahra58/DermAI-Skin-Lesion-Detection-System&type=Date)](https://star-history.com/#Zahra58/DermAI-Skin-Lesion-Detection-System&Date)

-----

## Project Status

![Status](https://img.shields.io/badge/Status-Active-success)
![Maintenance](https://img.shields.io/badge/Maintained-Yes-green.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

**Last Updated:** December 2024  
**Current Version:** 1.0.0  
**Status:** Production-ready deployment ✅

-----

<p align="center">
  <strong>Built with ❤️ for educational purposes</strong>
  <br>
  <sub>Not intended for clinical use | Always consult healthcare professionals</sub>
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/Zahra58/dermai-skin-lesion-detector">
    <img src="https://img.shields.io/badge/-Try%20Live%20Demo-success?style=for-the-badge" alt="Try Demo">
  </a>
</p>

-----

**© 2025 Zahra Etebari | Licensed under CC BY-NC-SA 4.0**