# AI/ML-Driven Automated Feature Detection and Change Analysis of Glacial Lakes

This project uses deep learning for automated detection of **glacial lakes** from multi-source satellite imagery (Sentinel-2 `.tif`, PNG). Built with a U-Net segmentation model, it enables climate change analysis, risk prediction, and future feature tracking.

## Table of Contents

- [Project Overview](#project-overview)
- [Application Architecture & User Workflow](#application-architecture--user-workflow)
- [Architecture Diagram](#architecture-diagram)
- [Workflow](#workflow)
- [Sample Results](#sample-results)
- [Confusion Matrix & Metrics](#confusion-matrix--metrics)
- [Features](#features)
- [Installation & Run](#installation--run)
- [Usage](#usage)
- [Model Details](#model-details)
- [Project Structure](#project-structure)
- [Future Work](#future-work)
- [Troubleshooting](#troubleshooting)
- [References & Resources](#references--resources)
- [Author](#author)
- [License](#license)

## 📖 Project Overview

This project automates the detection of glacial lakes using satellite imagery through semantic segmentation. It helps monitor the formation and expansion of glacial lakes, which are critical indicators of climate change and potential GLOF (Glacial Lake Outburst Flood) risks.

Due to climate change and accelerated glacial melt, new glacial lakes are forming and existing ones are expanding across mountainous regions. These lakes pose potential dangers through Glacial Lake Outburst Floods (GLOFs). Manual monitoring is time-consuming and resource-intensive, making automated detection crucial for timely risk assessment.

## 🔄 Application Architecture & User Workflow

### User Interaction Flow

```mermaid
flowchart TD
    A[Start Streamlit App] --> B[Upload Satellite Image]
    B --> C[Image Preprocessing]
    C --> D[Model Inference]
    D --> E[Display Results]
    E --> F[Download Options]
    F --> G[User Feedback]
    G -->|Positive| H[Continue Monitoring]
    G -->|Negative| I[Report Issue]
    I --> J[Review & Fix Issues]
    J --> E
```

1. **Start Streamlit App**: User launches the application.
2. **Upload Satellite Image**: User uploads a satellite image in `.tif`, `.png`, or `.jpg` format.
3. **Image Preprocessing**: The application preprocesses the image (resizing, normalization).
4. **Model Inference**: The trained U-Net model predicts the glacial lake mask.
5. **Display Results**: The application displays the original image, predicted mask, and overlay.
6. **Download Options**: User can download the results (masks, overlays, contours).
7. **User Feedback**: User provides feedback on the results (positive/negative).
8. **Continue Monitoring**: Positive feedback leads to continued monitoring of glacial lakes.
9. **Report Issue**: Negative feedback allows users to report issues or inaccuracies.
10. **Review & Fix Issues**: The development team reviews and addresses reported issues.

### System Architecture

```
+-------------------------+
| Download Sentinel-2 .tif images |
+-------------------------+
            |
            v
+-------------------------+
| Preprocess with Rasterio |
+-------------------------+
            |
            v
+-------------------------+
| Crop to 256x256 patches |
+-------------------------+
            |
            v
+-------------------------+
| Manually annotate lakes using LabelMe |
+-------------------------+
            |
            v
+-------------------------+
| Convert annotations to binary .png masks |
+-------------------------+
            |
            v
+-------------------------+
| Convert PNG pairs to .npy |
+-------------------------+
            |
            v
+-------------------------+
| Train U-Net on (image, mask) pairs |
+-------------------------+
            |
            v
+-------------------------+
| Evaluate metrics, save model |
+-------------------------+
            |
            v
+-------------------------+
| Deploy using Streamlit app |
+-------------------------+
```

### Model Architecture

```text
U-Net Semantic Segmentation Model
----------------------------------
Encoder: ResNet-18 pretrained on ImageNet
Input: 3-channel RGB satellite crops (256x256)
Decoder: Transposed Convolutions + Concatenation Skip Connections
Output: 1-channel binary mask (Lake = 1, Background = 0)
Loss: Binary Cross Entropy with Logits
Optimizer: Adam
Framework: PyTorch
```

✅ Total images: 40
✅ Total masks: 40
🖥️ Trained on: Google Colab (CUDA GPU)

## 🔄 Workflow

1. **Data Acquisition**: Download `.tif` satellite images from Sentinel-2
2. **Data Preparation**: Crop and annotate glacial lake boundaries (LabelMe)
3. **Dataset Creation**: Convert annotations to `.npy` image–mask dataset
4. **Model Training**: Train U-Net segmentation model (ResNet-18 backbone)
5. **Model Deployment**: Export trained model for inference
6. **Interactive App**: Run inference via interactive Streamlit app

📥 Total `.tif` files used: 9
📤 Final `.npy` image-mask pairs: 40
🧠 Model file: `unet_model_augmented.pth`

## 🎨 Sample Results

| Input Image | Binary Mask | Overlay Image |
|-------------|-------------|---------------|
| ![Input](images/inupt.png) | ![Mask](images/glacial_lake_mask.png) | ![Overlay](images/glacial_lake_overlay.png) |

![Analysis](images/pred.png)

## 📉 Confusion Matrix & Metrics

Based on our final training logs, here are the real evaluation metrics:

| Metric                  | Value  |
|-------------------------|--------|
| **Accuracy**            | 78.15% |
| **Precision**           | 79.85% |
| **Recall**              | 95.32% |
| **F1 Score**            | 86.86% |
| **IoU (Jaccard Index)** | 76.37% |

🧾 From final epoch's confusion matrix:

```
[[ 752  192]
 [  39  753]]
```

## ✨ Features

- **Multi-format Image Support**: Upload `.tif`, `.png`, `.jpg` satellite images
- **Detailed Visualization Options**:
  - Binary lake mask generation
  - Overlay visualization (original + prediction)
  - Contour extraction and display
  - Statistical analysis of lake coverage
- **Download Options**: Export masks, overlays and contour maps as PNG files
- **User-friendly Interface**: Intuitive Streamlit interface with tabbed visualization
- **Robust Processing**: Fallback mechanisms for handling different file types and missing dependencies

## 🚀 Installation & Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/glacial-lake-detection.git
   cd glacial-lake-app
   ```

2. **Create a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/MacOS
   ```

3. **Install dependencies**:
   ```bash
   # Option 1: Using the provided script
   install_dependencies.bat
   
   # Option 2: Manual installation
   pip install -r requirements.txt
   ```

4. **Download the model**:
   Ensure `unet_model_augmented.pth` is placed in the root directory of the application.

5. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 💻 Usage

1. **Start the application**:
   ```
   streamlit run app.py
   ```

2. **Upload an image**:
   - Use the file uploader to select a `.tif`, `.png`, or `.jpg` file
   - For best results, use RGB satellite imagery

3. **View results**:
   - Navigate through the tabs to view different visualizations
   - Download any of the outputs using the provided buttons

4. **Interpret results**:
   - Blue regions in the mask represent detected glacial lakes
   - Contours show lake boundaries
   - The Analysis tab provides quantitative information about coverage

## 🧠 Model Details

- **Architecture**: U-Net with ResNet-18 encoder
- **Input Size**: 256×256 pixels (images are automatically resized)
- **Output**: Binary segmentation mask
- **Training Data**: Custom annotated dataset from Sentinel-2 imagery
- **Regions**: Primarily trained on imagery from Ladakh, Himachal Pradesh, and Nepal

## 📂 Project Structure

```
glacial-lake-detection/
│
├── app.py                   # Main Streamlit application
├── requirements.txt          # Python package dependencies
├── install_dependencies.bat  # Windows dependency installation script
├── assets/                   # Folder for demo images and assets
│   ├── demo.png              # Demo image showing results
│   ├── sample_input.png      # Sample input image
│   ├── sample_mask.png       # Sample binary mask
│   ├── sample_overlay.png    # Sample overlay image
│   └── confusion_matrix.png  # Confusion matrix image
├── models/                  # Folder for storing trained models
│   └── unet_model_augmented.pth  # Pre-trained U-Net model
├── utils.py                 # Utility functions for image processing and model inference
└── README.md                # Project documentation
```

## ⏭️ Future Work

- Integrate additional data sources (e.g., Landsat, MODIS) for broader coverage
- Implement change detection to monitor lake expansion/contraction over time
- Enhance model architecture for improved accuracy and generalization
- Develop a web-based GIS interface for interactive data exploration

## 🛠️ Troubleshooting

- **Issue**: Application crashes on startup
  - **Solution**: Ensure all dependencies are correctly installed. Check Python and package versions.
- **Issue**: Model fails to load
  - **Solution**: Verify `unet_model_augmented.pth` is in the correct directory. Re-download the model if necessary.
- **Issue**: Image upload not working
  - **Solution**: Check file format and size. Ensure the file is not corrupted.

## 📚 References & Resources

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [Streamlit Documentation](https://docs.streamlit.io/library)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## 👤 Author

[Your Name](https://github.com/yourusername)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

