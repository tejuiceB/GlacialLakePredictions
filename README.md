# AI/ML-Driven Automated Feature Detection and Change Analysis of Glacial Lakes

This project uses deep learning for automated detection of **glacial lakes** from multi-source satellite imagery (Sentinel-2 `.tif`, PNG). Built with a U-Net segmentation model, it enables climate change analysis, risk prediction, and future feature tracking.

## 📖 Project Overview

This project automates the detection of glacial lakes using satellite imagery through semantic segmentation. It helps monitor the formation and expansion of glacial lakes, which are critical indicators of climate change and potential GLOF (Glacial Lake Outburst Flood) risks.

Due to climate change and accelerated glacial melt, new glacial lakes are forming and existing ones are expanding across mountainous regions. These lakes pose potential dangers through Glacial Lake Outburst Floods (GLOFs). Manual monitoring is time-consuming and resource-intensive, making automated detection crucial for timely risk assessment.

## 🏗️ System Architecture Overview

### High-Level System Architecture

```mermaid
graph TB
    subgraph "User Interface Layer"
        WebApp[Streamlit Web Application]
        FileUpload[File Upload Interface]
        Visualization[Visualization Dashboard]
    end
    
    subgraph "Processing Layer"
        ImageProcessor[Image Preprocessing]
        MLInference[ML Inference Engine]
        PostProcessor[Post Processing]
    end
    
    subgraph "Model Layer"
        UNet[U-Net Segmentation Model]
        ResNet[ResNet-18 Backbone]
        ModelWeights[Trained Model Weights]
    end
    
    subgraph "Storage Layer"
        TempStorage[Temporary File Storage]
        Results[Results Cache]
        Downloads[Download Files]
    end
    
    subgraph "External Data Sources"
        Sentinel[Sentinel-2 Imagery]
        UserImages[User Uploaded Images]
        Annotations[Manual Annotations]
    end
    
    WebApp --> FileUpload
    FileUpload --> ImageProcessor
    ImageProcessor --> MLInference
    MLInference --> UNet
    UNet --> ResNet
    ResNet --> ModelWeights
    MLInference --> PostProcessor
    PostProcessor --> Visualization
    Visualization --> WebApp
    
    ImageProcessor --> TempStorage
    PostProcessor --> Results
    Results --> Downloads
    
    Sentinel --> ImageProcessor
    UserImages --> FileUpload
    Annotations --> ModelWeights
```

### Detailed Component Architecture

```mermaid
graph LR
    subgraph "Frontend Components"
        UI[Streamlit UI]
        Tabs[Visualization Tabs]
        Downloads[Download Buttons]
        Metrics[Analysis Metrics]
    end
    
    subgraph "Core Processing"
        Preprocess[Image Preprocessing]
        ModelLoad[Model Loading]
        Inference[Inference Pipeline]
        Postprocess[Post Processing]
    end
    
    subgraph "Visualization Engine"
        MaskViz[Mask Visualization]
        OverlayViz[Overlay Generation]
        ContourViz[Contour Extraction]
        StatsViz[Statistical Analysis]
    end
    
    subgraph "File Management"
        Upload[File Upload Handler]
        Validation[Format Validation]
        TempFiles[Temporary Storage]
        Export[Export Manager]
    end
    
    UI --> Upload
    Upload --> Validation
    Validation --> Preprocess
    Preprocess --> ModelLoad
    ModelLoad --> Inference
    Inference --> Postprocess
    
    Postprocess --> MaskViz
    Postprocess --> OverlayViz
    Postprocess --> ContourViz
    Postprocess --> StatsViz
    
    MaskViz --> Tabs
    OverlayViz --> Tabs
    ContourViz --> Tabs
    StatsViz --> Metrics
    
    Tabs --> Downloads
    Downloads --> Export
    Export --> TempFiles
```

## 🔄 User Interaction Flow

### Complete User Journey

```mermaid
flowchart TD
    Start([User Access Application]) --> LoadApp[Application Loads]
    LoadApp --> ModelCheck{Model Available?}
    ModelCheck -->|No| Error[Display Error Message]
    ModelCheck -->|Yes| Dashboard[Show Main Dashboard]
    
    Dashboard --> Upload[User Uploads Image]
    Upload --> ValidateFile{Valid File Type?}
    ValidateFile -->|No| FileError[Show File Error]
    ValidateFile -->|Yes| DisplayOriginal[Display Original Image]
    
    DisplayOriginal --> StartProcessing[Start ML Processing]
    StartProcessing --> Preprocess[Image Preprocessing]
    Preprocess --> RunModel[Run U-Net Model]
    RunModel --> GenerateMask[Generate Binary Mask]
    GenerateMask --> CreateVisualizations[Create Visualizations]
    
    CreateVisualizations --> ShowTabs[Display Result Tabs]
    ShowTabs --> UserChoice{User Selects Tab}
    
    UserChoice -->|Mask| MaskTab[Binary Mask View]
    UserChoice -->|Overlay| OverlayTab[Overlay View]
    UserChoice -->|Contours| ContourTab[Contour View]
    UserChoice -->|Analysis| AnalysisTab[Statistical Analysis]
    
    MaskTab --> DownloadOption[Download Option]
    OverlayTab --> DownloadOption
    ContourTab --> DownloadOption
    AnalysisTab --> DownloadOption
    
    DownloadOption --> Continue{Continue?}
    Continue -->|Yes| Upload
    Continue -->|No| End([Session End])
    
    FileError --> Upload
    Error --> End
```

### Detailed Processing Pipeline

```mermaid
sequenceDiagram
    participant User
    participant StreamlitUI as Streamlit UI
    participant ImageProc as Image Processor
    participant Model as U-Net Model
    participant Visualizer as Visualization Engine
    participant FileManager as File Manager
    
    User->>StreamlitUI: Upload satellite image
    StreamlitUI->>ImageProc: Process uploaded file
    
    Note over ImageProc: File type validation
    ImageProc->>ImageProc: Validate file format
    ImageProc->>ImageProc: Load image (PIL/Rasterio)
    ImageProc->>ImageProc: Resize to 256x256
    ImageProc->>ImageProc: Normalize pixel values
    
    ImageProc->>Model: Send preprocessed tensor
    
    Note over Model: ML Inference
    Model->>Model: Forward pass through U-Net
    Model->>Model: Apply sigmoid activation
    Model->>Model: Generate probability mask
    
    Model->>Visualizer: Return prediction probabilities
    
    Note over Visualizer: Post-processing
    Visualizer->>Visualizer: Apply threshold (0.5)
    Visualizer->>Visualizer: Create binary mask
    Visualizer->>Visualizer: Generate overlay image
    Visualizer->>Visualizer: Extract contours
    Visualizer->>Visualizer: Calculate statistics
    
    Visualizer->>StreamlitUI: Send all visualizations
    StreamlitUI->>User: Display tabbed results
    
    User->>StreamlitUI: Request download
    StreamlitUI->>FileManager: Prepare download file
    FileManager->>User: Provide download link
```

## 🧠 Model Architecture Flow

### U-Net Architecture Details

```mermaid
graph TD
    subgraph "Input Layer"
        Input[RGB Image 256x256x3]
    end
    
    subgraph "Encoder (ResNet-18)"
        Conv1[Conv Block 1]
        Conv2[Conv Block 2]
        Conv3[Conv Block 3]
        Conv4[Conv Block 4]
        Bottleneck[Bottleneck Features]
    end
    
    subgraph "Decoder"
        Up1[Upconv Block 1]
        Up2[Upconv Block 2]
        Up3[Upconv Block 3]
        Up4[Upconv Block 4]
    end
    
    subgraph "Output Layer"
        Output[Binary Mask 256x256x1]
    end
    
    subgraph "Skip Connections"
        Skip1[Skip Connection 1]
        Skip2[Skip Connection 2]
        Skip3[Skip Connection 3]
        Skip4[Skip Connection 4]
    end
    
    Input --> Conv1
    Conv1 --> Conv2
    Conv2 --> Conv3
    Conv3 --> Conv4
    Conv4 --> Bottleneck
    
    Bottleneck --> Up1
    Up1 --> Up2
    Up2 --> Up3
    Up3 --> Up4
    Up4 --> Output
    
    Conv1 --> Skip1
    Conv2 --> Skip2
    Conv3 --> Skip3
    Conv4 --> Skip4
    
    Skip1 --> Up4
    Skip2 --> Up3
    Skip3 --> Up2
    Skip4 --> Up1
```

### Training and Inference Pipeline

```mermaid
flowchart LR
    subgraph "Training Phase"
        SatelliteData[Sentinel-2 Data] --> Annotation[Manual Annotation]
        Annotation --> TrainingSet[Training Dataset]
        TrainingSet --> ModelTraining[U-Net Training]
        ModelTraining --> SavedModel[Saved Model Weights]
    end
    
    subgraph "Inference Phase"
        UserImage[User Input Image] --> Preprocessing[Image Preprocessing]
        Preprocessing --> LoadModel[Load Trained Model]
        SavedModel --> LoadModel
        LoadModel --> Prediction[Generate Prediction]
        Prediction --> Postprocessing[Post-processing]
        Postprocessing --> Results[Final Results]
    end
    
    subgraph "Validation"
        Results --> Metrics[Calculate Metrics]
        Metrics --> Accuracy[Accuracy: 78.15%]
        Metrics --> Precision[Precision: 79.85%]
        Metrics --> Recall[Recall: 95.32%]
        Metrics --> F1Score[F1 Score: 86.86%]
        Metrics --> IoU[IoU: 76.37%]
    end
```

## 📊 Data Flow Architecture

### End-to-End Data Processing

```mermaid
flowchart TD
    subgraph "Data Sources"
        Sentinel2[Sentinel-2 Satellite Imagery]
        UserUploads[User Uploaded Images]
        HistoricalData[Historical Lake Data]
    end
    
    subgraph "Data Ingestion"
        FileUpload[File Upload Handler]
        FormatValidation[Format Validation]
        TypeDetection[Image Type Detection]
    end
    
    subgraph "Preprocessing Pipeline"
        LoadImage[Image Loading]
        Resize[Resize to 256x256]
        Normalize[Pixel Normalization]
        TensorConversion[Tensor Conversion]
    end
    
    subgraph "ML Processing"
        ModelInference[U-Net Inference]
        ProbabilityMask[Probability Mask]
        BinaryThreshold[Binary Thresholding]
        MaskGeneration[Binary Mask Generation]
    end
    
    subgraph "Visualization Processing"
        OverlayCreation[Overlay Image Creation]
        ContourExtraction[Contour Extraction]
        StatisticalAnalysis[Statistical Analysis]
        MetricsCalculation[Metrics Calculation]
    end
    
    subgraph "Output Generation"
        TabVisualization[Tabbed Visualization]
        DownloadFiles[Download File Generation]
        ResultDisplay[Result Display]
    end
    
    Sentinel2 --> FileUpload
    UserUploads --> FileUpload
    HistoricalData --> FileUpload
    
    FileUpload --> FormatValidation
    FormatValidation --> TypeDetection
    TypeDetection --> LoadImage
    
    LoadImage --> Resize
    Resize --> Normalize
    Normalize --> TensorConversion
    
    TensorConversion --> ModelInference
    ModelInference --> ProbabilityMask
    ProbabilityMask --> BinaryThreshold
    BinaryThreshold --> MaskGeneration
    
    MaskGeneration --> OverlayCreation
    MaskGeneration --> ContourExtraction
    MaskGeneration --> StatisticalAnalysis
    StatisticalAnalysis --> MetricsCalculation
    
    OverlayCreation --> TabVisualization
    ContourExtraction --> TabVisualization
    MetricsCalculation --> TabVisualization
    TabVisualization --> DownloadFiles
    TabVisualization --> ResultDisplay
```

## 🔧 Technical Implementation Flow

### Application Startup and Model Loading

```mermaid
sequenceDiagram
    participant App as Streamlit App
    participant Cache as Streamlit Cache
    participant Model as Model Loader
    participant UI as User Interface
    
    App->>Cache: Check for cached model
    Cache->>Model: Load U-Net model
    Model->>Model: Load segmentation_models_pytorch
    Model->>Model: Initialize ResNet-18 backbone
    Model->>Model: Load trained weights
    Model->>Model: Set model to eval mode
    Model->>Cache: Return loaded model
    Cache->>App: Model ready
    App->>UI: Display main interface
    UI->>App: Ready for user input
```

### Error Handling and Fallback Systems

```mermaid
flowchart TD
    UserAction[User Action] --> TryMain[Try Main Function]
    TryMain --> CheckDependencies{Check Dependencies}
    
    CheckDependencies -->|SMP Missing| SMPError[segmentation_models_pytorch Error]
    CheckDependencies -->|GDAL Missing| GDALError[GDAL Error]
    CheckDependencies -->|SKImage Missing| SKImageError[scikit-image Error]
    CheckDependencies -->|All OK| ProcessNormal[Normal Processing]
    
    SMPError --> FallbackModel[Use Fallback Model Loading]
    GDALError --> FallbackImage[Use PIL for Image Loading]
    SKImageError --> FallbackContour[Use OpenCV for Contours]
    
    FallbackModel --> LimitedFunction[Limited Functionality Mode]
    FallbackImage --> ProcessNormal
    FallbackContour --> ProcessNormal
    
    LimitedFunction --> ShowWarning[Display Warning Message]
    ProcessNormal --> Success[Successful Processing]
    ShowWarning --> Success
```

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

[Tejas Bhurbhure](https://github.com/tejuiceB)
