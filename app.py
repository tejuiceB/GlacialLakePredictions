import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import io
from io import BytesIO
import tempfile
import segmentation_models_pytorch as smp
import torchvision.transforms as T
import cv2
import sys
import subprocess

# Check if scikit-image is installed, if not offer to install it
try:
    from skimage import measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    st.error("📦 Required package 'scikit-image' is not installed.")
    
    if st.button("Install scikit-image Now"):
        st.info("Installing scikit-image... This may take a moment.")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-image"])
            st.success("Installation successful! Please refresh the page.")
            from skimage import measure
            SKIMAGE_AVAILABLE = True
        except Exception as e:
            st.error(f"Installation failed: {e}")
            st.info("You can manually install it by running: pip install scikit-image")

# Define a fallback contour extraction function if skimage is not available
def fallback_extract_contours(binary_mask):
    """Simple contour extraction using OpenCV if skimage is not available"""
    # Convert to the format expected by OpenCV
    mask_8bit = (binary_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Convert OpenCV contours to a format similar to skimage's
    result_contours = []
    for contour in contours:
        if len(contour) > 5:  # Only keep contours with sufficient points
            # Reshape and convert to the expected format
            points = contour.squeeze().astype(np.float64)
            if points.ndim == 2 and points.shape[0] > 2:  # Valid contour
                # Swap x,y to match skimage's y,x format
                swapped = np.column_stack((points[:, 1], points[:, 0]))
                result_contours.append(swapped)
    
    return result_contours

# Set page configuration
st.set_page_config(
    page_title="Glacial Lake Detection",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .download-btn {
        background-color: #1E88E5;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        text-decoration: none;
    }
    </style>
""", unsafe_allow_html=True)

# Create two columns for the title and an optional image
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<p class="main-header">🌊 Glacial Lake Detection</p>', unsafe_allow_html=True)
    st.markdown("Upload satellite imagery to identify glacial lakes using deep learning")

# Function to load the model
@st.cache_resource
def load_model():
    model = smp.Unet(encoder_name="resnet18", in_channels=3, classes=1)
    model.load_state_dict(torch.load("unet_model_augmented.pth", map_location="cpu"))
    model.eval()
    return model

# Function to preprocess images
def preprocess_image(img):
    # Resize to model input size (256x256)
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])
    return transform(img).unsqueeze(0)  # Add batch dimension

# Function to predict mask
def predict_mask(model, img_tensor):
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.sigmoid(output)
        return probabilities[0, 0].numpy()  # Remove batch & channel dimensions

# Function to overlay mask on original image
def create_overlay(img_array, mask_array, alpha=0.5, color=[0, 255, 0]):
    # Resize mask to match original image
    mask_resized = cv2.resize(
        mask_array, (img_array.shape[1], img_array.shape[0]), 
        interpolation=cv2.INTER_NEAREST
    )
    
    # Create a color mask
    color_mask = np.zeros_like(img_array)
    color_mask[mask_resized == 1] = color
    
    # Blend original image with color mask
    overlay = cv2.addWeighted(img_array, 1, color_mask, alpha, 0)
    return overlay

# Function to extract contours from mask - updated to handle missing skimage
def extract_contours(mask, min_size=10):
    if SKIMAGE_AVAILABLE:
        # Use skimage if available
        # Label connected components
        labels = measure.label(mask)
        # Extract properties for each region
        props = measure.regionprops(labels)
        
        # Filter by size and extract contours
        contours = []
        for prop in props:
            if prop.area >= min_size:
                # Extract boundary points
                contour = measure.find_contours(labels == prop.label, 0.5)
                if contour:
                    contours.extend(contour)
        
        return contours
    else:
        # Use fallback OpenCV implementation
        return fallback_extract_contours(mask)

# Main function
def main():
    # Load model
    try:
        model = load_model()
        st.sidebar.success("✅ Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load model: {str(e)}")
        st.stop()
    
    # File uploader for both TIF and PNG
    uploaded_file = st.file_uploader("Upload satellite image", type=["tif", "tiff", "png", "jpg", "jpeg"])
    
    if uploaded_file:
        # Read file bytes
        file_bytes = uploaded_file.read()
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        # Process based on file type
        try:
            if file_type in ['png', 'jpg', 'jpeg']:
                # Process PNG/JPG
                img = Image.open(BytesIO(file_bytes)).convert("RGB")
                img_array = np.array(img)
                
            elif file_type in ['tif', 'tiff']:
                # Process TIF - first try with rasterio
                try:
                    import rasterio
                    from rasterio.io import MemoryFile
                    
                    with MemoryFile(file_bytes) as memfile:
                        with memfile.open() as src:
                            # Read first 3 bands as RGB
                            img_array = np.zeros((src.height, src.width, 3), dtype=np.uint8)
                            for i in range(min(3, src.count)):
                                img_array[:, :, i] = src.read(i + 1)
                            
                            # Normalize if needed
                            if img_array.max() > 255:
                                img_array = (img_array / img_array.max() * 255).astype(np.uint8)
                            
                            # Handle single band images
                            if src.count == 1:
                                img_array = np.stack([img_array[:,:,0]]*3, axis=-1)
                except Exception as e:
                    st.warning(f"Rasterio processing failed, trying with PIL: {str(e)}")
                    # Fallback to PIL for TIF
                    img = Image.open(BytesIO(file_bytes)).convert("RGB")
                    img_array = np.array(img)
            
            # Display original image
            st.subheader("📷 Original Image")
            st.image(img_array, use_column_width=True)
            
            # Prepare for prediction
            img_pil = Image.fromarray(img_array)
            img_tensor = preprocess_image(img_pil)
            
            # Make prediction with progress bar
            with st.spinner("🧠 Running model inference..."):
                # Get raw probabilities
                prob_mask = predict_mask(model, img_tensor)
                
                # Convert to binary mask using threshold
                threshold = 0.5
                binary_mask = (prob_mask > threshold).astype(np.uint8)
                
                # Resize mask to match display size (for visualization)
                display_mask = cv2.resize(
                    binary_mask, (img_array.shape[1], img_array.shape[0]), 
                    interpolation=cv2.INTER_NEAREST
                )
            
            # Create tabs for different visualizations
            viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
                "Lake Mask", "Overlay", "Contours", "Analysis"
            ])
            
            # Tab 1: Binary Mask
            with viz_tab1:
                st.subheader("🧠 Predicted Lake Mask")
                # Display mask
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(display_mask, cmap='Blues')
                ax.axis('off')
                st.pyplot(fig)
                
                # Download button for mask
                mask_img = Image.fromarray(display_mask * 255).convert("L")
                buf = BytesIO()
                mask_img.save(buf, format="PNG")
                st.download_button(
                    label="⬇️ Download Mask (PNG)",
                    data=buf.getvalue(),
                    file_name="glacial_lake_mask.png",
                    mime="image/png"
                )
            
            # Tab 2: Overlay
            with viz_tab2:
                st.subheader("📌 Lake Overlay")
                # Create overlay image
                overlay_img = create_overlay(img_array, display_mask, alpha=0.4, color=[0, 255, 0])
                st.image(overlay_img, use_column_width=True)
                
                # Download button for overlay
                overlay_pil = Image.fromarray(overlay_img)
                buf = BytesIO()
                overlay_pil.save(buf, format="PNG")
                st.download_button(
                    label="⬇️ Download Overlay (PNG)",
                    data=buf.getvalue(),
                    file_name="glacial_lake_overlay.png", 
                    mime="image/png"
                )
            
            # Tab 3: Contours
            with viz_tab3:
                st.subheader("🔍 Lake Contours")
                
                if not SKIMAGE_AVAILABLE:
                    st.warning("For better contour detection, please install scikit-image.")
                
                # Extract contours
                contours = extract_contours(binary_mask, min_size=20)
                
                if not contours:
                    st.info("No significant contours detected in this image.")
                else:
                    # Plot contours
                    fig, ax = plt.subplots(figsize=(10, 10))
                    ax.imshow(img_array)
                    
                    for contour in contours:
                        if contour.shape[0] > 2:  # Make sure contour has enough points
                            ax.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=2)
                    
                    ax.axis('off')
                    st.pyplot(fig)
                    
                    # Download contour image
                    buf = BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                    buf.seek(0)
                    st.download_button(
                        label="⬇️ Download Contour Map (PNG)",
                        data=buf.getvalue(),
                        file_name="glacial_lake_contours.png",
                        mime="image/png"
                    )
            
            # Tab 4: Analysis - Modify to work without skimage
            with viz_tab4:
                st.subheader("📊 Lake Analysis")
                
                # Calculate lake coverage
                total_pixels = display_mask.size
                lake_pixels = np.sum(display_mask)
                lake_percentage = (lake_pixels / total_pixels) * 100
                
                # Lake count - simplified if skimage is not available
                if SKIMAGE_AVAILABLE:
                    labels = measure.label(display_mask)
                    num_lakes = len(np.unique(labels)) - 1  # Subtract background
                    
                    # Display metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Lake Coverage", f"{lake_percentage:.2f}%")
                    with col2:
                        st.metric("Number of Lakes", num_lakes)
                    
                    # Show lake sizes
                    if num_lakes > 0:
                        props = measure.regionprops(labels)
                        areas = [prop.area for prop in props]
                        
                        st.subheader("Lake Size Distribution")
                        fig, ax = plt.subplots()
                        ax.hist(areas, bins=20)
                        ax.set_xlabel("Area (pixels)")
                        ax.set_ylabel("Count")
                        st.pyplot(fig)
                else:
                    # Simplified metrics without skimage
                    st.metric("Lake Coverage", f"{lake_percentage:.2f}%")
                    st.info("Install scikit-image for more detailed analysis (lake count, size distribution)")
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.exception(e)
    
    # About section - Updated with the new blog content
    with st.sidebar.expander("📰 About This Project", expanded=True):
        st.markdown("""
        ## 📰 **AI/ML-Driven Automated Feature Detection and Change Analysis of Glacial Lakes from Multi-source Satellite Imagery**
        
        *(Focused Feature: Glacial Lake Detection)*

        ### ✅ What We Achieved

        In this project, we focused on **automated detection of glacial lakes** using AI/ML techniques and satellite imagery from regions like **Ladakh, Himachal Pradesh, and Nepal**.

        Key achievements include:

        * ✅ **Custom dataset** creation from Sentinel-2 `.tif` files
        * ✅ Manual **annotation of glacial lakes** to generate training masks
        * ✅ Training of a **U-Net segmentation model** (ResNet-18 backbone)
        * ✅ Creation of an interactive **Streamlit web app** for:
          * Uploading satellite `.tif` or `.png` images
          * Predicting glacial lake areas
          * Visualizing binary masks and overlays
          * Downloading predictions
        * ✅ Generation of over **25+ binary mask outputs** from real satellite tiles

        This prototype demonstrates the **power of deep learning** in automating climate-change-related feature detection with ease and speed.

        ### 🚀 Future Goals

        We aim to expand this project by including:

        * 🌍 **Change analysis over time** using multi-temporal satellite data
        * 🛰️ Integration of official **reference inventories** (GLIMS, NRSC, ICIMOD) for validation
        * 🧾 Export of **polygon shapefiles** for GIS tools
        * 📊 **Quantification of lake area**, growth rate, and formation of new lakes
        * 🧠 Expansion to detect **roads and drainage systems** in future phases
        * 📈 Implementing **accuracy evaluation** (IoU, precision, recall) with real-world data

        This project lays the foundation for **AI-powered climate resilience monitoring**, offering scalable insights for researchers, disaster management authorities, and policymakers.
        
        Made with ❤️ by Tejas Bhurbhure
        """)
    
    # Technical details
    with st.sidebar.expander("🔧 Technical Details"):
        st.markdown("""
        - **Model**: U-Net with ResNet18 encoder
        - **Training Data**: Satellite imagery with labeled glacial lakes
        - **Input Size**: 256x256 pixels
        - **Output**: Binary segmentation mask
        
        Preprocessing involves resizing and normalizing the input images. The model outputs probability maps which are thresholded to create binary masks.
        """)

if __name__ == "__main__":
    main()
