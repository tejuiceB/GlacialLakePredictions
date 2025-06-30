import torch
import numpy as np
import cv2
import os
from PIL import Image

# Try to import segmentation_models_pytorch, with a fallback
try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    print("Warning: segmentation_models_pytorch not found.")
    print("Please install it using: pip install segmentation-models-pytorch")

# Try to import GDAL, with a fallback to use PIL
try:
    from osgeo import gdal
    GDAL_AVAILABLE = True
except ImportError:
    import PIL.Image
    GDAL_AVAILABLE = False
    print("GDAL not available, using PIL for image processing")

def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
        
    if SMP_AVAILABLE:
        # Use segmentation_models_pytorch if available
        model = smp.Unet(encoder_name="resnet18", in_channels=3, classes=1)
        model.load_state_dict(torch.load(path, map_location='cpu'))
    else:
        # Simple fallback - load only the state dict
        print("Loading model in limited mode (no segmentation_models_pytorch)")
        print("Some functionality may be unavailable")
        model = torch.load(path, map_location='cpu')
        
    if hasattr(model, 'eval'):
        model.eval()
    return model

def preprocess_tif(tif_path):
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"Image file not found: {tif_path}")
        
    if GDAL_AVAILABLE:
        # GDAL approach for proper geospatial processing
        ds = gdal.Open(tif_path)
        img = np.stack([ds.GetRasterBand(i+1).ReadAsArray() for i in range(3)], axis=-1)
    else:
        # Fallback to PIL for basic image loading
        from PIL import Image
        img = np.array(Image.open(tif_path).convert('RGB'))
    
    # Common processing regardless of loader
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 255.0
    return torch.tensor(img).permute(2, 0, 1).unsqueeze(0)

def preprocess_png(png_path):
    """
    Preprocess PNG images for model input
    """
    img = Image.open(png_path).convert("RGB")
    img = np.array(img.resize((256, 256)))
    img = img.astype(np.float32) / 255.0
    return torch.tensor(img).permute(2, 0, 1).unsqueeze(0)

def predict_mask(model, img_tensor):
    with torch.no_grad():
        if SMP_AVAILABLE or hasattr(model, 'forward'):
            pred = torch.sigmoid(model(img_tensor)).squeeze().numpy()
        else:
            print("Error: Model cannot make predictions without segmentation_models_pytorch")
            return np.zeros((256, 256), dtype=np.uint8)
            
        return (pred > 0.5).astype(np.uint8)
