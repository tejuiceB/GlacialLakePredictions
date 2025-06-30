@echo off
echo Installing required packages for Glacial Lake Segmentation app...

pip install streamlit torch torchvision matplotlib numpy opencv-python pillow
pip install segmentation-models-pytorch albumentations
pip install requests gdown rasterio scikit-image
pip install geojson shapely

echo Installation complete!
echo Run the app with: streamlit run app.py
