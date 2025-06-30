@echo off
echo Creating Python virtual environment...
python -m venv venv
call venv\Scripts\activate

echo Installing required packages...
pip install --upgrade pip
pip install streamlit torch torchvision matplotlib numpy opencv-python pillow

echo Installing GDAL using Christoph Gohlke's wheels...
pip install https://download.lfd.uci.edu/pythonlibs/archived/cp310/GDAL-3.4.3-cp310-cp310-win_amd64.whl

echo Installing segmentation models...
pip install segmentation-models-pytorch

echo Setup complete! Run the app with: streamlit run app.py
