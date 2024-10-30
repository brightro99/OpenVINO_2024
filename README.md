# OpenVINO 2024.0

## INSTALL
### OpenVINO
[DOCS](https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-archive-windows.html)  
1. Command Prompt from the Start menu and select Run as administrator
   ```shell
   mkdir "C:/Program Files (x86)/Intel"
   ```
2. Download the [OpenVINO Runtime archive file for Windows](https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.4/windows) to your local Downloads folder.
   ```shell
   cd <user_home>/Downloads
   curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.4/windows/w_openvino_toolkit_windows_2024.4.0.16579.c3152d32c9c_x86_64.zip --output openvino_2024.4.0.zip
   ```
3. move it to the C:/Program Files (x86)/Intel directory.
   ```shell
   tar -xf openvino_2024.4.0.zip
   ren w_openvino_toolkit_windows_2024.4.0.16579.c3152d32c9c_x86_64 openvino_2024.4.0
   move openvino_2024.4.0 "C:/Program Files (x86)/Intel"
   ```
4. (Optional) Install numpy Python Library
   ```shell
   cd "C:/Program Files (x86)/Intel/openvino_2024.4.0"
   python -m pip install -r ./python/requirements.txt
   ```
### OpenCV
[OpenCV 4.10.0](https://github.com/opencv/opencv/releases/download/4.10.0/opencv-4.10.0-windows.exe)
1. Extract OpenCV
2. Copy to OpenVINO 3rdparty folder
   ```
   mkdir "C:/Program Files (x86)/Intel/openvino_2024.4.0/runtime/3rdparty/opencv-4.10.0"
   cd <Extract OpenCV Folder>
   xcopy "build" "C:/Program Files (x86)/Intel/openvino_2024.4.0/runtime/3rdparty/opencv-4.10.0" /E /H /I
   ```

## Configure the Environment
```shell
"C:/Program Files (x86)/Intel/openvino_2024/setupvars.bat"
```
or
## CMake

## Python
```
conda create -n ov20240 python=3.10
conda activate ov20240
cd C:/Program Files (x86)/Intel/openvino_2024.4.0/python
pip install -r requirements.txt
../setupvars.bat
```
### 
OpenVINO Model ZOO - [Model Download](https://github.com/openvinotoolkit/open_model_zoo/blob/master/tools/model_tools/README.md)
```
pip install openvino-dev==2024.4.0
omz_downloader --all
```
