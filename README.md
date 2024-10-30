# OpenVINO 2024.0

## INSTALL
[DOCS](https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-archive-windows.html)  

1. Command Prompt from the Start menu and select Run as administrator
   ```shell
   mkdir "C:\Program Files (x86)\Intel"
   ```
2. Download the [OpenVINO Runtime archive file for Windows](https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.4/windows) to your local Downloads folder.
   ```shell
   cd <user_home>/Downloads
   curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.4/windows/w_openvino_toolkit_windows_2024.4.0.16579.c3152d32c9c_x86_64.zip --output openvino_2024.4.0.zip
   ```
3. move it to the C:\Program Files (x86)\Intel directory.
   ```shell
   tar -xf openvino_2024.4.0.zip
   ren w_openvino_toolkit_windows_2024.4.0.16579.c3152d32c9c_x86_64 openvino_2024.4.0
   move openvino_2024.4.0 "C:\Program Files (x86)\Intel"
   ```
4. (Optional) Install numpy Python Library
   ```shell
   cd "C:\Program Files (x86)\Intel\openvino_2024.4.0"
   python -m pip install -r .\python\requirements.txt
   ```
   
## Configure the Environment
```shell
"C:\Program Files (x86)\Intel\openvino_2024\setupvars.bat"
```

## CMake
