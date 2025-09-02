etup Guide for AI Proctoring Project
1. Install Python

Use Python 3.10.x (64-bit) (because Mediapipe doesn’t support the latest Python versions yet).

Download from: https://www.python.org/downloads/release/python-3109/

During install, check “Add Python to PATH”.

2. Create Project Folder
cd Desktop
mkdir MinorProject
cd MinorProject

3. Create & Activate Virtual Environment
python -m venv venv
venv\Scripts\activate

4. Install Dependencies

Run these inside the activated venv:

pip install opencv-python mediapipe numpy pandas sounddevice


⚠️ If pyaudio is required for microphone input:

pip install pyaudio


If that fails, download .whl from:
👉 https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

Then install with:

pip install path\to\PyAudio-0.2.11-cp310-cp310-win_amd64.whl

5. Save the Code

Save the script as interview_monitor.py inside the MinorProject folder.

6. Run the Project

From inside MinorProject (and with venv active):

py interview_monitor.py
