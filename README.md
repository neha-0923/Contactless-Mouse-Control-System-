EYEPRO — Hands-Free Computer Control

EYEPRO is a Python-based assistive technology tool that lets you control your computer using eye tracking, hand gestures, or a combination of both — no mouse or keyboard needed.


✨ Features

ModeDescription👁️ Eye ControlMove the cursor with your iris, click by blinking or dwelling🖐️ Hand ControlMove the cursor and trigger actions using hand gestures🔀 Dual ControlEyes move the cursor; hands trigger clicks and scrolls


🖱️ Controls

Eye Mode

ActionGestureMove cursorMove your irisLeft clickLong blink (both eyes)Double clickBlink left eye onlyRight clickBlink right eye onlyScroll up/downLook toward top/bottom of frameMinimize windowSquint both eyesAuto-clickDwell on target (hold gaze)

Hand Mode

ActionGestureMove cursorMove index fingertipLeft clickPinch (thumb + index finger close)Right clickIndex finger up 👆Double clickIndex + middle fingers up ✌️Scroll upThumb up 👍Scroll downPinky up 🤙Minimize windowClosed fist ✊

Dual Mode

Eyes move the cursor; all hand gestures from Hand Mode apply for clicks and scrolls.


🚀 Getting Started

Prerequisites


Python 3.8+
Webcam


Installation

bash# Clone the repository
git clone https://github.com/your-username/eyepro.git
cd eyepro

# Install dependencies
pip install -r requirements.txt

Dependencies

flask
opencv-python
mediapipe
numpy
pyautogui
scikit-learn
filterpy

Running the App

bashpython app.py

This will start a local Flask server and open the control panel in your browser at http://127.0.0.1:5000/.

From there, select a control mode (Eye / Hand / Dual) to begin.


⚙️ Configuration

Settings are stored in config.json:

json{
  "smoothing_window": 5,
  "dwell_time": 0.7,
  "calibration_duration": 5,
  "mouse_speed_factor": 2.0,
  "min_distance_threshold": 0.01
}

ParameterDescriptionsmoothing_windowNumber of frames averaged to smooth cursor movementdwell_timeSeconds of held gaze before auto-click triggers (seconds)calibration_durationDuration of the eye calibration phase on startup (seconds)mouse_speed_factorMultiplier for cursor movement speedmin_distance_thresholdMinimum iris movement required to move the cursor


🧠 How It Works

Webcam Feed
    │
    ├── FaceTracker (MediaPipe Face Mesh)
    │     ├── Iris position → cursor movement
    │     ├── EAR (Eye Aspect Ratio) → blink detection
    │     ├── Kalman filter → smooth head pose estimation
    │     └── SVM classifier → eye open/closed state
    │
    ├── HandTracker (MediaPipe Hands)
    │     ├── Landmark positions → gesture recognition
    │     └── Finger state detection → click/scroll actions
    │
    └── MouseController (PyAutoGUI)
          ├── Smoothed cursor movement
          └── Dwell-click logic

Calibration runs automatically on startup. You'll be prompted to look straight ahead, then blink a few times. This sets personalised EAR thresholds for accurate blink detection.


📁 Project Structure

eyepro/
├── app.py               # Flask web server and route definitions
├── main_controller.py   # Core control loop for all modes
├── face_tracker.py      # Eye/iris/head pose tracking via MediaPipe
├── hand_tracker.py      # Hand gesture recognition via MediaPipe
├── mouse_controller.py  # Cursor movement, smoothing, and click logic
├── config.json          # Runtime configuration
└── templates/
    └── index.html       # Web UI for mode selection


🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.


📄 License

MIT
