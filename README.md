
# Face Recognition Based Attendance System

An easy-to-run Flask + OpenCV prototype for small-scale, in-person attendance tracking using face recognition. The app:

- Captures face images from a webcam.
- Trains a small scikit-learn KNN classifier on saved face crops.
- Identifies users during a live capture session and records daily attendance rows to CSV files.

This README explains how the project is organized, how to run it locally on Windows, and how the core flows (add user, start attendance) work.

## Quick summary

- Web app served by `app.py` (Flask).
- Face detection: OpenCV Haar cascade (`haarcascade_frontalface_default.xml`).
- Face storage: `static/faces/{Name}_{Roll}/` contains per-user face crops.
- Trained model file: `static/face_recognition_model.pkl` (KNN on resized crops).
- Attendance output: `Attendance/Attendance-<MM_DD_YY>.csv` (one CSV per day).

## Features

- Add new users via the web UI and capture face images from the webcam.
- Retrain the recognition model after new users are added.
- Start a live attendance session that identifies faces and appends unique Name/Roll rows to today's CSV.
- Simple camera backend fallback and warm-up logic to improve reliability on Windows.

## Repository layout (important files)

- `app.py` — main Flask application and capture/training logic.
- `haarcascade_frontalface_default.xml` — Haar cascade used for face detection.
- `static/faces/` — directory containing subfolders per user with saved face images.
- `static/face_recognition_model.pkl` — trained KNN model (auto-generated after training).
- `Attendance/` — generated CSV files for each day.
- `test_camera.py` — quick script to test which OpenCV camera backend works on your machine.

## Quick requirements

- Python 3.8+ (project tested on Python 3.13 but should work on 3.8+). If you must match the environment in the repo use Python 3.13.
- See `requirements.txt` for exact pins used in development. Main packages:
  - flask
  - opencv-python
  - scikit-learn
  - pandas
  - joblib

## Install (Windows / PowerShell)

1. Create and activate a virtual environment (optional but recommended):

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

If you don't have `requirements.txt`, install the common deps:

```powershell
pip install flask opencv-python scikit-learn pandas joblib
```

## Run the app locally

From the project root:

```powershell
# If you use Flask CLI
flask run
# Or run the script directly with the same interpreter
python app.py
```

Open http://127.0.0.1:5000 in your browser.

Note: The app opens OpenCV GUI windows for capture. These windows require a local (non-headless) session and keyboard interaction (press ESC or Q to stop capture loops).

## Main flows

### Add a new user (capture)

1. Use the web UI's "Add user" form or POST to `/add` with `newusername` and `newuserid`.
2. The app creates a folder `static/faces/{Name}_{Roll}`.
3. The server opens the webcam, detects faces, crops and saves images into the user's folder.
4. After capture the model is retrained and saved to `static/face_recognition_model.pkl`.

Implementation notes:
- The code currently targets ~50 saved images per user by default (`nimgs` in `app.py`). You can change this constant or the saving cadence.
- Folder and label format must be `Name_Roll` (underscore) to match label parsing used during training.

### Take attendance (live recognition)

1. Click "Take Attendance" in the web UI or visit `/start`.
2. The app opens the webcam and processes frames in a loop:
   - Detect faces with Haar cascade.
   - Crop and resize detections (50x50), then classify with the KNN model.
   - For each recognized `Name_Roll`, append a row to `Attendance/Attendance-<MM_DD_YY>.csv` if that Name+Roll isn't already present for today.
3. Stop the capture by focusing the GUI window and pressing ESC or Q.

CSV format (columns):

- Date, Time, Name, Roll (the exact column names depend on the implementation in `app.py`, but CSVs are plain text and viewable in Excel).

## Training & model details

- A simple K-Nearest Neighbors classifier is used on raw resized face crops. This is intentionally lightweight and easy to reason about.
- Model artifacts are saved with `joblib` to `static/face_recognition_model.pkl`.
- For better accuracy consider replacing the raw-pixel approach with an embedding model (FaceNet, ArcFace) and a small classifier on top.

## Troubleshooting

- Camera errors (MSMF / DirectShow) on Windows:
  - Close other applications that might be using the camera.
  - Check Windows Settings → Privacy → Camera to ensure access is allowed.
  - Reconnect camera or reboot.
  - See `test_camera.py` to probe which backend works best.

- Headless/deployed servers:
  - OpenCV GUI windows won't work in a headless environment. For remote deployments replace GUI capture with a web-based capture flow (e.g., using browser JS to post frames to the server).

- Incorrect labels / missing users:
  - Ensure user folder names use the `Name_Roll` format.
  - Re-run the training route or restart the server after adding new images if the model file isn't updated.

## Development notes

- Code is intentionally minimal for learning and prototyping.
- If you plan to extend the app consider:
  - Replacing the KNN+raw-pixel pipeline with a feature embedding model.
  - Moving model training to a background worker so add-user requests return quickly.
  - Adding authentication around add/start routes to restrict who can modify attendance data.

## Tests

- `test_camera.py` can help identify camera backend issues on Windows. Run it with the same interpreter used to run the app:

```powershell
python test_camera.py
```

## License & contact

MIT