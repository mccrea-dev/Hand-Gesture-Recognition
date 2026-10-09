# Gesture Recognition with IMU Sensors

## Description

This project implements a real-time hand gesture recognition system using a Seeed Studio IMU sensor (accelerometer and gyroscope). It collects sensor data for various hand gestures, extracts statistical features via sliding windows, trains a Random Forest classifier, and performs live prediction over a serial connection.

**Supported Gestures** (with the label the model outputs):

| Gesture | Label | Description |
|---|---|---|
| Petting | `petting` | Hand near the floor slowly petting left to right as if petting a cat. |
| Waving | `waving` | Normal waving, moving hand left to right with a 90° angle in the elbow. |
| Raise the Roof | `raise` | Hands above your head pumping up and down repeatedly. |
| Rotate Wrist | `rotate` | Turning your hand from palm down to palm up or vice versa. |
| Stationary | `stationary` | Holding your hand still. |

The system uses a sliding window approach (250 ms windows with 50% overlap at 100 Hz sampling) to extract features like mean, standard deviation, RMS, RMS of the first difference, peak-to-peak, and an overall activity level from the IMU data.

## Repository Layout

```
Hand-Gesture-Recognition/
├── src/                        # Core pipeline
│   ├── mlogger.py              # Log IMU data from the serial port to CSV
│   ├── step01.py               # Sliding-window feature extraction
│   ├── forest.py               # Random Forest training
│   └── live_predictor.py       # Real-time gesture prediction
├── analysis/                   # Exploration and visualization
│   ├── dbscan.py               # Activity filtering + DBSCAN clustering
│   ├── pca.py                  # 2D/3D PCA plots
│   └── decision_boundary.py    # Decision boundaries on PCA projection
├── data/
│   ├── raw/<gesture>/          # Raw IMU recordings, one folder per gesture
│   ├── interim/features/       # Per-recording feature CSVs
│   └── processed/              # features_all.csv, features_top_activity.csv,
│                               # training_clean.csv, features_schema.json
└── models/
    └── gesture_model.pkl       # Trained model
```

`data/raw/` also contains recordings that were **not** used to train the current model (`adjusting_hair`, `punching`, `running`, `wave_dance`, `petting1-4`, `stationary_1-5`). The current model was trained on `petting_redone2-5`, `raise_roof1-4`, `rotate_wrist1-5`, `stationary_6-8`, and `waving_1-4`.

## Requirements

- Python 3.x
- Libraries: `numpy`, `pandas`, `scikit-learn`, `joblib`, `matplotlib`, `pyserial`

```bash
pip install numpy pandas scikit-learn joblib matplotlib pyserial
```

## Hardware Setup

- **IMU Sensor**: Seeed Studio board with built-in IMU (e.g., XIAO nRF52840 Sense).
- **Connection**: USB serial at 115200 baud.
- **Firmware**: The board should output CSV-formatted IMU data: `timestamp_ms,ax,ay,az,gx,gy,gz[,tempC]`.

## Quick Start (Pretrained Model)

The trained model is already in `models/`. Plug in the board and run the predictor **from the `models/` directory** (it loads `gesture_model.pkl` from the current directory):

```bash
cd models
python ../src/live_predictor.py              # auto-detect port
python ../src/live_predictor.py --port COM5  # or specify a port
```

## Working Directories

The scripts read and write their files relative to the **current working directory**, not the repo layout. The commands below are written to run from the directory where each file lives:

| Script | Reads | Writes | Run from |
|---|---|---|---|
| `src/step01.py` | CSVs passed as arguments | `features_all.csv`, `features_schema.json`, `features/` | `data/processed` |
| `analysis/dbscan.py` | `features_all.csv` | `features_top_activity.csv`, `training_clean.csv` | `data/processed` |
| `analysis/pca.py` | `features_all.csv` | plots only | `data/processed` |
| `analysis/decision_boundary.py` | `features_top_activity.csv` | plots only | `data/processed` |
| `src/forest.py` | `features_top_activity.csv` | `gesture_model.pkl` | `data/processed` (then move model to `models/`) |
| `src/live_predictor.py` | `gesture_model.pkl` | — | `models` |

## Training Workflow

### 1. Data Collection

Record each gesture several times with `mlogger.py`, one CSV per recording.

```bash
# List available ports
python src/mlogger.py --show-ports

# Record 10 seconds of a gesture
python src/mlogger.py --port COM5 --out data/raw/petting/petting_01.csv --duration 10
```

- `--port`, `-p`: Serial port (e.g., `COM5` on Windows, `/dev/ttyACM0` on Linux). Auto-detected if omitted.
- `--out`, `-o`: Output CSV path (default: `imu_session_YYYYmmdd_HHMMSS.csv`).
- `--duration`: Recording time in seconds (default: until Ctrl-C).
- `--append`: Append to the output file if it exists.
- `--baud`, `-b`, `--no-auto`, `--quiet`: Same as the live predictor.

> **File naming matters:** the gesture label is taken from the filename up to the first `_` (e.g., `raise_roof3.csv` → `raise`, `petting_redone2.csv` → `petting`). Name recordings `<label>_<anything>.csv`.

### 2. Feature Extraction

Pass the recordings you want to train on to `step01.py` (it takes CSV files, not directories):

```bash
cd data/processed
python ../../src/step01.py ../raw/petting/petting_redone*.csv ../raw/raise_roof/*.csv \
    ../raw/rotate_wrist/*.csv ../raw/stationary/stationary_[678].csv ../raw/waving/*.csv
```

(This uses bash wildcards and line continuation. PowerShell and cmd don't expand `*.csv` for Python, so use Git Bash or list the files explicitly.)

This generates `features_all.csv`, `features_schema.json`, and a `features/` directory of per-recording feature CSVs (the copy kept in the repo is in `data/interim/features/`).

**Constants** (in `step01.py`):
- `WIN = 25`: Window size (25 samples at 100 Hz = 250 ms).
- `HOP = 12`: Hop size (50% overlap).
- `AXES = ["ax","ay","az","gx","gy","gz"]`: IMU axes.

### 3. Activity Filtering and Clustering

Keep the top 30% highest-activity windows per gesture, then run DBSCAN on them:

```bash
python ../../analysis/dbscan.py
```

Outputs:
- `features_top_activity.csv`: Filtered features used for training.
- `training_clean.csv`: Filtered features with a `dbscan_label` column.
- A PCA plot of the clusters.

**Parameters** (in `dbscan.py`): `EPS = 1.6`, `MINPTS = 25`, `TOP_PERCENT = 0.30`.

### 4. Model Training

```bash
python ../../src/forest.py
mv gesture_model.pkl ../../models/
```

**Model parameters** (in `forest.py`): `n_estimators=300`, `max_features="sqrt"`, `random_state=42`.

### 5. Visualization (optional)

```bash
python ../../analysis/pca.py                # 2D and 3D PCA plots colored by gesture
python ../../analysis/decision_boundary.py  # decision boundaries on the 2D PCA projection
```

### 6. Live Prediction

```bash
cd ../../models
python ../src/live_predictor.py --port COM5
```

- `--port`, `-p`: Serial port (auto-detected if omitted).
- `--baud`, `-b`: Baud rate (default 115200).
- `--duration`: Stop after N seconds (default: until Ctrl-C).
- `--no-auto`: Don't auto-pick a port; require `--port`.
- `--show-ports`: List available ports and exit.
- `--quiet`, `-q`: Less output.

The predictor buffers incoming samples, computes features for each 25-sample window, and prints the predicted gesture every 12 samples.

## Customization

- **Gestures**: Record a new gesture following the naming convention above, re-run steps 2–4, and include its files in the `step01.py` command.
- **Features**: Keep `compute_features()` in `live_predictor.py` and `features_from_csv()` in `step01.py` in sync. The live feature order must match the training column order.
- **Model**: Experiment with different classifiers in `forest.py`.
- **Windowing**: `WIN` and `HOP` can be changed in `step01.py` (the live predictor imports them), but this isn't recommended. The model must be retrained after any change.

## Troubleshooting

- **No serial data**: Make sure the board is connected, the firmware is running, the baud rate is 115200, and no other program (e.g., Arduino Serial Monitor) has the port open.
- **`FileNotFoundError` for a CSV or `gesture_model.pkl`**: You're running from the wrong directory. See [Working Directories](#working-directories).
- **Low accuracy**: Collect more data, adjust `TOP_PERCENT` or the DBSCAN parameters, or tune the model.
- **Port issues**: Use `--show-ports` to list available ports.
