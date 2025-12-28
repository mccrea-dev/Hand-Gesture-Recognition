# Gesture Recognition with IMU Sensors

## Description

This project implements a real-time hand gesture recognition system using Seeed Studio IMU sensors (accelerometer and gyroscope). It collects sensor data for various hand gestures, extracts statistical features via sliding windows, trains a machine learning model using Random Forest, and performs live prediction via serial connection.

**Supported Gestures:**
- **Petting**: Hand near the floor slowly stroking right to left as if petting a cat.
- **Waving**: Normal waving, moving hand left to right with a 90° angle in the elbow.
- **Raise the Roof**: Hands above your head pumping up and down repeatedly.
- **Rotate Wrist**: Turning your hand from palm down to palm up or vice versa.
- **Stationary**: Holding your hand still.

The system uses a sliding window approach (250ms windows with 50% overlap at 100Hz sampling) to extract features like mean, standard deviation, RMS, peak-to-peak, and activity level from IMU data.

## Requirements

- Python 3.x
- Libraries: `numpy`, `pandas`, `scikit-learn`, `joblib`, `matplotlib`, `pyserial`

Install dependencies:
```bash
pip install numpy pandas scikit-learn joblib matplotlib pyserial
```

## Hardware Setup

- **IMU Sensor**: Seeed Studio board with built-in IMU (e.g., XIAO nRF52840 Sense with BMI088 IMU).
- **Connection**: USB serial at 115200 baud.
- **Firmware**: The board should output CSV-formatted IMU data: `timestamp_ms,ax,ay,az,gx,gy,gz[,tempC]`.

## Project Workflow

### 1. Data Collection

Collect training data for each gesture using `mlogger.py`. Perform each gesture multiple times and save to separate CSV files.

```bash
# List available ports
python mlogger.py --show-ports

# Log data for a gesture (e.g., petting)
python mlogger.py --port COM5 --out petting1.csv --duration 10

# Repeat for other gestures: waving1.csv, raise_roof1.csv, etc.
```

- `--port`: Serial port (e.g., COM5 on Windows, /dev/ttyACM0 on Linux).
- `--duration`: Recording time in seconds.
- `--out`: Output CSV filename.

### 2. Feature Extraction

Process the collected CSV files to extract features using sliding windows.

```bash
# Process all CSV files in the directory
python step01.py *.csv
```

This generates:
- `features_all.csv`: All extracted features.
- `features_schema.json`: Feature column schema.
- `features/` directory: Individual feature CSVs.

**Constants** (adjust in `step01.py` if needed):
- `WIN = 25`: Window size (25 samples at 100Hz = 250ms).
- `HOP = 12`: Hop size (50% overlap).
- `AXES = ["ax","ay","az","gx","gy","gz"]`: IMU axes.

### 3. Clustering and Filtering

Use DBSCAN to cluster the features and filter for high-activity windows (top 30% per gesture).

```bash
python dbscan.py
```

This outputs:
- `features_top_activity.csv`: Filtered features for training.
- `training_clean.csv`: Features with DBSCAN labels.
- PCA plot showing clusters.

**Parameters** (adjust in `dbscan.py`):
- `EPS = 1.6`: DBSCAN epsilon.
- `MINPTS = 25`: Minimum samples per cluster.

### 4. Model Training

Train a Random Forest classifier on the filtered features.

```bash
python forest.py
```

This saves `gesture_model.pkl` (the trained model).

**Model Parameters** (in `forest.py`):
- `n_estimators=300`
- `max_features="sqrt"`

### 5. Visualization

#### PCA Visualization
```bash
python pca.py
```
Shows 2D and 3D PCA plots of the features, colored by gesture.

#### Decision Boundary
```bash
python decision_boundary.py
```
Displays decision boundaries on the PCA 2D projection.

### 6. Live Prediction

Perform real-time gesture recognition.

```bash
# Auto-detect port
python live_predictor.py

# Specify port
python live_predictor.py --port COM5

# Show available ports
python live_predictor.py --show-ports
```

- `--port`: Serial port.
- `--baud`: Baud rate (default 115200).
- `--duration`: Run for N seconds.
- `--quiet`: Less output.

The script connects to the IMU sensor, extracts features in real-time, and prints predicted gestures.

## File Structure

- `live_predictor.py`: Real-time gesture predictor.
- `mlogger.py`: IMU data logger from serial port.
- `step01.py`: Feature extraction from CSV data.
- `dbscan.py`: DBSCAN clustering and activity filtering.
- `forest.py`: Random Forest model training.
- `pca.py`: PCA visualization.
- `decision_boundary.py`: Decision boundary plot on PCA.
- `features_all.csv`: All extracted features.
- `features_top_activity.csv`: Filtered high-activity features.
- `training_clean.csv`: Training data with cluster labels.
- `gesture_model.pkl`: Trained model.
- `features_schema.json`: Feature schema.
- `features/`: Directory with per-file features.
- Various extra gesture CSV directories: `petting/`, `waving/`, etc.

## Customization

- **Gestures**: Add new gestures by collecting data and retraining the model.
- **Features**: Modify `compute_features()` in `live_predictor.py` and `features_from_csv()` in `step01.py`.
- **Model**: Experiment with different classifiers in `forest.py`.
- **Windowing**: Adjust `WIN` and `HOP` for different temporal resolutions *not suggested.

## Troubleshooting

- **No serial data**: Ensure the board is connected, firmware is running, and baud rate matches (115200).
- **Low accuracy**: Collect more data, adjust DBSCAN parameters, or tune the model.
- **Port issues**: Use `--show-ports` to list available ports.
