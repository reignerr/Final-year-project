# Final-year-project
Development of an Air Quality Monitoring Device with Predictive Model
# Air Quality Monitoring and Prediction System

A low-cost IoT-based air quality monitoring system with deep learning-powered AQI prediction using an LSTM neural network.

---

## 📌 Project Overview

This project combines embedded hardware and machine learning to monitor, log, and predict air quality in real time. A custom-built sensor node collects particulate matter (PM2.5, PM10), temperature, and humidity data, which is then used to train a Long Short-Term Memory (LSTM) model to forecast the Air Quality Index (AQI).

---

## 🛠️ System Components

### Hardware
- **PMS5003** — Particulate matter sensor (PM1.0, PM2.5, PM10)
- **DHT11** — Temperature and humidity sensor
- **DS3231 RTC** — Real-time clock for timestamping
- **16x2 LCD** — Live data display
- **SD Card Module** — Local data logging in CSV format
- **Arduino Microcontroller** — Central processing unit

### Software
- **Arduino IDE** — Microcontroller firmware (`sensor_monitor.ino`)
- **Python / Google Colab** — Data processing and model training (`aqi_prediction.py`)
- **TensorFlow / Keras** — LSTM model development
- **Pandas, NumPy, Matplotlib, Scikit-learn** — Data analysis and visualisation

---

## 📁 Repository Structure

```
air-quality-monitoring-system/
├── sensor_monitor.ino      # Arduino firmware for sensor data collection and logging
├── aqi_prediction.py       # Python script for LSTM-based AQI prediction
└── README.md               # Project documentation
```

---

## ⚙️ How It Works

1. **Data Collection** — The Arduino reads PM2.5, PM10, temperature, and humidity every few seconds, timestamps each reading using the RTC, displays it on the LCD, and saves it to an SD card in CSV format.

2. **Data Preprocessing** — The CSV dataset is transferred to Google Colab, cleaned, normalised using MinMaxScaler, and restructured into time-series sequences using a sliding window approach.

3. **Model Training** — An LSTM neural network is trained on 80% of the data using the Adam optimiser and Mean Squared Error (MSE) loss function.

4. **Prediction & Evaluation** — The trained model predicts AQI on unseen test data. Performance is evaluated using MSE and MAE metrics, and results are visualised with Matplotlib.

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Training Loss (MSE) | 0.012 |
| Validation Loss (MSE) | 0.015 |
| Optimiser | Adam |
| Epochs | 20 |
| Batch Size | 32 |

---

## 🚀 Getting Started

### Arduino (Hardware Setup)
1. Install the [Arduino IDE](https://www.arduino.cc/en/software)
2. Install required libraries: `PMS5003`, `DHT`, `RTClib`, `SD`, `LiquidCrystal_I2C`
3. Upload `sensor_monitor.ino` to your Arduino board

### Python (Model Training)
1. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn tensorflow matplotlib
   ```
2. Run the script:
   ```bash
   python aqi_prediction.py
   ```
   Or open it directly in [Google Colab](https://colab.research.google.com/)

---

## 📖 AQI Reference

| AQI Range | Category | PM2.5 (µg/m³) |
|---|---|---|
| 0 – 50 | Good | 0.0 – 12.0 |
| 51 – 100 | Moderate | 12.1 – 35.4 |
| 101 – 150 | Unhealthy for Sensitive Groups | 35.5 – 55.4 |
| 151 – 200 | Unhealthy | 55.5 – 150.4 |
| 201 – 300 | Very Unhealthy | 150.5 – 250.4 |
| 301 – 500 | Hazardous | 250.5 – 500.4 |

---

## 👤 Author

**George Reigner**  
Final Year Project — Covenant University

---

## 📄 License

This project was developed for academic purposes at Covenant University. All rights reserved.
