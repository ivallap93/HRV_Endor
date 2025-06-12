# HRV Feature Analysis

This repository provides Python-based tools to compute and visualize Heart Rate Variability (HRV) features from RR interval data, including:

- **Time-domain features**
- **Spectral-domain features**
- **Poincaré plots**
- **Visualization of raw heart rate and accelerometer signals**
- **Training load distribution based on heart rate zones**

---

## 📁 Folder Structure

```
.
├── data/
│   ├── HRV_anonymized_data/        # Folder containing heart rate (RR interval) files
│   └── accelerometer_anonymized_data/  # Folder containing accelerometer files
├── framing.py                      # External helper for signal framing (user-defined)
├── hrv_analysis.py                # Main script to run HRV analysis
```

---
## Data Source

The datasets used in this project are publicly available and were obtained from [Zenodo](https://zenodo.org/records/8171266).  
Please refer to the original source for more information and licensing details.

## 📊 Features Extracted

### Time Domain Features

- Mean and standard deviation of heart rate
- RMSSD (Root Mean Square of Successive Differences)
- SDSD (Standard Deviation of Successive Differences)
- SDNN (Standard Deviation of NN intervals)
- SD1, SD2 (from Poincaré plot)
- RR mean, median, and variance

### Frequency Domain Features

- VLF, LF, HF power using Welch PSD method
- LF/HF ratio

### Non-linear Feature

- **Poincaré Plot**:
  - SD1: Short-term HRV
  - SD2: Long-term HRV
  - SD1/SD2 ratio

---

## 🖼️ Visualization Tools

- Histogram of heart rate
- Training zone intensity based on % of HRmax
- Raw signal plots of HR and accelerometer data
- Poincaré plot with SD1/SD2 ellipse

---

## ▶️ How to Use

1. Place HR and accelerometer `.csv` files in the appropriate folders.
2. Run the script `hrv_analysis.py`.
3. Enter the base filename (e.g., `treatment_1`) when prompted.
4. The script will:
   - Plot raw signals
   - Visualize heart rate histogram
   - Show training zones
   - Compute and print HRV features
   - Display Poincaré and spectral plots

---

## ⚠️ Requirements

Install dependencies via pip:

```bash
pip install -r requirements.txt
```

---


## 📄 License

This project is for research and educational purposes.

---

## 🙋 Acknowledgements

This work uses anonymized data for HRV and accelerometer analysis. It includes contributions from biomedical signal processing enthusiasts and domain experts.
