import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.interpolate import interp1d
from typing import Dict, List, Tuple
from matplotlib.patches import Ellipse

# Your helper functions here (extract_hrv_time_domain_features, extract_hrv_spectral_domain_features, plot_poincare)
def extract_hrv_time_domain_features(rr_intervals_ms: np.ndarray) -> Dict[str, float]:
    """Compute HRV time-domain features from RR intervals in milliseconds."""
    rr_intervals_ms = rr_intervals_ms[~np.isnan(rr_intervals_ms)]
    if rr_intervals_ms.size < 2:
        raise ValueError("Too few RR intervals for time-domain analysis.")

    rr_diffs = np.diff(rr_intervals_ms)
    hr = np.where(rr_intervals_ms == 0, np.nan, 60_000 / rr_intervals_ms)

    features = {
        "HR_AVG": np.nanmean(hr),
        "HR_Max": np.nanmax(hr),
        "HR_std": np.nanstd(hr),
        "RR_RMSSD": np.sqrt(np.mean(rr_diffs ** 2)),
        "RR_SDSD": np.std(rr_diffs, ddof=1),
        "RR_SDNN": np.std(rr_intervals_ms, ddof=1),
        "RR_SD1": np.sqrt(0.5) * np.std(rr_diffs, ddof=0),
        "RR_mean": np.mean(rr_intervals_ms),
        "RR_median": np.median(rr_intervals_ms),
        "RR_variance": np.var(rr_intervals_ms, ddof=1),
    }

    return features


def extract_hrv_spectral_domain_features(rr_intervals_ms: np.ndarray,
                                         fs_interp: int = 4,
                                         plot: bool = True) -> Dict[str, float]:
    """Compute HRV frequency-domain features using Welch's method."""
    time_rr = np.cumsum(rr_intervals_ms) / 1000  # convert ms to seconds for time axis
    time_interp = np.arange(0, time_rr[-1], 1 / fs_interp)
    rr_interp_func = interp1d(time_rr, rr_intervals_ms / 1000, kind='cubic', fill_value="extrapolate")
    rr_interp = rr_interp_func(time_interp)

    fxx, pxx = welch(rr_interp, fs=fs_interp, nperseg=256)

    def band_power(f, p, band):
        mask = (f >= band[0]) & (f <= band[1])
        return np.trapz(p[mask], f[mask]), mask

    vlf_band = (0.003, 0.04)
    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.4)

    vlf_power, vlf_mask = band_power(fxx, pxx, vlf_band)
    lf_power, lf_mask = band_power(fxx, pxx, lf_band)
    hf_power, hf_mask = band_power(fxx, pxx, hf_band)
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else np.inf

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(fxx, pxx, label='PSD')
        plt.fill_between(fxx[vlf_mask], pxx[vlf_mask], color='blue', alpha=0.3, label='VLF (0.003–0.04 Hz)')
        plt.fill_between(fxx[lf_mask], pxx[lf_mask], color='orange', alpha=0.4, label='LF (0.04–0.15 Hz)')
        plt.fill_between(fxx[hf_mask], pxx[hf_mask], color='green', alpha=0.4, label='HF (0.15–0.4 Hz)')
        plt.title('HRV Spectral Analysis')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (s²/Hz)')
        plt.legend()
        plt.grid(True)
        plt.text(0.42, max(pxx) * 0.8, f'LF Power: {lf_power:.4f}\nHF Power: {hf_power:.4f}\nLF/HF Ratio: {lf_hf_ratio:.2f}',
                 bbox=dict(facecolor='white', alpha=0.7))
        plt.tight_layout()
        plt.show()

    return {
        'VLF_power': vlf_power,
        'LF_power': lf_power,
        'HF_power': hf_power,
        'LF_HF_ratio': lf_hf_ratio
    }


def plot_poincare(rr_intervals_ms: np.ndarray,
                  show_axes: bool = True,
                  ellipse: bool = True) -> Dict[str, float]:
    """Plot the Poincaré plot and return SD1, SD2 metrics."""
    rr_n = rr_intervals_ms[:-1]
    rr_n1 = rr_intervals_ms[1:]

    diff = rr_n1 - rr_n
    sum_ = rr_n1 + rr_n

    sd1 = np.sqrt(np.var(diff) / 2)
    sd2 = np.sqrt(np.var(sum_) / 2)

    plt.figure(figsize=(6, 6))
    plt.scatter(rr_n, rr_n1, alpha=0.3, color='blue', label='RRn+1 vs RRn')
    plt.title('Poincaré Plot of HRV')
    plt.xlabel('RRₙ (ms)')
    plt.ylabel('RRₙ₊₁ (ms)')
    plt.axis('equal')
    plt.grid(True)

    rr_mean = np.mean(rr_intervals_ms)
    plt.plot(rr_mean, rr_mean, 'ro', label='Mean RR')

    if show_axes:
        plt.plot([rr_mean - sd2, rr_mean + sd2], [rr_mean - sd2, rr_mean + sd2], 'k--', label='SD2 axis')
        plt.plot([rr_mean - sd1, rr_mean + sd1], [rr_mean + sd1, rr_mean - sd1], 'r--', label='SD1 axis')

    if ellipse:
        ell = Ellipse((rr_mean, rr_mean), width=2 * sd2, height=2 * sd1,
                      angle=45, edgecolor='green', facecolor='none', linestyle='--', label='Ellipse (SD1/SD2)')
        plt.gca().add_patch(ell)

    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        'SD1': sd1,
        'SD2': sd2,
        'SD1/SD2_ratio': sd1 / sd2 if sd2 != 0 else np.inf
    }


def load_and_process_accelerometer(acc_file: str) -> pd.DataFrame:
    """Load accelerometer CSV file and return processed dataframe."""
    df_acceleration = pd.read_csv(acc_file)
    df_acceleration.columns = ['raw']
    df_split_acceleration = df_acceleration['raw'].str.split(';', expand=True)
    df_split_acceleration.columns = ['timestamp', 'val1', 'val2', 'val3']
    df_split_acceleration['val1'] = df_split_acceleration['val1'].astype(int)
    df_split_acceleration['val2'] = df_split_acceleration['val2'].astype(int)
    df_split_acceleration['val3'] = df_split_acceleration['val3'].astype(int)
    df_split_acceleration['timestamp'] = pd.to_datetime(df_split_acceleration['timestamp'], format='%H:%M:%S.%f')
    return df_split_acceleration


def plot_accelerometer(df_acc: pd.DataFrame) -> None:
    """Plot accelerometer data."""
    plt.figure(figsize=(12, 6))
    plt.plot(df_acc['timestamp'], df_acc['val1'], label='X', color='red')
    plt.plot(df_acc['timestamp'], df_acc['val2'], label='Y', color='green')
    plt.plot(df_acc['timestamp'], df_acc['val3'], label='Z', color='blue')
    plt.xticks(df_acc['timestamp'][::max(len(df_acc)//10,1)])
    plt.title('Raw Accelerometer Data')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def load_and_process_hr(hr_file: str) -> pd.DataFrame:
    """Load heart rate CSV file and return processed dataframe."""
    df_hr = pd.read_csv(hr_file)
    df_hr.columns = ['raw']
    df_split_hr = df_hr['raw'].str.split(';', expand=True)
    df_split_hr.columns = ['timestamp', 'val1']
    df_split_hr['val1'] = df_split_hr['val1'].astype(int)
    df_split_hr['timestamp'] = pd.to_datetime(df_split_hr['timestamp'], format='%H:%M:%S.%f')
    return df_split_hr


def plot_hr(df_hr: pd.DataFrame) -> None:
    """Plot heart rate data."""
    plt.figure(figsize=(12, 6))
    plt.plot(df_hr['timestamp'], df_hr['val1'], label='RR Interval (ms)', color='red')
    plt.title('Raw Heart Rate Data')
    plt.xticks(df_hr['timestamp'][::max(len(df_hr)//10,1)])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_hr_distribution(rr_intervals_ms: np.ndarray) -> None:
    """Plot heart rate distribution as histogram with color bins."""
    bpm = 60000 / rr_intervals_ms
    plt.figure(figsize=(12, 6))
    n, bins, patches = plt.hist(bpm, bins=30, edgecolor='black')
    colors = plt.cm.plasma(np.linspace(0, 1, len(patches)))
    for patch, color in zip(patches, colors):
        patch.set_facecolor(color)

    plt.xlabel('Heart Rate (bpm)')
    plt.ylabel('Frequency')
    plt.title('Heart Rate Distribution with Colored Bins')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_training_zones(rr_intervals_ms: np.ndarray) -> None:
    """Plot histogram of heart rate training zones based on %HRmax."""
    bpm = 60000 / rr_intervals_ms
    hr_max = np.max(bpm)
    zones = {
        "Zone 1\n(50-60%)": {"range": (0.50 * hr_max, 0.60 * hr_max), "color": "grey"},
        "Zone 2\n(60-70%)": {"range": (0.60 * hr_max, 0.70 * hr_max), "color": "lightblue"},
        "Zone 3\n(70-80%)": {"range": (0.70 * hr_max, 0.80 * hr_max), "color": "green"},
        "Zone 4\n(80-90%)": {"range": (0.80 * hr_max, 0.90 * hr_max), "color": "orange"},
        "Zone 5\n(90-100+%)": {"range": (0.90 * hr_max, hr_max + 20), "color": "red"}
    }

    zone_labels = []
    zone_counts = []
    zone_colors = []

    for label, info in zones.items():
        low, high = info["range"]
        count = np.sum((bpm >= low) & (bpm < high))
        zone_labels.append(label)
        zone_counts.append(count)
        zone_colors.append(info["color"])

    plt.figure(figsize=(10, 6))
    plt.bar(zone_labels, zone_counts, color=zone_colors, edgecolor='black')
    plt.title(f"Heart Rate Zones Distribution (HRmax={hr_max:.1f} bpm)")
    plt.xlabel('Training Zones')
    plt.ylabel('Counts')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()


def main(acc_file: str, hr_file: str, plot_spectral: bool = True) -> None:
    # Load and plot accelerometer data
    print(f"Loading accelerometer data from: {acc_file}")
    df_acc = load_and_process_accelerometer(acc_file)
    plot_accelerometer(df_acc)

    # Load and plot heart rate data
    print(f"Loading heart rate data from: {hr_file}")
    df_hr = load_and_process_hr(hr_file)
    plot_hr(df_hr)

    # Extract RR intervals
    rr_intervals_ms = df_hr['val1'].to_numpy()

    # Plot HR distribution and training zones
    plot_hr_distribution(rr_intervals_ms)
    plot_training_zones(rr_intervals_ms)

    # Calculate time domain features
    print("Extracting HRV time-domain features...")
    time_domain_features = extract_hrv_time_domain_features(rr_intervals_ms)
    for k, v in time_domain_features.items():
        print(f"{k}: {v:.2f}")

    # Calculate and plot Poincaré plot
    print("Plotting Poincaré plot and extracting SD1/SD2 features...")
    poincare_features = plot_poincare(rr_intervals_ms)
    for k, v in poincare_features.items():
        print(f"{k}: {v:.2f}")

    # Calculate spectral features
    print("Extracting HRV spectral-domain features...")
    spectral_features = extract_hrv_spectral_domain_features(rr_intervals_ms, plot=plot_spectral)
    for k, v in spectral_features.items():
        print(f"{k}: {v:.4f}")

HR_folder_path = r'C:\Users\Pallavi\Downloads\HRV_anonymized_data\HRV_anonymized_data' # pick the correct path to your HR data
ACC_folder_path = r'C:\Users\Pallavi\Downloads\accelerometer_anonymized_data\accelerometer_anonymized_data' # pick the correct path to your acc data

def run_interactive():
    base_name = input("Enter the base filename (e.g., treatment_1): ").strip()

    hr_file = os.path.join(HR_folder_path, f"{base_name}.csv")
    acc_file = os.path.join(ACC_folder_path, f"{base_name}.csv")

    missing = []
    if not os.path.isfile(hr_file):
        missing.append(f"HR file not found: {hr_file}")
    if not os.path.isfile(acc_file):
        missing.append(f"ACC file not found: {acc_file}")

    if missing:
        for m in missing:
            print(m)
        return

    main(acc_file, hr_file, plot_spectral=True)

# Then run
run_interactive()