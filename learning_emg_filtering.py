import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from scipy.signal import butter, sosfiltfilt

# =============================================================================
# CONFIGURABLE PARAMETERS
# =============================================================================
ZC_THRESHOLD_PERCENT = 0.6   # Zero Crossing threshold as fraction of RMS
SSC_THRESHOLD_PERCENT = 0.3  # Slope Sign Change threshold as fraction of RMS

file = h5py.File("assets/discrete_gestures_user_000_dataset_000.hdf5", "r")

def print_tree(name, obj):
    print(name)

file.visititems(print_tree)

print(list(file.keys()))

data = file["data"]
print(type(data))
print(data)
print(data.dtype)

raw = data[:]
print(raw.shape)
print(raw.dtype)

emg = raw['emg']     
time = raw['time']
print(emg.shape)

dt = np.diff(time)
fs = 1.0 / np.median(dt)
print("fs =", fs)
print("dt min/median/max =", dt.min(), np.median(dt), dt.max())

# 1) pick one channel
emg_ch = emg[:, 0].astype(np.float32)

# 2) drop the initial transient (recommended)
drop_s = 0.5
drop = int(drop_s * fs)
emg_ch = emg_ch[drop:]
time_ch = time[drop:]

# 3) design + apply bandpass
low, high = 20.0, 450.0
sos = butter(4, [low/(0.5*fs), high/(0.5*fs)], btype="bandpass", output="sos")
emg_bp = sosfiltfilt(sos, emg_ch)

# 4) RMS envelope extraction
# Step A: Define window size (150ms)
window_ms = 150
window_samples = int(window_ms / 1000 * fs)  # Convert ms to samples
print(f"Window: {window_ms}ms = {window_samples} samples")

# Step B: Square the bandpassed signal
emg_squared = emg_bp ** 2

# Step C: Moving average via convolution (rectangular kernel, normalized)
kernel = np.ones(window_samples) / window_samples
emg_mean_squared = np.convolve(emg_squared, kernel, mode='same')

# Step D: Square root to complete RMS
emg_envelope = np.sqrt(emg_mean_squared)

# =============================================================================
# TIME-DOMAIN FEATURE FUNCTIONS (RMS, WL, ZC, SSC)
# Computed on band-passed EMG, not RMS envelope.
# Designed for easy porting to embedded C.
# =============================================================================

def compute_rms(x):
    """Root Mean Square: sqrt(mean(x^2))"""
    return np.sqrt(np.mean(x ** 2))


def compute_wl(x):
    """Waveform Length: sum of absolute differences between consecutive samples."""
    return np.sum(np.abs(np.diff(x)))


def compute_zc(x, threshold):
    """Zero Crossings: count of sign changes where amplitude change exceeds threshold."""
    sign_change = x[:-1] * x[1:] < 0
    amp_diff = np.abs(np.diff(x)) > threshold
    return np.sum(sign_change & amp_diff)


def compute_ssc(x, threshold):
    """Slope Sign Changes: count of slope direction reversals exceeding threshold."""
    diff_left = x[1:-1] - x[:-2]
    diff_right = x[1:-1] - x[2:]
    return np.sum(diff_left * diff_right > threshold)


def compute_all_features_windowed(x, window_len, threshold_zc, threshold_ssc):
    """
    Compute RMS, WL, ZC, SSC using non-overlapping windows.
    Returns arrays of feature values, one per window (for ML).
    """
    n_samples = len(x)
    n_windows = n_samples // window_len
    x_trim = x[:n_windows * window_len]
    x_win = x_trim.reshape(n_windows, window_len)

    rms = np.sqrt(np.mean(x_win ** 2, axis=1))
    wl = np.sum(np.abs(np.diff(x_win, axis=1)), axis=1)

    sign_change = x_win[:, :-1] * x_win[:, 1:] < 0
    amp_diff = np.abs(np.diff(x_win, axis=1)) > threshold_zc
    zc = np.sum(sign_change & amp_diff, axis=1)

    diff_left = x_win[:, 1:-1] - x_win[:, :-2]
    diff_right = x_win[:, 1:-1] - x_win[:, 2:]
    ssc = np.sum(diff_left * diff_right > threshold_ssc, axis=1)

    return rms, wl, zc, ssc


# =============================================================================
# COMPUTE FEATURES FOR ALL 16 CHANNELS
# =============================================================================

n_channels = 16
all_features = {}  # Non-overlapping windows - for ML

for ch in range(n_channels):
    emg_ch_i = emg[drop:, ch].astype(np.float32)
    emg_bp_i = sosfiltfilt(sos, emg_ch_i)

    signal_rms_i = np.sqrt(np.mean(emg_bp_i ** 2))
    threshold_zc_i = ZC_THRESHOLD_PERCENT * signal_rms_i
    threshold_ssc_i = (SSC_THRESHOLD_PERCENT * signal_rms_i) ** 2

    # Non-overlapping windowed features (for ML)
    rms_i, wl_i, zc_i, ssc_i = compute_all_features_windowed(
        emg_bp_i, window_samples, threshold_zc_i, threshold_ssc_i
    )
    all_features[ch] = {'rms': rms_i, 'wl': wl_i, 'zc': zc_i, 'ssc': ssc_i}

# Time vector for windowed features
n_windows = len(all_features[0]['rms'])
window_centers = np.arange(n_windows) * window_samples + window_samples // 2
time_windows = time_ch[window_centers]

print(f"\nComputed features for {n_channels} channels")
print(f"Windows per channel (non-overlapping): {n_windows}")
print(f"Window size: {window_samples} samples ({window_ms} ms)")

# 5) Load gesture labels (prompts)
prompts = pd.read_hdf("assets/discrete_gestures_user_000_dataset_000.hdf5", key="prompts")
print("\nUnique gestures:", prompts['name'].unique())

t_abs = time_ch  # absolute timestamps

# Find first occurrence of each gesture type
index_gestures = prompts[prompts['name'].str.contains('index')]
middle_gestures = prompts[prompts['name'].str.contains('middle')]
thumb_gestures = prompts[prompts['name'].str.contains('thumb')]

# Define plot configurations: 1) Index+Middle combined, 2) Thumb separate
plot_configs = [
    {'name': 'Index & Middle Finger', 'start_time': index_gestures['time'].iloc[0], 'filter': 'index|middle'},
    {'name': 'Thumb', 'start_time': thumb_gestures['time'].iloc[0], 'filter': 'thumb'},
]

# Color function for markers
def get_gesture_color(name):
    if 'index' in name:
        return 'green'
    elif 'middle' in name:
        return 'blue'
    elif 'thumb' in name:
        return 'orange'
    return 'gray'

# =============================================================================
# 6) PLOT ALL FEATURES (RMS, WL, ZC, SSC) FOR ALL 16 CHANNELS
# =============================================================================

feature_names = ['rms', 'wl', 'zc', 'ssc']
feature_titles = ['RMS Envelope', 'Waveform Length (WL)', 'Zero Crossings (ZC)', 'Slope Sign Changes (SSC)']
feature_colors = ['red', 'blue', 'green', 'purple']
feature_ylabels = ['Amplitude', 'WL (a.u.)', 'Count', 'Count']

for config in plot_configs:
    t_start = config['start_time'] - 0.5
    t_end = t_start + 10.0

    # Mask for windowed time vector
    mask_win = (time_windows >= t_start) & (time_windows <= t_end)
    t_win_rel = time_windows[mask_win] - t_start

    # Get gestures in window
    gesture_mask = (prompts['time'] >= t_start) & (prompts['time'] <= t_end) & \
                   (prompts['name'].str.contains(config['filter']))
    gestures_in_window = prompts[gesture_mask]

    # Create one figure per feature
    for feat_idx, feat_name in enumerate(feature_names):
        fig, axes = plt.subplots(4, 4, figsize=(10, 8), sharex=True, sharey=True)
        axes = axes.flatten()

        for ch in range(n_channels):
            ax = axes[ch]

            # Plot windowed feature
            feat_data = all_features[ch][feat_name][mask_win]
            ax.plot(t_win_rel, feat_data, linewidth=1, color=feature_colors[feat_idx])
            ax.set_title(f"Ch {ch}", fontsize=9)

            # Add gesture markers
            for _, row in gestures_in_window.iterrows():
                t_g = row['time'] - t_start
                color = get_gesture_color(row['name'])
                ax.axvline(t_g, color=color, linestyle='--', alpha=0.5, linewidth=0.5)

        # Set subtitle based on gesture type
        if 'Index' in config['name']:
            subtitle = "(Green=index, Blue=middle)"
        else:
            subtitle = "(Orange=thumb)"

        fig.suptitle(f"{feature_titles[feat_idx]} - {config['name']} Gestures\n{subtitle}", fontsize=12)
        fig.supxlabel("Time (s)")
        fig.supylabel(feature_ylabels[feat_idx])
        plt.tight_layout()
        plt.show()

# =============================================================================
# SUMMARY: Feature statistics across all channels
# =============================================================================
print("\n" + "=" * 60)
print("FEATURE SUMMARY (all channels, all windows)")
print("=" * 60)
for feat_name in ['rms', 'wl', 'zc', 'ssc']:
    all_vals = np.concatenate([all_features[ch][feat_name] for ch in range(n_channels)])
    print(f"{feat_name.upper():4s} | min: {all_vals.min():10.4f} | max: {all_vals.max():10.4f} | "
          f"mean: {all_vals.mean():10.4f} | std: {all_vals.std():10.4f}")