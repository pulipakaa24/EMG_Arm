"""
EMG Data Collection Pipeline
============================
A complete pipeline for collecting, labeling, and classifying EMG signals.

OPTIONS:
  1. Collect Data    - Run a labeled collection session with timed prompts
  2. Inspect Data    - Load saved sessions, view raw EMG and features
  3. Train Classifier - Train LDA on collected data with cross-validation
  q. Quit

FEATURES:
  - Simulated EMG stream (swap for serial.Serial with real hardware)
  - Timed prompt system for consistent data collection
  - Automatic labeling based on prompt timing
  - HDF5 storage with metadata
  - Time-domain feature extraction (RMS, WL, ZC, SSC)
  - LDA classifier with evaluation metrics
"""

import time
import threading
import queue
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from datetime import datetime
import json
import h5py
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib  # For model persistence
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================
NUM_CHANNELS = 4          # Number of EMG channels (MyoWare sensors)
SAMPLING_RATE_HZ = 1000   # Must match ESP32's EMG_SAMPLE_RATE_HZ
SERIAL_BAUD = 921600      # High baud rate to prevent serial buffer backlog

# Windowing configuration
WINDOW_SIZE_MS = 150      # Window size in milliseconds
WINDOW_OVERLAP = 0.0      # Overlap ratio (0.0 = no overlap, 0.5 = 50% overlap)

# Labeling configuration
GESTURE_HOLD_SEC = 3.0    # How long to hold each gesture
REST_BETWEEN_SEC = 2.0    # Rest period between gestures
REPS_PER_GESTURE = 3      # Repetitions per gesture in a session

# Storage configuration
DATA_DIR = Path("collected_data")  # Directory to store session files
MODEL_DIR = Path("models")         # Directory to store trained models
USER_ID = "user_001"               # Current user ID (change per user)

# =============================================================================
# LABEL ALIGNMENT CONFIGURATION
# =============================================================================
# Human reaction time causes EMG activity to lag behind label prompts.
# We detect when EMG actually rises and shift labels to match.

ENABLE_LABEL_ALIGNMENT = True     # Enable/disable automatic label alignment
ONSET_THRESHOLD = 2             # Signal must exceed baseline + threshold * std
ONSET_SEARCH_MS = 2000           # Search window after prompt (ms)

# =============================================================================
# TRANSITION WINDOW FILTERING
# =============================================================================
# Windows near gesture transitions contain ambiguous data (reaction time at start,
# muscle relaxation at end). Discard these during training for cleaner labels.
# This is standard practice in EMG research (see Frontiers Neurorobotics 2023).

DISCARD_TRANSITION_WINDOWS = True  # Enable/disable transition filtering during training
TRANSITION_START_MS = 300          # Discard windows within this time AFTER gesture starts
TRANSITION_END_MS = 150            # Discard windows within this time BEFORE gesture ends

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class EMGSample:
    """Single sample from all channels at one point in time."""
    timestamp: float                    # Python-side timestamp (seconds, monotonic)
    channels: list[float]               # Raw ADC values per channel
    esp_timestamp_ms: Optional[int] = None  # Optional: timestamp from ESP32


@dataclass
class EMGWindow:
    """
    A window of samples - this is what we'll feed to ML models.

    NOTE: This class intentionally contains NO label information.
    Labels are stored separately to enforce training/inference separation.
    This ensures inference code cannot accidentally access ground truth.
    """
    window_id: int
    start_time: float
    end_time: float
    samples: list[EMGSample]

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array of shape (n_samples, n_channels)."""
        return np.array([s.channels for s in self.samples])

    def get_channel(self, ch: int) -> np.ndarray:
        """Get single channel as 1D array."""
        return np.array([s.channels[ch] for s in self.samples])


# =============================================================================
# SIMULATED EMG STREAM (Mimics what ESP32 would send over serial)
# =============================================================================

class SimulatedEMGStream:
    """
    Simulates ESP32 sending EMG data over serial.

    In reality, you'd replace this with:
        import serial
        ser = serial.Serial('COM3', 115200)
        line = ser.readline()

    The ESP32 would send lines like:
        "1234,512,489,501,523\n"  (timestamp_ms, ch0, ch1, ch2, ch3)
    """

    def __init__(self, num_channels: int = 4, sample_rate: int = 1000):
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.running = False
        self.output_queue = queue.Queue()
        self._thread = None
        self._esp_time_ms = 0

    def start(self):
        """Start the simulated data stream."""
        self.running = True
        self._thread = threading.Thread(target=self._generate_data, daemon=True)
        self._thread.start()
        print(f"[SIM] Started simulated EMG stream: {self.num_channels} channels @ {self.sample_rate}Hz")

    def stop(self):
        """Stop the simulated data stream."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        print("[SIM] Stopped simulated EMG stream")

    def readline(self) -> Optional[str]:
        """
        Mimics serial.readline() - blocks until data available.
        Returns line in format: "timestamp_ms,ch0,ch1,ch2,ch3"
        """
        try:
            return self.output_queue.get(timeout=1.0)
        except queue.Empty:
            return None

    def _generate_data(self):
        """Background thread that generates fake EMG data."""
        interval = 1.0 / self.sample_rate

        while self.running:
            # Simulate EMG signal: baseline noise + occasional bursts
            channels = []
            for ch in range(self.num_channels):
                # Base noise (typical ADC noise around 512 center for 10-bit ADC)
                base = 512 + np.random.randn() * 10

                # Occasionally add muscle activation burst (simulates gesture)
                if np.random.random() < 0.01:  # 1% chance each sample
                    base += np.random.randn() * 100

                channels.append(int(np.clip(base, 0, 1023)))

            # Format like ESP32 would send
            line = f"{self._esp_time_ms},{','.join(map(str, channels))}\n"
            self.output_queue.put(line)

            self._esp_time_ms += 1  # Increment ESP32 timestamp
            time.sleep(interval)


# =============================================================================
# DATA PARSER (Converts serial lines to EMGSample objects)
# =============================================================================

class EMGParser:
    """
    Parses incoming serial data into structured EMGSample objects.

    LESSON: Always validate incoming data. Serial lines can be:
      - Corrupted (partial lines, garbage bytes)
      - Missing (dropped packets)
      - Out of order (buffer issues)
    """

    def __init__(self, num_channels: int):
        self.num_channels = num_channels
        self.parse_errors = 0
        self.samples_parsed = 0

    def parse_line(self, line: str) -> Optional[EMGSample]:
        """
        Parse a line from ESP32 into an EMGSample.

        Expected format: "ch0,ch1,ch2,ch3\n" (channels only, no ESP32 timestamp)
        Python assigns timestamp on receipt for label alignment.
        Returns None if parsing fails.
        """
        try:
            # Strip whitespace and split
            parts = line.strip().split(',')

            # Validate we have correct number of fields (channels only)
            if len(parts) != self.num_channels:
                self.parse_errors += 1
                return None

            # Parse channel values
            channels = [float(parts[i]) for i in range(self.num_channels)]

            # Create sample with Python-side timestamp (aligned with label clock)
            sample = EMGSample(
                timestamp=time.perf_counter(),  # High-resolution monotonic clock
                channels=channels,
                esp_timestamp_ms=None  # No longer using ESP32 timestamp
            )

            self.samples_parsed += 1
            return sample

        except (ValueError, IndexError) as e:
            self.parse_errors += 1
            return None


# =============================================================================
# WINDOWING (Groups samples into fixed-size windows)
# =============================================================================

class Windower:
    """
    Groups incoming samples into fixed-size windows.

    LESSON: ML models need fixed-size inputs. We can't feed them a continuous
    stream - we need to chunk it into windows of consistent size.

    Window size tradeoffs:
      - Too small (50ms): Not enough data, noisy features
      - Too large (500ms): Slow response, gesture transitions blurred
      - Sweet spot: 150-250ms for EMG gesture recognition
    """

    def __init__(self, window_size_ms: int, sample_rate: int, overlap: float = 0.0):
        self.window_size_ms = window_size_ms
        self.sample_rate = sample_rate
        self.overlap = overlap

        # Calculate window size in samples
        self.window_size_samples = int(window_size_ms / 1000 * sample_rate)
        self.step_size_samples = int(self.window_size_samples * (1 - overlap))

        # Buffer for incoming samples
        self.buffer: list[EMGSample] = []
        self.window_count = 0

        print(f"[Windower] Window: {window_size_ms}ms = {self.window_size_samples} samples")
        print(f"[Windower] Step: {self.step_size_samples} samples (overlap={overlap*100:.0f}%)")

    def add_sample(self, sample: EMGSample) -> Optional[EMGWindow]:
        """
        Add a sample to the buffer. Returns a window if we have enough samples.

        Returns None if buffer isn't full yet.
        """
        self.buffer.append(sample)

        # Check if we have enough samples for a window
        if len(self.buffer) >= self.window_size_samples:
            # Extract window
            window_samples = self.buffer[:self.window_size_samples]
            window = EMGWindow(
                window_id=self.window_count,
                start_time=window_samples[0].timestamp,
                end_time=window_samples[-1].timestamp,
                samples=window_samples.copy()
            )
            self.window_count += 1

            # Slide buffer by step size
            self.buffer = self.buffer[self.step_size_samples:]

            return window

        return None

    def flush(self) -> Optional[EMGWindow]:
        """Flush remaining samples as a partial window (if any)."""
        if len(self.buffer) > 0:
            window = EMGWindow(
                window_id=self.window_count,
                start_time=self.buffer[0].timestamp,
                end_time=self.buffer[-1].timestamp,
                samples=self.buffer.copy()
            )
            self.buffer = []
            return window
        return None


# =============================================================================
# PROMPT SYSTEM (Timed prompts for labeling)
# =============================================================================

@dataclass
class GesturePrompt:
    """Defines a single gesture prompt in the collection sequence."""
    gesture_name: str       # e.g., "index_flex", "rest", "fist"
    duration_sec: float     # How long to hold this gesture
    start_time: float = 0.0 # Filled in by scheduler when session starts


@dataclass
class PromptSchedule:
    """A complete sequence of prompts for a collection session."""
    prompts: list[GesturePrompt]
    total_duration: float = 0.0

    def __post_init__(self):
        """Calculate start times and total duration."""
        current_time = 0.0
        for prompt in self.prompts:
            prompt.start_time = current_time
            current_time += prompt.duration_sec
        self.total_duration = current_time


class PromptScheduler:
    """
    Manages timed prompts during data collection.

    LESSON: Timed prompts give you consistent, repeatable data collection.
    The user knows exactly when to perform each gesture, and you know
    exactly when each gesture should be happening for labeling.
    """

    def __init__(self, gestures: list[str], hold_sec: float, rest_sec: float, reps: int):
        """
        Build a prompt schedule.

        Args:
            gestures: List of gesture names (e.g., ["index_flex", "fist"])
            hold_sec: How long to hold each gesture
            rest_sec: Rest period between gestures
            reps: Number of repetitions per gesture
        """
        self.gestures = gestures
        self.hold_sec = hold_sec
        self.rest_sec = rest_sec
        self.reps = reps

        # Build the schedule
        self.schedule = self._build_schedule()
        self.session_start_time: Optional[float] = None

    def _build_schedule(self) -> PromptSchedule:
        """Create the sequence of prompts."""
        prompts = []

        # Initial rest period
        prompts.append(GesturePrompt("rest", self.rest_sec))

        # For each repetition
        for rep in range(self.reps):
            # Cycle through all gestures
            for gesture in self.gestures:
                prompts.append(GesturePrompt(gesture, self.hold_sec))
                prompts.append(GesturePrompt("rest", self.rest_sec))

        return PromptSchedule(prompts)

    def start_session(self):
        """Mark the start of a collection session."""
        self.session_start_time = time.perf_counter()
        print(f"\n[Scheduler] Session started. Duration: {self.schedule.total_duration:.1f}s")
        print(f"[Scheduler] {len(self.schedule.prompts)} prompts scheduled")

    def get_current_prompt(self) -> Optional[GesturePrompt]:
        """Get the prompt that should be active right now."""
        if self.session_start_time is None:
            return None

        elapsed = time.perf_counter() - self.session_start_time

        # Find which prompt is active
        for prompt in self.schedule.prompts:
            prompt_end = prompt.start_time + prompt.duration_sec
            if prompt.start_time <= elapsed < prompt_end:
                return prompt

        return None  # Session complete

    def get_elapsed_time(self) -> float:
        """Get seconds elapsed since session start."""
        if self.session_start_time is None:
            return 0.0
        return time.perf_counter() - self.session_start_time

    def is_session_complete(self) -> bool:
        """Check if we've passed the end of the schedule."""
        return self.get_elapsed_time() >= self.schedule.total_duration

    def get_label_for_time(self, timestamp: float) -> str:
        """
        Get the gesture label for a specific timestamp.

        This is used to label windows after collection.
        """
        if self.session_start_time is None:
            return "unlabeled"

        elapsed = timestamp - self.session_start_time

        for prompt in self.schedule.prompts:
            prompt_end = prompt.start_time + prompt.duration_sec
            if prompt.start_time <= elapsed < prompt_end:
                return prompt.gesture_name

        return "unlabeled"

    def print_schedule(self):
        """Print the full prompt schedule."""
        print("\n" + "-" * 40)
        print("PROMPT SCHEDULE")
        print("-" * 40)
        for i, p in enumerate(self.schedule.prompts):
            print(f"  {i+1:2d}. [{p.start_time:5.1f}s - {p.start_time + p.duration_sec:5.1f}s] {p.gesture_name}")
        print(f"\n  Total duration: {self.schedule.total_duration:.1f}s")


# =============================================================================
# SIMULATED EMG STREAM (Gesture-aware signal generation)
# =============================================================================

class GestureAwareEMGStream(SimulatedEMGStream):
    """
    Enhanced simulation that generates different EMG patterns based on
    which gesture is currently being prompted.

    This makes the simulated data more realistic for testing your pipeline.
    Each gesture activates different "muscles" (channels) with different intensities.
    """

    # Define which channels activate for each gesture (0-1 intensity per channel)
    GESTURE_PATTERNS = {
        "rest":        [0.0, 0.0, 0.0, 0.0],
        "open":        [0.3, 0.3, 0.3, 0.3],  # Moderate all channels (extension)
        "fist":        [0.7, 0.7, 0.6, 0.6],  # All channels active (flexion)
        "hook_em":     [0.8, 0.2, 0.7, 0.1],  # Index + pinky extended (ch0 + ch2)
        "thumbs_up":   [0.1, 0.1, 0.2, 0.8],  # Thumb dominant (ch3)
    }

    def __init__(self, num_channels: int = 4, sample_rate: int = 1000):
        super().__init__(num_channels, sample_rate)
        self.current_gesture = "rest"
        self._gesture_lock = threading.Lock()

    def set_gesture(self, gesture: str):
        """Set the current gesture being performed."""
        with self._gesture_lock:
            self.current_gesture = gesture

    def _generate_data(self):
        """Generate EMG data based on current gesture."""
        interval = 1.0 / self.sample_rate

        while self.running:
            with self._gesture_lock:
                gesture = self.current_gesture

            # Get activation pattern for current gesture
            pattern = self.GESTURE_PATTERNS.get(gesture, [0.0] * self.num_channels)

            channels = []
            for ch in range(self.num_channels):
                # Base signal around 512 (10-bit ADC center)
                base = 512

                # Add noise (always present)
                noise = np.random.randn() * 10

                # Add muscle activation based on pattern
                activation = pattern[ch] * np.random.randn() * 150  # Scaled EMG burst

                # Combine and clip to ADC range
                value = int(np.clip(base + noise + activation, 0, 1023))
                channels.append(value)

            # Format like ESP32 would send
            line = f"{self._esp_time_ms},{','.join(map(str, channels))}\n"
            self.output_queue.put(line)

            self._esp_time_ms += 1
            time.sleep(interval)


# =============================================================================
# LABEL ALIGNMENT (Simple Onset Detection)
# =============================================================================
from scipy.signal import butter, sosfiltfilt


def align_labels_with_onset(
    labels: list[str],
    window_start_times: np.ndarray,
    raw_timestamps: np.ndarray,
    raw_channels: np.ndarray,
    sampling_rate: int,
    threshold_factor: float = 2.0,
    search_ms: float = 800
) -> list[str]:
    """
    Align labels to EMG onset by detecting when signal rises above baseline.

    Simple algorithm:
    1. High-pass filter to remove DC offset
    2. Compute RMS envelope across channels
    3. At each label transition, find where envelope exceeds baseline + threshold
    4. Move label boundary to that point
    """
    if len(labels) == 0:
        return labels.copy()

    # High-pass filter to remove DC (raw ADC has ~2340mV offset)
    nyquist = sampling_rate / 2
    sos = butter(2, 20.0 / nyquist, btype='high', output='sos')

    # Filter and compute envelope (RMS across channels)
    filtered = np.zeros_like(raw_channels)
    for ch in range(raw_channels.shape[1]):
        filtered[:, ch] = sosfiltfilt(sos, raw_channels[:, ch])
    envelope = np.sqrt(np.mean(filtered ** 2, axis=1))

    # Smooth envelope
    sos_lp = butter(2, 10.0 / nyquist, btype='low', output='sos')
    envelope = sosfiltfilt(sos_lp, envelope)

    # Find transitions and detect onsets
    search_samples = int(search_ms / 1000 * sampling_rate)
    baseline_samples = int(200 / 1000 * sampling_rate)

    boundaries = []  # (time, new_label)

    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            prompt_time = window_start_times[i]

            # Find index in raw signal closest to prompt time
            prompt_idx = np.searchsorted(raw_timestamps, prompt_time)

            # Get baseline (before transition)
            base_start = max(0, prompt_idx - baseline_samples)
            baseline = envelope[base_start:prompt_idx]
            if len(baseline) == 0:
                boundaries.append((prompt_time + 0.3, labels[i]))
                continue

            threshold = np.mean(baseline) + threshold_factor * np.std(baseline)

            # Search forward for onset
            search_end = min(len(envelope), prompt_idx + search_samples)
            onset_idx = None

            for j in range(prompt_idx, search_end):
                if envelope[j] > threshold:
                    onset_idx = j
                    break

            if onset_idx is not None:
                onset_time = raw_timestamps[onset_idx]
            else:
                onset_time = prompt_time + 0.3  # fallback

            boundaries.append((onset_time, labels[i]))

    # Assign labels based on detected boundaries
    aligned = []
    boundary_idx = 0
    current_label = labels[0]

    for t in window_start_times:
        while boundary_idx < len(boundaries) and t >= boundaries[boundary_idx][0]:
            current_label = boundaries[boundary_idx][1]
            boundary_idx += 1
        aligned.append(current_label)

    return aligned


def filter_transition_windows(
    X: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    start_times: np.ndarray,
    end_times: np.ndarray,
    transition_start_ms: float = TRANSITION_START_MS,
    transition_end_ms: float = TRANSITION_END_MS
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Filter out windows that fall within transition zones at gesture boundaries.

    This removes ambiguous data where:
    - User is still reacting to prompt (start of gesture)
    - User is anticipating next gesture (end of gesture)

    Args:
        X: EMG data array (n_windows, samples, channels)
        y: Label indices (n_windows,)
        labels: String labels (n_windows,)
        start_times: Window start times in seconds (n_windows,)
        end_times: Window end times in seconds (n_windows,)
        transition_start_ms: Discard windows within this time after gesture start
        transition_end_ms: Discard windows within this time before gesture end

    Returns:
        Filtered (X, y, labels) with transition windows removed
    """
    if len(X) == 0:
        return X, y, labels

    transition_start_sec = transition_start_ms / 1000.0
    transition_end_sec = transition_end_ms / 1000.0

    # Find gesture boundaries (where label changes)
    # Each boundary is the START of a new gesture segment
    boundaries = [0]  # First window starts a segment
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:
            boundaries.append(i)
    boundaries.append(len(labels))  # End marker

    # For each segment, find start_time and end_time of the gesture
    # Then mark windows that are within transition zones
    keep_mask = np.ones(len(X), dtype=bool)

    for seg_idx in range(len(boundaries) - 1):
        seg_start_idx = boundaries[seg_idx]
        seg_end_idx = boundaries[seg_idx + 1]

        # Get the time boundaries of this gesture segment
        gesture_start_time = start_times[seg_start_idx]
        gesture_end_time = end_times[seg_end_idx - 1]  # Last window's end time

        # Mark windows in transition zones
        for i in range(seg_start_idx, seg_end_idx):
            window_start = start_times[i]
            window_end = end_times[i]

            # Check if window is too close to gesture START (reaction time zone)
            if window_start < gesture_start_time + transition_start_sec:
                keep_mask[i] = False

            # Check if window is too close to gesture END (anticipation zone)
            if window_end > gesture_end_time - transition_end_sec:
                keep_mask[i] = False

    # Apply filter
    X_filtered = X[keep_mask]
    y_filtered = y[keep_mask]
    labels_filtered = [l for l, keep in zip(labels, keep_mask) if keep]

    n_removed = len(X) - len(X_filtered)
    if n_removed > 0:
        print(f"[Filter] Removed {n_removed} transition windows ({n_removed/len(X)*100:.1f}%)")
        print(f"[Filter] Kept {len(X_filtered)} windows for training")

    return X_filtered, y_filtered, labels_filtered


# =============================================================================
# SESSION STORAGE (Save/Load labeled data to HDF5)
# =============================================================================

@dataclass
class SessionMetadata:
    """Metadata for a collection session."""
    user_id: str
    session_id: str
    timestamp: str
    sampling_rate: int
    window_size_ms: int
    num_channels: int
    gestures: list[str]
    notes: str = ""


class SessionStorage:
    """Handles saving and loading EMG collection sessions to HDF5 files."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def generate_session_id(self, user_id: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{user_id}_{timestamp}"

    def get_session_filepath(self, session_id: str) -> Path:
        return self.data_dir / f"{session_id}.hdf5"

    def save_session(
        self,
        windows: list[EMGWindow],
        labels: list[str],
        metadata: SessionMetadata,
        raw_samples: Optional[list[EMGSample]] = None,
        session_start_time: Optional[float] = None,
        enable_alignment: bool = ENABLE_LABEL_ALIGNMENT
    ) -> Path:
        """
        Save a collection session to HDF5 with optional label alignment.

        When raw_samples and session_start_time are provided and enable_alignment
        is True, automatically detects EMG onset and corrects labels for human
        reaction time delay.

        Args:
            windows: List of EMGWindow objects (no label info)
            labels: List of gesture labels, parallel to windows
            metadata: Session metadata
            raw_samples: Raw samples (required for alignment)
            session_start_time: When session started (required for alignment)
            enable_alignment: Whether to perform automatic label alignment
        """
        filepath = self.get_session_filepath(metadata.session_id)

        if not windows:
            raise ValueError("No windows to save!")

        if len(windows) != len(labels):
            raise ValueError(f"Windows ({len(windows)}) and labels ({len(labels)}) must have same length!")

        window_samples = len(windows[0].samples)
        num_channels = len(windows[0].samples[0].channels)

        # Prepare timing arrays
        start_times = np.array([w.start_time for w in windows], dtype=np.float64)
        end_times = np.array([w.end_time for w in windows], dtype=np.float64)

        # Label alignment using onset detection
        aligned_labels = labels
        original_labels = labels

        if enable_alignment and raw_samples and len(raw_samples) > 0:
            print("[Storage] Aligning labels to EMG onset...")

            raw_timestamps = np.array([s.timestamp for s in raw_samples], dtype=np.float64)
            raw_channels = np.array([s.channels for s in raw_samples], dtype=np.float32)

            aligned_labels = align_labels_with_onset(
                labels=labels,
                window_start_times=start_times,
                raw_timestamps=raw_timestamps,
                raw_channels=raw_channels,
                sampling_rate=metadata.sampling_rate,
                threshold_factor=ONSET_THRESHOLD,
                search_ms=ONSET_SEARCH_MS
            )

            changed = sum(1 for a, b in zip(labels, aligned_labels) if a != b)
            print(f"[Storage] Labels aligned: {changed}/{len(labels)} windows shifted")
        elif enable_alignment:
            print("[Storage] Warning: No raw samples, skipping alignment")

        with h5py.File(filepath, 'w') as f:
            # Metadata as attributes
            f.attrs['user_id'] = metadata.user_id
            f.attrs['session_id'] = metadata.session_id
            f.attrs['timestamp'] = metadata.timestamp
            f.attrs['sampling_rate'] = metadata.sampling_rate
            f.attrs['window_size_ms'] = metadata.window_size_ms
            f.attrs['num_channels'] = metadata.num_channels
            f.attrs['gestures'] = json.dumps(metadata.gestures)
            f.attrs['notes'] = metadata.notes
            f.attrs['num_windows'] = len(windows)
            f.attrs['window_samples'] = window_samples

            # Windows group
            windows_grp = f.create_group('windows')

            emg_data = np.array([w.to_numpy() for w in windows], dtype=np.float32)
            windows_grp.create_dataset('emg_data', data=emg_data, compression='gzip', compression_opts=4)

            # Store ALIGNED labels as primary (what training will use)
            max_label_len = max(len(l) for l in aligned_labels)
            dt = h5py.string_dtype(encoding='utf-8', length=max_label_len + 1)
            windows_grp.create_dataset('labels', data=aligned_labels, dtype=dt)

            # Also store original labels for reference/debugging
            windows_grp.create_dataset('labels_original', data=original_labels, dtype=dt)

            window_ids = np.array([w.window_id for w in windows], dtype=np.int32)
            windows_grp.create_dataset('window_ids', data=window_ids)

            windows_grp.create_dataset('start_times', data=start_times)
            windows_grp.create_dataset('end_times', data=end_times)

            # Store alignment metadata
            f.attrs['alignment_enabled'] = enable_alignment
            f.attrs['alignment_method'] = 'onset_detection' if (enable_alignment and raw_samples) else 'none'

            if raw_samples:
                raw_grp = f.create_group('raw_samples')
                timestamps = np.array([s.timestamp for s in raw_samples], dtype=np.float64)
                channels = np.array([s.channels for s in raw_samples], dtype=np.float32)
                raw_grp.create_dataset('timestamps', data=timestamps, compression='gzip')
                raw_grp.create_dataset('channels', data=channels, compression='gzip')

        print(f"[Storage] Saved session to: {filepath}")
        print(f"[Storage] File size: {filepath.stat().st_size / 1024:.1f} KB")
        return filepath

    def load_session(self, session_id: str) -> tuple[list[EMGWindow], list[str], SessionMetadata]:
        """
        Load a collection session from HDF5.

        Returns:
            windows: List of EMGWindow objects (no label info)
            labels: List of gesture labels, parallel to windows
            metadata: Session metadata
        """
        filepath = self.get_session_filepath(session_id)
        if not filepath.exists():
            raise FileNotFoundError(f"Session not found: {filepath}")

        windows = []
        labels_out = []
        with h5py.File(filepath, 'r') as f:
            metadata = SessionMetadata(
                user_id=f.attrs['user_id'],
                session_id=f.attrs['session_id'],
                timestamp=f.attrs['timestamp'],
                sampling_rate=int(f.attrs['sampling_rate']),
                window_size_ms=int(f.attrs['window_size_ms']),
                num_channels=int(f.attrs['num_channels']),
                gestures=json.loads(f.attrs['gestures']),
                notes=f.attrs.get('notes', '')
            )

            windows_grp = f['windows']
            emg_data = windows_grp['emg_data'][:]
            labels_raw = windows_grp['labels'][:]
            window_ids = windows_grp['window_ids'][:]
            start_times = windows_grp['start_times'][:]
            end_times = windows_grp['end_times'][:]

            for i in range(len(emg_data)):
                samples = []
                window_data = emg_data[i]
                for j in range(len(window_data)):
                    sample = EMGSample(
                        timestamp=start_times[i] + j * (1.0 / metadata.sampling_rate),
                        channels=window_data[j].tolist()
                    )
                    samples.append(sample)

                # Decode label
                label = labels_raw[i]
                if isinstance(label, bytes):
                    label = label.decode('utf-8')
                labels_out.append(label)

                # Window contains NO label - labels stored separately
                window = EMGWindow(
                    window_id=int(window_ids[i]),
                    start_time=float(start_times[i]),
                    end_time=float(end_times[i]),
                    samples=samples
                )
                windows.append(window)

        print(f"[Storage] Loaded session: {session_id}")
        print(f"[Storage] {len(windows)} windows, {len(metadata.gestures)} gesture types")
        return windows, labels_out, metadata

    def load_for_training(self, session_id: str, filter_transitions: bool = DISCARD_TRANSITION_WINDOWS) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Load a single session in ML-ready format: X, y, label_names.

        Args:
            session_id: The session to load
            filter_transitions: If True, remove windows in transition zones (default from config)
        """
        filepath = self.get_session_filepath(session_id)

        with h5py.File(filepath, 'r') as f:
            X = f['windows/emg_data'][:]
            labels_raw = f['windows/labels'][:]
            start_times = f['windows/start_times'][:]
            end_times = f['windows/end_times'][:]

        labels = []
        for l in labels_raw:
            if isinstance(l, bytes):
                labels.append(l.decode('utf-8'))
            else:
                labels.append(l)

        print(f"[Storage] Loaded session: {session_id} ({X.shape[0]} windows)")

        # Apply transition filtering if enabled
        if filter_transitions:
            label_names_pre = sorted(set(labels))
            label_to_idx_pre = {name: idx for idx, name in enumerate(label_names_pre)}
            y_pre = np.array([label_to_idx_pre[l] for l in labels], dtype=np.int32)

            X, y_pre, labels = filter_transition_windows(
                X, y_pre, labels, start_times, end_times
            )

        label_names = sorted(set(labels))
        label_to_idx = {name: idx for idx, name in enumerate(label_names)}
        y = np.array([label_to_idx[l] for l in labels], dtype=np.int32)

        print(f"[Storage] Ready for training: X{X.shape}, y{y.shape}")
        print(f"[Storage] Labels: {label_names}")
        return X, y, label_names

    def load_all_for_training(self, filter_transitions: bool = DISCARD_TRANSITION_WINDOWS) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
        """
        Load ALL sessions combined into a single training dataset.

        Args:
            filter_transitions: If True, remove windows in transition zones (default from config)

        Returns:
            X: Combined EMG windows from all sessions (n_total_windows, samples, channels)
            y: Combined labels as integers (n_total_windows,)
            label_names: Sorted list of unique gesture labels across all sessions
            session_ids: List of session IDs that were loaded

        Raises:
            ValueError: If no sessions found or sessions have incompatible shapes
        """
        sessions = self.list_sessions()

        if not sessions:
            raise ValueError("No sessions found to load!")

        print(f"[Storage] Loading {len(sessions)} session(s) for combined training...")
        if filter_transitions:
            print(f"[Storage] Transition filtering: START={TRANSITION_START_MS}ms, END={TRANSITION_END_MS}ms")

        all_X = []
        all_labels = []
        loaded_sessions = []
        reference_shape = None
        total_removed = 0
        total_original = 0

        for session_id in sessions:
            filepath = self.get_session_filepath(session_id)

            with h5py.File(filepath, 'r') as f:
                X = f['windows/emg_data'][:]
                labels_raw = f['windows/labels'][:]
                start_times = f['windows/start_times'][:]
                end_times = f['windows/end_times'][:]

            # Validate shape compatibility
            if reference_shape is None:
                reference_shape = X.shape[1:]  # (samples_per_window, channels)
            elif X.shape[1:] != reference_shape:
                print(f"[Storage] WARNING: Skipping {session_id} - incompatible shape {X.shape[1:]} vs {reference_shape}")
                continue

            # Decode labels
            labels = []
            for l in labels_raw:
                if isinstance(l, bytes):
                    labels.append(l.decode('utf-8'))
                else:
                    labels.append(l)

            original_count = len(X)
            total_original += original_count

            # Apply transition filtering per session (each has its own gesture boundaries)
            if filter_transitions:
                # Need temporary y for filtering function
                temp_label_names = sorted(set(labels))
                temp_label_to_idx = {name: idx for idx, name in enumerate(temp_label_names)}
                temp_y = np.array([temp_label_to_idx[l] for l in labels], dtype=np.int32)

                X, temp_y, labels = filter_transition_windows(
                    X, temp_y, labels, start_times, end_times
                )
                total_removed += original_count - len(X)

            all_X.append(X)
            all_labels.extend(labels)
            loaded_sessions.append(session_id)
            print(f"[Storage]   - {session_id}: {len(X)} windows" +
                  (f" (was {original_count})" if filter_transitions and len(X) != original_count else ""))

        if not all_X:
            raise ValueError("No compatible sessions found!")

        # Combine all data
        X_combined = np.concatenate(all_X, axis=0)

        # Create unified label mapping across all sessions
        label_names = sorted(set(all_labels))
        label_to_idx = {name: idx for idx, name in enumerate(label_names)}
        y_combined = np.array([label_to_idx[l] for l in all_labels], dtype=np.int32)

        print(f"[Storage] Combined dataset: X{X_combined.shape}, y{y_combined.shape}")
        if filter_transitions and total_removed > 0:
            print(f"[Storage] Total removed: {total_removed}/{total_original} windows ({total_removed/total_original*100:.1f}%)")
        print(f"[Storage] Labels: {label_names}")
        print(f"[Storage] Sessions loaded: {len(loaded_sessions)}")

        return X_combined, y_combined, label_names, loaded_sessions

    def list_sessions(self) -> list[str]:
        """List all available session IDs."""
        return sorted([f.stem for f in self.data_dir.glob("*.hdf5")])

    def get_session_info(self, session_id: str) -> dict:
        """Get quick info about a session without loading all data."""
        filepath = self.get_session_filepath(session_id)
        with h5py.File(filepath, 'r') as f:
            return {
                'user_id': f.attrs['user_id'],
                'timestamp': f.attrs['timestamp'],
                'num_windows': f.attrs['num_windows'],
                'gestures': json.loads(f.attrs['gestures']),
                'sampling_rate': f.attrs['sampling_rate'],
                'window_size_ms': f.attrs['window_size_ms'],
            }


# =============================================================================
# COLLECTION LOOP (Core collection pattern)
# =============================================================================

def run_collection_demo(duration_seconds: float = 5.0):
    """
    Demonstrates the core data collection loop.

    This is the pattern you'll use with real hardware - only the
    data source changes (SimulatedEMGStream -> serial.Serial).
    """
    print("\n" + "=" * 60)
    print("EMG DATA COLLECTION DEMO")
    print("=" * 60)

    # Initialize components
    stream = SimulatedEMGStream(num_channels=NUM_CHANNELS, sample_rate=SAMPLING_RATE_HZ)
    parser = EMGParser(num_channels=NUM_CHANNELS)

    # Storage for collected samples
    collected_samples: list[EMGSample] = []

    # Start the stream
    stream.start()
    start_time = time.perf_counter()

    print(f"\nCollecting for {duration_seconds} seconds...")
    print("(In real use, this would read from serial port)\n")

    try:
        while (time.perf_counter() - start_time) < duration_seconds:
            # Read line from stream (blocks briefly if no data)
            line = stream.readline()

            if line:
                # Parse into structured sample
                sample = parser.parse_line(line)

                if sample:
                    collected_samples.append(sample)

                    # Print progress every 500 samples
                    if len(collected_samples) % 500 == 0:
                        elapsed = time.perf_counter() - start_time
                        rate = len(collected_samples) / elapsed
                        print(f"  Collected {len(collected_samples)} samples "
                              f"({rate:.1f} samples/sec)")

    except KeyboardInterrupt:
        print("\n[Interrupted by user]")

    finally:
        stream.stop()

    # Report results
    print("\n" + "-" * 40)
    print("COLLECTION RESULTS")
    print("-" * 40)
    print(f"Total samples collected: {len(collected_samples)}")
    print(f"Parse errors: {parser.parse_errors}")
    print(f"Actual duration: {time.perf_counter() - start_time:.2f}s")

    if collected_samples:
        actual_rate = len(collected_samples) / (time.perf_counter() - start_time)
        print(f"Effective sample rate: {actual_rate:.1f} Hz")

        # Show sample data structure
        print("\nExample sample:")
        s = collected_samples[0]
        print(f"  timestamp: {s.timestamp:.6f}")
        print(f"  channels: {s.channels}")
        print(f"  esp_timestamp_ms: {s.esp_timestamp_ms}")

        # Quick statistics
        data = np.array([s.channels for s in collected_samples])
        print(f"\nChannel statistics (mean/std):")
        for ch in range(NUM_CHANNELS):
            print(f"  Ch{ch}: {data[:, ch].mean():.1f} / {data[:, ch].std():.1f}")

    return collected_samples


# =============================================================================
# COLLECTION SESSION
# =============================================================================

def run_labeled_collection_demo():
    """
    Run a labeled EMG collection session:
      1. Prompt scheduler guides the user through gestures
      2. EMG stream generates/collects signals
      3. Windower groups samples into fixed-size windows
      4. Labels are assigned based on which prompt was active
      5. Session is saved to HDF5 with user ID
    """
    print("\n" + "=" * 60)
    print("LABELED EMG COLLECTION")
    print("=" * 60)

    # Get user ID
    user_id = input("\nEnter your user ID (e.g., user_001): ").strip()
    if not user_id:
        user_id = USER_ID  # Fall back to default
        print(f"  Using default: {user_id}")
    else:
        print(f"  User ID: {user_id}")

    # Define gestures to collect (names match ESP32 gesture definitions)
    gestures = ["open", "fist", "hook_em", "thumbs_up"]

    # Create the prompt scheduler
    scheduler = PromptScheduler(
        gestures=gestures,
        hold_sec=GESTURE_HOLD_SEC,
        rest_sec=REST_BETWEEN_SEC,
        reps=REPS_PER_GESTURE
    )
    scheduler.print_schedule()

    # Create components
    stream = GestureAwareEMGStream(num_channels=NUM_CHANNELS, sample_rate=SAMPLING_RATE_HZ)
    parser = EMGParser(num_channels=NUM_CHANNELS)
    windower = Windower(
        window_size_ms=WINDOW_SIZE_MS,
        sample_rate=SAMPLING_RATE_HZ,
        overlap=WINDOW_OVERLAP
    )

    # Storage for windows and labels (kept separate to enforce training/inference separation)
    collected_windows: list[EMGWindow] = []
    collected_labels: list[str] = []
    last_prompt_name = None

    # Start collection
    input("\nPress ENTER to start collection session...")
    stream.start()
    scheduler.start_session()

    print("\n" + "-" * 40)
    print("COLLECTING... Watch the prompts!")
    print("-" * 40)

    try:
        while not scheduler.is_session_complete():
            # Get current prompt
            prompt = scheduler.get_current_prompt()

            # Display prompt changes
            if prompt and prompt.gesture_name != last_prompt_name:
                elapsed = scheduler.get_elapsed_time()
                if prompt.gesture_name == "rest":
                    print(f"\n  [{elapsed:5.1f}s] >>> REST <<<")
                else:
                    print(f"\n  [{elapsed:5.1f}s] >>> {prompt.gesture_name.upper()} <<<")
                last_prompt_name = prompt.gesture_name

                # Update simulated stream to generate appropriate signal
                stream.set_gesture(prompt.gesture_name)

            # Read and parse data
            line = stream.readline()
            if line:
                sample = parser.parse_line(line)
                if sample:
                    # Try to form a window
                    window = windower.add_sample(sample)
                    if window:
                        # Store window and label separately (training/inference separation)
                        label = scheduler.get_label_for_time(window.start_time)
                        collected_windows.append(window)
                        collected_labels.append(label)

    except KeyboardInterrupt:
        print("\n[Interrupted by user]")

    finally:
        stream.stop()

    # Report results
    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)
    print(f"Total windows collected: {len(collected_windows)}")
    print(f"Parse errors: {parser.parse_errors}")

    # Count labels (from separate labels list, not from windows)
    label_counts = {}
    for label in collected_labels:
        label_counts[label] = label_counts.get(label, 0) + 1

    print(f"\nWindows per label:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

    # Show example windows
    if collected_windows:
        print(f"\nExample windows:")
        for i, w in enumerate(collected_windows[:3]):
            data = w.to_numpy()
            print(f"  Window {w.window_id}: label='{collected_labels[i]}', "
                  f"samples={len(w.samples)}, "
                  f"ch0_mean={data[:, 0].mean():.1f}")

        # Show one window from each gesture type
        print(f"\nSignal comparison (channel 0 std dev by gesture):")
        for label in sorted(label_counts.keys()):
            # Get indices where label matches
            indices = [i for i, l in enumerate(collected_labels) if l == label]
            if indices:
                all_ch0 = np.concatenate([collected_windows[i].get_channel(0) for i in indices])
                print(f"  {label}: std={all_ch0.std():.1f}")

    # --- Save the session ---
    if collected_windows:
        save_choice = input("\nSave this session? (y/n): ").strip().lower()
        if save_choice == 'y':
            storage = SessionStorage()
            session_id = storage.generate_session_id(user_id)

            metadata = SessionMetadata(
                user_id=user_id,
                session_id=session_id,
                timestamp=datetime.now().isoformat(),
                sampling_rate=SAMPLING_RATE_HZ,
                window_size_ms=WINDOW_SIZE_MS,
                num_channels=NUM_CHANNELS,
                gestures=gestures,
                notes=""
            )

            # Pass windows and labels separately (enforces separation)
            filepath = storage.save_session(collected_windows, collected_labels, metadata)
            print(f"\nSession saved! ID: {session_id}")

    return collected_windows, collected_labels


# =============================================================================
# INSPECT SESSIONS (Load and view saved sessions)
# =============================================================================

def run_storage_demo():
    """Demonstrates loading and inspecting saved sessions."""
    print("\n" + "=" * 60)
    print("INSPECT SAVED SESSIONS")
    print("=" * 60)

    storage = SessionStorage()
    sessions = storage.list_sessions()

    if not sessions:
        print("\nNo saved sessions found!")
        print(f"Run option 2 first to collect and save a session.")
        print(f"Sessions are stored in: {storage.data_dir.absolute()}")
        return None

    print(f"\nFound {len(sessions)} saved session(s):")
    print("-" * 40)

    for i, session_id in enumerate(sessions):
        info = storage.get_session_info(session_id)
        print(f"\n  [{i+1}] {session_id}")
        print(f"      User: {info['user_id']}")
        print(f"      Time: {info['timestamp']}")
        print(f"      Windows: {info['num_windows']}")
        print(f"      Gestures: {info['gestures']}")

    print("\n" + "-" * 40)
    choice = input("Enter session number to load (or 'q' to quit): ").strip()

    if choice.lower() == 'q':
        return None

    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(sessions):
            print("Invalid selection!")
            return None
        session_id = sessions[idx]
    except ValueError:
        print("Invalid input!")
        return None

    print(f"\n{'=' * 60}")
    print(f"LOADING SESSION: {session_id}")
    print("=" * 60)

    # Labels returned separately from windows (enforces training/inference separation)
    windows, labels, metadata = storage.load_session(session_id)

    print(f"\nMetadata:")
    print(f"  User: {metadata.user_id}")
    print(f"  Timestamp: {metadata.timestamp}")
    print(f"  Sampling rate: {metadata.sampling_rate} Hz")
    print(f"  Window size: {metadata.window_size_ms} ms")
    print(f"  Channels: {metadata.num_channels}")
    print(f"  Gestures: {metadata.gestures}")

    # Count from separate labels list
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1

    print(f"\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} windows")

    print(f"\n{'-' * 40}")
    print("LOADING FOR MACHINE LEARNING")
    print("-" * 40)

    X, y, label_names = storage.load_for_training(session_id)

    print(f"\nData shapes:")
    print(f"  X (features): {X.shape}")
    print(f"     - {X.shape[0]} windows")
    print(f"     - {X.shape[1]} samples per window")
    print(f"     - {X.shape[2]} channels")
    print(f"  y (labels): {y.shape}")
    print(f"  Label mapping: {dict(enumerate(label_names))}")

    # --- Feature Extraction & Visualization ---
    print(f"\n{'-' * 40}")
    print("EXTRACTING FEATURES FOR VISUALIZATION")
    print("-" * 40)

    # Note: Per-window centering is done inside EMGFeatureExtractor.
    # This is the correct approach for real-time inference (causal, no future data).
    # Global centering across all windows would leak information and not work in real-time.

    extractor = EMGFeatureExtractor()
    n_windows = X.shape[0]
    n_channels = X.shape[2]

    # Extract features per channel: shape (n_windows, n_channels, 4)
    # Features order: [rms, wl, zc, ssc]
    features_by_channel = np.zeros((n_windows, n_channels, 4))

    for i in range(n_windows):
        for ch in range(n_channels):
            ch_features = extractor.extract_features_single_channel(X[i, :, ch])
            features_by_channel[i, ch, 0] = ch_features['rms']
            features_by_channel[i, ch, 1] = ch_features['wl']
            features_by_channel[i, ch, 2] = ch_features['zc']
            features_by_channel[i, ch, 3] = ch_features['ssc']

    print(f"  Extracted features for {n_windows} windows, {n_channels} channels")
    print(f"  (Per-window centering applied inside feature extractor)")

    # Create time axis (window indices as proxy for time)
    time_axis = np.arange(n_windows)

    # Find gesture transition points (where label changes)
    transitions = []
    current_label = y[0]
    for i in range(1, len(y)):
        if y[i] != current_label:
            transitions.append((i, label_names[y[i]]))
            current_label = y[i]

    # Define colors for gesture markers
    def get_gesture_color(name):
        if 'index' in name.lower():
            return 'green'
        elif 'fist' in name.lower():
            return 'blue'
        elif 'rest' in name.lower():
            return 'gray'
        elif 'thumb' in name.lower():
            return 'orange'
        elif 'middle' in name.lower():
            return 'purple'
        return 'red'

    feature_titles = ['RMS', 'Waveform Length (WL)', 'Zero Crossings (ZC)', 'Slope Sign Changes (SSC)']
    feature_colors = ['red', 'blue', 'green', 'purple']
    feature_ylabels = ['Amplitude', 'WL (a.u.)', 'Count', 'Count']

    # --- Figure 1: Raw EMG Signal ---
    fig_raw, axes_raw = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axes_raw = axes_raw.flatten()

    # Concatenate all windows to show continuous signal
    samples_per_window = X.shape[1]
    total_samples = n_windows * samples_per_window
    raw_time = np.arange(total_samples)

    for ch in range(n_channels):
        ax = axes_raw[ch]

        # Flatten all windows into continuous signal for this channel
        # Center per-channel for visualization only (subtract channel mean)
        raw_signal = X[:, :, ch].flatten()
        raw_signal_centered = raw_signal - raw_signal.mean()
        ax.plot(raw_time, raw_signal_centered, linewidth=0.5, color='black')
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.set_title(f"Channel {ch}", fontsize=11)
        ax.set_ylabel("Amplitude (centered)")
        ax.grid(True, alpha=0.3)

        # Add vertical lines at gesture transitions (scaled to sample index)
        for trans_idx, trans_label in transitions:
            sample_idx = trans_idx * samples_per_window
            color = get_gesture_color(trans_label)
            ax.axvline(sample_idx, color=color, linestyle='--', alpha=0.6, linewidth=1)

    # Add legend for gesture colors
    legend_elements = []
    for label_name in label_names:
        color = get_gesture_color(label_name)
        legend_elements.append(plt.Line2D([0], [0], color=color, linestyle='--', label=label_name))
    axes_raw[0].legend(handles=legend_elements, loc='upper right', fontsize=8)

    fig_raw.suptitle("Raw EMG Signal (Centered for Display) - All Channels", fontsize=14, fontweight='bold')
    fig_raw.supxlabel("Sample Index")
    plt.tight_layout()

    # --- Figures 2-5: Feature plots (one per feature type) ---
    for feat_idx, feat_title in enumerate(feature_titles):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        axes = axes.flatten()

        for ch in range(n_channels):
            ax = axes[ch]

            # Plot feature as line graph
            feat_data = features_by_channel[:, ch, feat_idx]
            ax.plot(time_axis, feat_data, linewidth=1, color=feature_colors[feat_idx])
            ax.set_title(f"Channel {ch}", fontsize=11)
            ax.grid(True, alpha=0.3)

            # Add vertical lines at gesture transitions
            for trans_idx, trans_label in transitions:
                color = get_gesture_color(trans_label)
                ax.axvline(trans_idx, color=color, linestyle='--', alpha=0.6, linewidth=1)

        # Add legend for gesture colors
        legend_elements = []
        for label_name in label_names:
            color = get_gesture_color(label_name)
            legend_elements.append(plt.Line2D([0], [0], color=color, linestyle='--', label=label_name))
        axes[0].legend(handles=legend_elements, loc='upper right', fontsize=8)

        fig.suptitle(f"{feat_title} - All Channels", fontsize=14, fontweight='bold')
        fig.supxlabel("Window Index (Time)")
        fig.supylabel(feature_ylabels[feat_idx])
        plt.tight_layout()

    plt.show()
    print(f"\n  Displayed 5 figures: Raw EMG + 4 features (close windows to continue)")

    return X, y, label_names


# =============================================================================
# FEATURE EXTRACTION (Time-domain features for EMG)
# =============================================================================

class EMGFeatureExtractor:
    """
    Extracts time-domain features from EMG windows.

    Features per channel:
      - RMS (Root Mean Square): Signal power/amplitude
      - WL (Waveform Length): Signal complexity
      - ZC (Zero Crossings): Frequency content indicator
      - SSC (Slope Sign Changes): Frequency content indicator

    These 4 features × N channels = 4N features per window.
    For 4 channels: 16 features total.

    IMPORTANT: Per-window centering (DC offset removal) is applied before
    feature extraction. This is critical because:
      - EMG sensors have DC offset (e.g., ~512 for 10-bit ADC)
      - Zero crossings require signal centered around 0
      - Per-window centering is causal (works in real-time inference)
      - Global centering would leak information across windows

    LESSON: These features are:
      - Fast to compute (good for real-time)
      - Work well with LDA
      - Proven effective for EMG gesture recognition
    """

    def __init__(self, zc_threshold_percent: float = 0.1, ssc_threshold_percent: float = 0.1):
        """
        Args:
            zc_threshold_percent: ZC threshold as fraction of signal RMS
            ssc_threshold_percent: SSC threshold as fraction of signal RMS squared
        """
        self.zc_threshold_percent = zc_threshold_percent
        self.ssc_threshold_percent = ssc_threshold_percent

    def extract_features_single_channel(self, signal: np.ndarray) -> dict:
        """
        Extract all features from a single channel signal.

        Per-window centering is applied first to remove DC offset.
        This is the standard approach for EMG feature extraction.
        """
        # Per-window centering: remove DC offset (critical for ZC/SSC)
        # This only uses data from the current window (causal for real-time)
        signal = signal - np.mean(signal)

        # RMS - Root Mean Square (now measures AC power, not DC offset)
        rms = np.sqrt(np.mean(signal ** 2))

        # WL - Waveform Length
        wl = np.sum(np.abs(np.diff(signal)))

        # Dynamic thresholds based on signal RMS
        zc_thresh = self.zc_threshold_percent * rms
        ssc_thresh = (self.ssc_threshold_percent * rms) ** 2

        # ZC - Zero Crossings (with threshold to reject noise)
        # Now meaningful because signal is centered around 0
        sign_changes = signal[:-1] * signal[1:] < 0
        amplitude_diff = np.abs(np.diff(signal)) > zc_thresh
        zc = np.sum(sign_changes & amplitude_diff)

        # SSC - Slope Sign Changes
        diff_left = signal[1:-1] - signal[:-2]
        diff_right = signal[1:-1] - signal[2:]
        ssc = np.sum((diff_left * diff_right) > ssc_thresh)

        return {'rms': rms, 'wl': wl, 'zc': zc, 'ssc': ssc}

    def extract_features_window(self, window: np.ndarray) -> np.ndarray:
        """
        Extract features from a window of shape (samples, channels).

        Returns flat array: [ch0_rms, ch0_wl, ch0_zc, ch0_ssc, ch1_rms, ...]
        """
        n_channels = window.shape[1]
        features = []

        for ch in range(n_channels):
            ch_features = self.extract_features_single_channel(window[:, ch])
            features.extend([ch_features['rms'], ch_features['wl'],
                           ch_features['zc'], ch_features['ssc']])

        return np.array(features)

    def extract_features_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Extract features from batch of windows.

        Args:
            X: Shape (n_windows, n_samples, n_channels)

        Returns:
            Features array of shape (n_windows, n_features)
            where n_features = n_channels * 4
        """
        n_windows = X.shape[0]
        n_channels = X.shape[2]
        n_features = n_channels * 4  # 4 features per channel

        features = np.zeros((n_windows, n_features))

        for i in range(n_windows):
            features[i] = self.extract_features_window(X[i])

        return features

    def get_feature_names(self, n_channels: int) -> list[str]:
        """Get human-readable feature names."""
        names = []
        for ch in range(n_channels):
            names.extend([f'ch{ch}_rms', f'ch{ch}_wl', f'ch{ch}_zc', f'ch{ch}_ssc'])
        return names


# =============================================================================
# LDA CLASSIFIER
# =============================================================================

class EMGClassifier:
    """
    LDA-based EMG gesture classifier.

    LESSON: Why LDA for EMG?
      - Fast training and inference (good for embedded)
      - Works well with small datasets
      - Interpretable (can visualize decision boundaries)
      - Proven effective for EMG in literature
    """

    def __init__(self):
        self.feature_extractor = EMGFeatureExtractor()
        self.lda = LinearDiscriminantAnalysis()
        self.label_names: list[str] = []
        self.is_trained = False
        self.feature_names: list[str] = []

    def train(self, X: np.ndarray, y: np.ndarray, label_names: list[str]):
        """
        Train the classifier.

        Args:
            X: Raw EMG windows (n_windows, n_samples, n_channels)
            y: Integer labels (n_windows,)
            label_names: List of label strings
        """
        print("\n[Classifier] Extracting features...")
        X_features = self.feature_extractor.extract_features_batch(X)
        self.feature_names = self.feature_extractor.get_feature_names(X.shape[2])

        print(f"[Classifier] Feature matrix shape: {X_features.shape}")
        print(f"[Classifier] Features per window: {len(self.feature_names)}")

        print("\n[Classifier] Training LDA...")
        self.lda.fit(X_features, y)
        self.label_names = label_names
        self.is_trained = True

        # Training accuracy
        train_acc = self.lda.score(X_features, y)
        print(f"[Classifier] Training accuracy: {train_acc*100:.1f}%")

        return X_features

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate classifier on test data."""
        if not self.is_trained:
            raise ValueError("Classifier not trained!")

        X_features = self.feature_extractor.extract_features_batch(X)
        y_pred = self.lda.predict(X_features)

        accuracy = np.mean(y_pred == y)

        return {
            'accuracy': accuracy,
            'y_pred': y_pred,
            'y_true': y
        }

    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> np.ndarray:
        """Perform k-fold cross-validation."""
        print(f"\n[Classifier] Running {cv}-fold cross-validation...")
        X_features = self.feature_extractor.extract_features_batch(X)
        scores = cross_val_score(self.lda, X_features, y, cv=cv)
        return scores

    def predict(self, window: np.ndarray) -> tuple[str, np.ndarray]:
        """
        Predict gesture for a single window.

        Args:
            window: Shape (n_samples, n_channels)

        Returns:
            (predicted_label, probabilities)
        """
        if not self.is_trained:
            raise ValueError("Classifier not trained!")

        features = self.feature_extractor.extract_features_window(window)
        pred_idx = self.lda.predict([features])[0]
        proba = self.lda.predict_proba([features])[0]

        return self.label_names[pred_idx], proba

    def get_feature_importance(self) -> dict:
        """Get feature importance based on LDA coefficients."""
        if not self.is_trained:
            return {}

        # For multi-class, average absolute coefficients across classes
        coef = np.abs(self.lda.coef_).mean(axis=0)
        importance = dict(zip(self.feature_names, coef))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def save(self, filepath: Path) -> Path:
        """
        Save the trained classifier to disk.

        Saves:
          - LDA model parameters
          - Feature extractor settings
          - Label names
          - Feature names

        Args:
            filepath: Path to save the model (e.g., 'models/emg_classifier.joblib')

        Returns:
            Path to the saved model file
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained classifier!")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'lda': self.lda,
            'label_names': self.label_names,
            'feature_names': self.feature_names,
            'feature_extractor_params': {
                'zc_threshold_percent': self.feature_extractor.zc_threshold_percent,
                'ssc_threshold_percent': self.feature_extractor.ssc_threshold_percent,
            },
            'version': '1.0',  # For future compatibility
        }

        joblib.dump(model_data, filepath)
        print(f"[Classifier] Model saved to: {filepath}")
        print(f"[Classifier] File size: {filepath.stat().st_size / 1024:.1f} KB")
        return filepath

    @classmethod
    def load(cls, filepath: Path) -> 'EMGClassifier':
        """
        Load a trained classifier from disk.

        Args:
            filepath: Path to the saved model file

        Returns:
            Loaded EMGClassifier instance ready for prediction
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)

        # Create new instance and restore state
        classifier = cls()
        classifier.lda = model_data['lda']
        classifier.label_names = model_data['label_names']
        classifier.feature_names = model_data['feature_names']
        classifier.is_trained = True

        # Restore feature extractor params
        params = model_data.get('feature_extractor_params', {})
        classifier.feature_extractor = EMGFeatureExtractor(
            zc_threshold_percent=params.get('zc_threshold_percent', 0.1),
            ssc_threshold_percent=params.get('ssc_threshold_percent', 0.1),
        )

        print(f"[Classifier] Model loaded from: {filepath}")
        print(f"[Classifier] Labels: {classifier.label_names}")
        return classifier

    @staticmethod
    def get_default_model_path() -> Path:
        """Get the default path for saving/loading models."""
        return MODEL_DIR / "emg_lda_classifier.joblib"

    @staticmethod
    def list_saved_models() -> list[Path]:
        """List all saved model files."""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        return sorted(MODEL_DIR.glob("*.joblib"))


# =============================================================================
# PREDICTION SMOOTHING (Temporal smoothing, majority vote, debouncing)
# =============================================================================

class PredictionSmoother:
    """
    Smooths predictions to prevent twitchy/unstable output.

    Combines three techniques:
      1. Probability Smoothing: Exponential moving average on raw probabilities
      2. Majority Vote: Output most common prediction from last N predictions
      3. Debouncing: Only change output after N consecutive same predictions

    This prevents the robotic hand from twitching when there's an occasional
    misclassification in a stream of correct predictions.

    Example:
        Raw predictions:    FIST, FIST, OPEN, FIST, FIST, FIST
        Without smoothing:  Hand twitches open briefly
        With smoothing:     Hand stays as FIST (OPEN was filtered out)
    """

    def __init__(
        self,
        label_names: list[str],
        probability_smoothing: float = 0.7,
        majority_vote_window: int = 5,
        debounce_count: int = 3,
    ):
        """
        Args:
            label_names: List of gesture labels (must match classifier output)
            probability_smoothing: EMA factor (0-1). Higher = more smoothing.
                                   0 = no smoothing, 0.9 = very smooth
            majority_vote_window: Number of past predictions to consider for voting
            debounce_count: Number of consecutive same predictions needed to change output
        """
        self.label_names = label_names
        self.n_classes = len(label_names)

        # Probability smoothing (Exponential Moving Average)
        self.prob_smoothing = probability_smoothing
        self.smoothed_proba = np.ones(self.n_classes) / self.n_classes  # Start uniform

        # Majority vote
        self.vote_window = majority_vote_window
        self.prediction_history: list[str] = []

        # Debouncing
        self.debounce_count = debounce_count
        self.current_output = None
        self.pending_output = None
        self.pending_count = 0

        # Stats
        self.total_predictions = 0
        self.output_changes = 0

    def update(self, predicted_label: str, probabilities: np.ndarray) -> tuple[str, float, dict]:
        """
        Process a new prediction and return smoothed output.

        Args:
            predicted_label: Raw prediction from classifier
            probabilities: Raw probability array from classifier

        Returns:
            (smoothed_label, confidence, debug_info)
            - smoothed_label: The stable output label after all smoothing
            - confidence: Confidence in the smoothed output (0-1)
            - debug_info: Dict with intermediate values for debugging/display
        """
        self.total_predictions += 1

        # --- 1. Probability Smoothing (EMA) ---
        # Blend new probabilities with historical smoothed probabilities
        self.smoothed_proba = (
            self.prob_smoothing * self.smoothed_proba +
            (1 - self.prob_smoothing) * probabilities
        )

        # Get prediction from smoothed probabilities
        prob_smoothed_idx = np.argmax(self.smoothed_proba)
        prob_smoothed_label = self.label_names[prob_smoothed_idx]
        prob_smoothed_confidence = self.smoothed_proba[prob_smoothed_idx]

        # --- 2. Majority Vote ---
        # Add to history and keep window size
        self.prediction_history.append(prob_smoothed_label)
        if len(self.prediction_history) > self.vote_window:
            self.prediction_history.pop(0)

        # Count votes
        vote_counts = {}
        for pred in self.prediction_history:
            vote_counts[pred] = vote_counts.get(pred, 0) + 1

        # Get majority winner
        majority_label = max(vote_counts, key=vote_counts.get)
        majority_count = vote_counts[majority_label]
        majority_confidence = majority_count / len(self.prediction_history)

        # --- 3. Debouncing ---
        # Only change output after consistent predictions
        if self.current_output is None:
            # First prediction
            self.current_output = majority_label
            self.pending_output = majority_label
            self.pending_count = 1
        elif majority_label == self.current_output:
            # Same as current output, reset pending
            self.pending_output = majority_label
            self.pending_count = 1
        elif majority_label == self.pending_output:
            # Same as pending, increment count
            self.pending_count += 1
            if self.pending_count >= self.debounce_count:
                # Enough consecutive predictions, change output
                self.current_output = majority_label
                self.output_changes += 1
        else:
            # New prediction, start new pending
            self.pending_output = majority_label
            self.pending_count = 1

        # Final output
        final_label = self.current_output
        final_confidence = majority_confidence

        # Debug info
        debug_info = {
            'raw_label': predicted_label,
            'raw_confidence': float(np.max(probabilities)),
            'prob_smoothed_label': prob_smoothed_label,
            'prob_smoothed_confidence': float(prob_smoothed_confidence),
            'majority_label': majority_label,
            'majority_confidence': float(majority_confidence),
            'vote_counts': vote_counts,
            'pending_output': self.pending_output,
            'pending_count': self.pending_count,
            'debounce_threshold': self.debounce_count,
        }

        return final_label, final_confidence, debug_info

    def reset(self):
        """Reset all state (call when starting a new prediction session)."""
        self.smoothed_proba = np.ones(self.n_classes) / self.n_classes
        self.prediction_history = []
        self.current_output = None
        self.pending_output = None
        self.pending_count = 0
        self.total_predictions = 0
        self.output_changes = 0

    def get_stats(self) -> dict:
        """Get statistics about smoothing effectiveness."""
        return {
            'total_predictions': self.total_predictions,
            'output_changes': self.output_changes,
            'stability_ratio': 1 - (self.output_changes / max(1, self.total_predictions)),
        }


# =============================================================================
# TRAINING (Train LDA classifier)
# =============================================================================

def run_training_demo():
    """
    Train an LDA classifier on ALL collected sessions combined.

    Shows:
      1. Loading all session data combined
      2. Feature extraction
      3. Training LDA
      4. Cross-validation evaluation
      5. Feature importance analysis

    The model learns from all accumulated data, making it more robust
    as you collect more sessions over time.
    """
    print("\n" + "=" * 60)
    print("TRAIN LDA CLASSIFIER (ALL SESSIONS)")
    print("=" * 60)

    storage = SessionStorage()
    sessions = storage.list_sessions()

    if not sessions:
        print("\nNo saved sessions found!")
        print("Run option 1 first to collect and save training data.")
        return None

    # Show available sessions
    print(f"\nFound {len(sessions)} saved session(s):")
    print("-" * 40)

    total_windows = 0
    for session_id in sessions:
        info = storage.get_session_info(session_id)
        print(f"  - {session_id}: {info['num_windows']} windows, gestures: {info['gestures']}")
        total_windows += info['num_windows']

    print(f"\nTotal windows across all sessions: {total_windows}")
    print("-" * 40)

    confirm = input("\nTrain on ALL sessions combined? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Training cancelled.")
        return None

    # Load ALL data combined
    print(f"\n{'=' * 60}")
    print("TRAINING ON ALL SESSIONS COMBINED")
    print("=" * 60)

    X, y, label_names, loaded_sessions = storage.load_all_for_training()

    print(f"\nDataset:")
    print(f"  Windows: {X.shape[0]}")
    print(f"  Samples per window: {X.shape[1]}")
    print(f"  Channels: {X.shape[2]}")
    print(f"  Classes: {label_names}")

    # Count per class
    print(f"\nSamples per class:")
    for i, name in enumerate(label_names):
        count = np.sum(y == i)
        print(f"  {name}: {count}")

    # Create and train classifier
    classifier = EMGClassifier()
    X_features = classifier.train(X, y, label_names)

    # Cross-validation
    cv_scores = classifier.cross_validate(X, y, cv=5)
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean()*100:.1f}% (+/- {cv_scores.std()*100:.1f}%)")

    # Feature importance
    print(f"\n{'-' * 40}")
    print("FEATURE IMPORTANCE (top 8)")
    print("-" * 40)
    importance = classifier.get_feature_importance()
    for i, (name, score) in enumerate(list(importance.items())[:8]):
        bar = "█" * int(score * 20)
        print(f"  {name:12s}: {bar} ({score:.3f})")

    # Train/test split evaluation
    print(f"\n{'-' * 40}")
    print("TRAIN/TEST SPLIT EVALUATION")
    print("-" * 40)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train on train set
    test_classifier = EMGClassifier()
    test_classifier.train(X_train, y_train, label_names)

    # Evaluate on test set
    result = test_classifier.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {result['accuracy']*100:.1f}%")

    print(f"\nClassification Report:")
    print(classification_report(result['y_true'], result['y_pred'],
                               target_names=label_names))

    print(f"Confusion Matrix:")
    cm = confusion_matrix(result['y_true'], result['y_pred'])
    print(f"  {'':12s} ", end="")
    for name in label_names:
        print(f"{name[:8]:>8s} ", end="")
    print()
    for i, row in enumerate(cm):
        print(f"  {label_names[i]:12s} ", end="")
        for val in row:
            print(f"{val:8d} ", end="")
        print()

    # --- Save the model ---
    print(f"\n{'-' * 40}")
    print("SAVE MODEL")
    print("-" * 40)

    default_path = EMGClassifier.get_default_model_path()
    print(f"Default save path: {default_path}")

    save_choice = input("\nSave this model? (y/n): ").strip().lower()
    if save_choice == 'y':
        classifier.save(default_path)
        print(f"\nModel saved! You can now use 'Live prediction' without retraining.")

    return classifier


# =============================================================================
# LIVE PREDICTION (Real-time gesture classification)
# =============================================================================

def run_prediction_demo():
    """
    Live prediction demo - classifies gestures in real-time.

    Shows:
      1. Load saved model OR train fresh on all sessions
      2. Stream simulated EMG data
      3. Classify each window as it comes in
      4. Display predictions with confidence
    """
    print("\n" + "=" * 60)
    print("LIVE PREDICTION DEMO")
    print("=" * 60)

    # Check for saved model
    saved_models = EMGClassifier.list_saved_models()
    default_model = EMGClassifier.get_default_model_path()

    classifier = None

    if default_model.exists():
        print(f"\nSaved model found: {default_model}")
        print(f"  File size: {default_model.stat().st_size / 1024:.1f} KB")

        load_choice = input("\nLoad saved model? (y=load, n=retrain): ").strip().lower()
        if load_choice == 'y':
            classifier = EMGClassifier.load(default_model)

    if classifier is None:
        # Need to train a new model
        storage = SessionStorage()
        sessions = storage.list_sessions()

        if not sessions:
            print("\nNo saved sessions found! Collect data first (Option 1).")
            return None

        # Show available sessions
        print(f"\nNo saved model (or retraining requested).")
        print(f"Will train on ALL {len(sessions)} session(s):")
        print("-" * 40)

        total_windows = 0
        for session_id in sessions:
            info = storage.get_session_info(session_id)
            print(f"  - {session_id}: {info['num_windows']} windows")
            total_windows += info['num_windows']

        print(f"\nTotal training windows: {total_windows}")
        print("-" * 40)

        confirm = input("\nTrain and start prediction? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Prediction cancelled.")
            return None

        # Load ALL sessions and train model
        print(f"\n[Training model on all sessions...]")
        X, y, label_names, loaded_sessions = storage.load_all_for_training()

        classifier = EMGClassifier()
        classifier.train(X, y, label_names)

    # Start live prediction
    print("\n" + "=" * 60)
    print("STARTING LIVE PREDICTION (WITH SMOOTHING)")
    print("=" * 60)

    max_predictions = 50  # Stop after this many predictions
    print(f"Running {max_predictions} predictions with smoothing enabled...\n")
    print("  Smoothing: Probability EMA (0.7) + Majority Vote (5) + Debounce (3)\n")

    stream = GestureAwareEMGStream(num_channels=NUM_CHANNELS, sample_rate=SAMPLING_RATE_HZ)
    parser = EMGParser(num_channels=NUM_CHANNELS)
    windower = Windower(window_size_ms=WINDOW_SIZE_MS, sample_rate=SAMPLING_RATE_HZ, overlap=0.0)

    # Create prediction smoother
    smoother = PredictionSmoother(
        label_names=classifier.label_names,
        probability_smoothing=0.7,   # Higher = more smoothing
        majority_vote_window=5,      # Past predictions to consider
        debounce_count=3,            # Consecutive predictions needed to change
    )

    # Cycle through gestures for demo (names match ESP32 gesture definitions)
    gesture_cycle = ["rest", "open", "fist", "hook_em", "thumbs_up"]
    gesture_idx = 0
    gesture_duration = 2.5  # seconds per gesture
    gesture_start = time.perf_counter()
    current_gesture = gesture_cycle[0]
    stream.set_gesture(current_gesture)
    print(f"  [Simulating: {current_gesture.upper()}]")

    stream.start()
    prediction_count = 0

    try:
        while prediction_count < max_predictions:
            # Change simulated gesture periodically
            elapsed = time.perf_counter() - gesture_start
            if elapsed > gesture_duration:
                gesture_idx = (gesture_idx + 1) % len(gesture_cycle)
                gesture_start = time.perf_counter()
                current_gesture = gesture_cycle[gesture_idx]
                stream.set_gesture(current_gesture)
                print(f"\n  [Simulating: {current_gesture.upper()}]")

            # Read and process data
            line = stream.readline()
            if line:
                sample = parser.parse_line(line)
                if sample:
                    window = windower.add_sample(sample)
                    if window:
                        # Classify the window (raw prediction)
                        window_data = window.to_numpy()
                        raw_label, proba = classifier.predict(window_data)

                        # Apply smoothing
                        smoothed_label, smoothed_conf, debug = smoother.update(raw_label, proba)

                        prediction_count += 1

                        # Display both raw and smoothed predictions
                        raw_conf = max(proba) * 100
                        smoothed_conf_pct = smoothed_conf * 100

                        # Visual bar for smoothed confidence
                        bar_len = round(smoothed_conf_pct / 5)
                        bar = "█" * bar_len + "░" * (20 - bar_len)

                        # Show raw vs smoothed (smoothed is the stable output)
                        raw_marker = "  " if raw_label == smoothed_label else "!!"
                        print(f"  #{prediction_count:3d} │ {bar} │ {smoothed_label:12s} ({smoothed_conf_pct:5.1f}%) {raw_marker} raw:{raw_label[:8]:8s}")

    except KeyboardInterrupt:
        print("\n\n[Stopped by user]")

    finally:
        stream.stop()

    # Show smoothing stats
    stats = smoother.get_stats()
    print(f"\n" + "-" * 40)
    print(f"SMOOTHING STATISTICS")
    print("-" * 40)
    print(f"  Total predictions: {stats['total_predictions']}")
    print(f"  Output changes: {stats['output_changes']}")
    print(f"  Stability ratio: {stats['stability_ratio']*100:.1f}%")
    print(f"\n  (Without smoothing, output would change with every raw prediction)")

    return classifier


# =============================================================================
# LDA VISUALIZATION (Decision boundaries and feature space)
# =============================================================================

def run_visualization_demo():
    """
    Visualize the LDA model trained on ALL sessions with plots:
      1. 2D feature space scatter plot (LDA reduced)
      2. Decision boundaries
      3. Class distributions
      4. Confusion matrix heatmap

    Uses all accumulated session data for a complete picture of the model.
    """
    print("\n" + "=" * 60)
    print("LDA VISUALIZATION (ALL SESSIONS)")
    print("=" * 60)

    storage = SessionStorage()
    sessions = storage.list_sessions()

    if not sessions:
        print("\nNo saved sessions found! Collect data first (Option 1).")
        return None

    # Show available sessions
    print(f"\nFound {len(sessions)} saved session(s):")
    print("-" * 40)

    total_windows = 0
    for session_id in sessions:
        info = storage.get_session_info(session_id)
        print(f"  - {session_id}: {info['num_windows']} windows")
        total_windows += info['num_windows']

    print(f"\nTotal windows: {total_windows}")
    print("-" * 40)

    confirm = input("\nVisualize model trained on ALL sessions? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Visualization cancelled.")
        return None

    # Load ALL data combined
    X, y, label_names, loaded_sessions = storage.load_all_for_training()

    # Extract features
    extractor = EMGFeatureExtractor()
    X_features = extractor.extract_features_batch(X)

    # Train LDA
    print("\n[Training LDA for visualization...]")
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_features, y)

    # Transform to LDA space (reduces to n_classes - 1 dimensions)
    X_lda = lda.transform(X_features)

    n_classes = len(label_names)
    print(f"  LDA dimensions: {X_lda.shape[1]}")

    # Color scheme
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_classes))

    # --- Figure 1: LDA Feature Space (2D projection) ---
    fig1, ax1 = plt.subplots(figsize=(10, 8))

    for i, label in enumerate(label_names):
        mask = y == i
        ax1.scatter(X_lda[mask, 0],
                   X_lda[mask, 1] if X_lda.shape[1] > 1 else np.zeros(mask.sum()),
                   c=[colors[i]], label=label, s=100, alpha=0.7, edgecolors='white', linewidth=1)

    # Add class means
    for i, label in enumerate(label_names):
        mask = y == i
        mean_x = X_lda[mask, 0].mean()
        mean_y = X_lda[mask, 1].mean() if X_lda.shape[1] > 1 else 0
        ax1.scatter(mean_x, mean_y, c=[colors[i]], s=400, marker='X', edgecolors='black', linewidth=2)
        ax1.annotate(label.upper(), (mean_x, mean_y), fontsize=12, fontweight='bold',
                    ha='center', va='bottom', xytext=(0, 15), textcoords='offset points')

    ax1.set_xlabel("LDA Component 1", fontsize=12)
    ax1.set_ylabel("LDA Component 2", fontsize=12)
    ax1.set_title("LDA Feature Space - Gesture Clusters", fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # --- Figure 2: Decision Boundary Heatmap ---
    if X_lda.shape[1] >= 1:
        fig2, ax2 = plt.subplots(figsize=(10, 8))

        # Create mesh grid
        x_min, x_max = X_lda[:, 0].min() - 1, X_lda[:, 0].max() + 1
        if X_lda.shape[1] > 1:
            y_min, y_max = X_lda[:, 1].min() - 1, X_lda[:, 1].max() + 1
        else:
            y_min, y_max = -2, 2

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                            np.linspace(y_min, y_max, 200))

        # For prediction, we need to go back to original feature space
        # Use simplified approach: train new LDA on LDA-transformed features
        if X_lda.shape[1] > 1:
            lda_2d = LinearDiscriminantAnalysis()
            lda_2d.fit(X_lda[:, :2], y)
            Z = lda_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        else:
            # 1D case - simple threshold
            Z = lda.predict(X_features[0:1])  # dummy
            Z = np.zeros(xx.ravel().shape)
            for i, x_val in enumerate(xx.ravel()):
                if X_lda.shape[1] == 1:
                    # Find closest class mean
                    distances = [abs(x_val - X_lda[y == c, 0].mean()) for c in range(n_classes)]
                    Z[i] = np.argmin(distances)

        Z = Z.reshape(xx.shape)

        # Plot decision regions
        ax2.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(-0.5, n_classes, 1),
                    colors=[colors[i] for i in range(n_classes)])
        ax2.contour(xx, yy, Z, colors='black', linewidths=0.5, alpha=0.5)

        # Plot data points
        for i, label in enumerate(label_names):
            mask = y == i
            ax2.scatter(X_lda[mask, 0],
                       X_lda[mask, 1] if X_lda.shape[1] > 1 else np.zeros(mask.sum()),
                       c=[colors[i]], label=label, s=80, alpha=0.9, edgecolors='black', linewidth=0.5)

        ax2.set_xlabel("LDA Component 1", fontsize=12)
        ax2.set_ylabel("LDA Component 2", fontsize=12)
        ax2.set_title("LDA Decision Boundaries", fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)

    # --- Figure 3: Feature Importance Radar Chart ---
    fig3, ax3 = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

    feature_names = extractor.get_feature_names(X.shape[2])
    coef = np.abs(lda.coef_).mean(axis=0)
    coef_normalized = coef / coef.max()  # Normalize to 0-1

    # Number of features
    n_features = len(feature_names)
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()

    # Complete the loop
    coef_normalized = np.concatenate([coef_normalized, [coef_normalized[0]]])
    angles += angles[:1]

    ax3.plot(angles, coef_normalized, 'o-', linewidth=2, color='#2E86AB', markersize=8)
    ax3.fill(angles, coef_normalized, alpha=0.25, color='#2E86AB')
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(feature_names, fontsize=9)
    ax3.set_ylim(0, 1.1)
    ax3.set_title("Feature Importance (Radar)", fontsize=14, fontweight='bold', pad=20)

    # --- Figure 4: Class Distribution Histograms ---
    fig4, axes4 = plt.subplots(1, n_classes, figsize=(4*n_classes, 4))
    if n_classes == 1:
        axes4 = [axes4]

    for i, (ax, label) in enumerate(zip(axes4, label_names)):
        mask = y == i
        ax.hist(X_lda[mask, 0], bins=15, color=colors[i], alpha=0.7, edgecolor='black')
        ax.axvline(X_lda[mask, 0].mean(), color='black', linestyle='--', linewidth=2, label='Mean')
        ax.set_xlabel("LDA Component 1")
        ax.set_ylabel("Count")
        ax.set_title(f"{label.upper()}", fontsize=12, fontweight='bold')
        ax.legend()

    fig4.suptitle("Class Distributions on LDA Axis", fontsize=14, fontweight='bold')
    plt.tight_layout()

    # --- Figure 5: Confusion Matrix Heatmap ---
    from sklearn.model_selection import cross_val_predict

    fig5, ax5 = plt.subplots(figsize=(8, 6))

    y_pred = cross_val_predict(lda, X_features, y, cv=5)
    cm = confusion_matrix(y, y_pred)

    im = ax5.imshow(cm, interpolation='nearest', cmap='Blues')
    ax5.figure.colorbar(im, ax=ax5)

    ax5.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=label_names,
           yticklabels=label_names,
           xlabel='Predicted',
           ylabel='Actual',
           title='Confusion Matrix Heatmap')

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax5.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()

    print("\n  Displayed 5 visualization figures")
    return lda


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print(__doc__)

    while True:
        print("\n" + "=" * 60)
        print("EMG DATA COLLECTION PIPELINE")
        print("=" * 60)
        print("\nOptions:")
        print("  1. Collect data (labeled session)")
        print("  2. Inspect saved sessions (view features)")
        print("  3. Train LDA classifier")
        print("  4. Live prediction demo")
        print("  5. Visualize LDA model")
        print("  q. Quit")

        choice = input("\nEnter choice: ").strip().lower()

        if choice == 'q':
            print("\nGoodbye!")
            break

        elif choice == "1":
            windows, labels = run_labeled_collection_demo()

        elif choice == "2":
            result = run_storage_demo()

        elif choice == "3":
            classifier = run_training_demo()

        elif choice == "4":
            classifier = run_prediction_demo()

        elif choice == "5":
            lda = run_visualization_demo()

        else:
            print("\nInvalid choice. Please enter 1-5 or q.")