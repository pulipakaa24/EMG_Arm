"""
EMG Data Collection GUI
=======================
A modern GUI for the EMG data collection pipeline.

Features:
  - Data collection with live EMG visualization and gesture prompts
  - Session inspector with signal and feature plots
  - Model training with progress and results
  - Live prediction demo
  - LDA visualization

Requirements:
  pip install customtkinter matplotlib numpy

Run:
  python emg_gui.py
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import threading
import queue
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Import from the existing pipeline
from learning_data_collection import (
    # Configuration
    NUM_CHANNELS, SAMPLING_RATE_HZ, WINDOW_SIZE_MS, WINDOW_OVERLAP,
    GESTURE_HOLD_SEC, REST_BETWEEN_SEC, REPS_PER_GESTURE, DATA_DIR, USER_ID,
    # Classes
    EMGSample, EMGWindow, EMGParser, Windower,
    GestureAwareEMGStream, SimulatedEMGStream,
    PromptScheduler, SessionStorage, SessionMetadata,
    EMGFeatureExtractor, EMGClassifier, PredictionSmoother,
)

# Import real serial stream for ESP32 hardware
from serial_stream import RealSerialStream
import serial.tools.list_ports

# =============================================================================
# APPEARANCE SETTINGS
# =============================================================================

ctk.set_appearance_mode("dark")  # "dark", "light", or "system"
ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"

# Colors for gestures (names match ESP32 gesture definitions)
GESTURE_COLORS = {
    "rest": "#6c757d",        # Gray
    "open": "#17a2b8",        # Cyan
    "fist": "#007bff",        # Blue
    "hook_em": "#fd7e14",     # Orange (Hook 'em Horns)
    "thumbs_up": "#28a745",   # Green
}

def get_gesture_color(gesture_name: str) -> str:
    """Get color for a gesture name."""
    for key, color in GESTURE_COLORS.items():
        if key in gesture_name.lower():
            return color
    return "#dc3545"  # Red for unknown


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class EMGApp(ctk.CTk):
    """Main application window."""

    def __init__(self):
        super().__init__()

        self.title("EMG Data Collection Pipeline")
        self.geometry("1400x900")
        self.minsize(1200, 700)

        # Configure grid
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Create sidebar
        self.sidebar = Sidebar(self, self.show_page)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        # Create container for pages
        self.page_container = ctk.CTkFrame(self, fg_color="transparent")
        self.page_container.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.page_container.grid_columnconfigure(0, weight=1)
        self.page_container.grid_rowconfigure(0, weight=1)

        # Create pages
        self.pages = {}
        self.pages["collection"] = CollectionPage(self.page_container)
        self.pages["inspect"] = InspectPage(self.page_container)
        self.pages["training"] = TrainingPage(self.page_container)
        self.pages["prediction"] = PredictionPage(self.page_container)
        self.pages["visualization"] = VisualizationPage(self.page_container)

        # Show default page
        self.current_page = None
        self.show_page("collection")

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def show_page(self, page_name: str):
        """Show a specific page."""
        # Hide current page
        if self.current_page:
            self.pages[self.current_page].grid_forget()
            self.pages[self.current_page].on_hide()

        # Show new page
        self.pages[page_name].grid(row=0, column=0, sticky="nsew")
        self.pages[page_name].on_show()
        self.current_page = page_name

        # Update sidebar selection
        self.sidebar.set_active(page_name)

    def on_close(self):
        """Handle window close."""
        # Stop any running processes in pages
        for page in self.pages.values():
            if hasattr(page, 'stop'):
                page.stop()
        self.destroy()


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================

class Sidebar(ctk.CTkFrame):
    """Sidebar navigation panel."""

    def __init__(self, parent, on_select_callback):
        super().__init__(parent, width=200, corner_radius=0)
        self.on_select = on_select_callback

        # Logo/Title
        self.logo_label = ctk.CTkLabel(
            self, text="EMG Pipeline",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.logo_label.pack(pady=(20, 10))

        self.subtitle = ctk.CTkLabel(
            self, text="Data Collection & ML",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.subtitle.pack(pady=(0, 20))

        # Navigation buttons
        self.nav_buttons = {}

        nav_items = [
            ("collection", "1. Collect Data"),
            ("inspect", "2. Inspect Sessions"),
            ("training", "3. Train Model"),
            ("prediction", "4. Live Prediction"),
            ("visualization", "5. Visualize LDA"),
        ]

        for page_id, label in nav_items:
            btn = ctk.CTkButton(
                self, text=label,
                font=ctk.CTkFont(size=14),
                height=40,
                corner_radius=8,
                fg_color="transparent",
                text_color=("gray10", "gray90"),
                hover_color=("gray70", "gray30"),
                anchor="w",
                command=lambda p=page_id: self.on_select(p)
            )
            btn.pack(fill="x", padx=10, pady=5)
            self.nav_buttons[page_id] = btn

        # Spacer
        spacer = ctk.CTkLabel(self, text="")
        spacer.pack(expand=True)

        # Status area
        self.status_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.status_frame.pack(fill="x", padx=10, pady=10)

        self.session_count_label = ctk.CTkLabel(
            self.status_frame, text="Sessions: 0",
            font=ctk.CTkFont(size=12)
        )
        self.session_count_label.pack()

        self.model_status_label = ctk.CTkLabel(
            self.status_frame, text="Model: Not saved",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.model_status_label.pack()

        # Update status
        self.update_status()

    def set_active(self, page_id: str):
        """Set the active navigation button."""
        for pid, btn in self.nav_buttons.items():
            if pid == page_id:
                btn.configure(fg_color=("gray75", "gray25"))
            else:
                btn.configure(fg_color="transparent")

    def update_status(self):
        """Update the status display."""
        storage = SessionStorage()
        sessions = storage.list_sessions()
        self.session_count_label.configure(text=f"Sessions: {len(sessions)}")

        model_path = EMGClassifier.get_default_model_path()
        if model_path.exists():
            self.model_status_label.configure(text="Model: Saved", text_color="green")
        else:
            self.model_status_label.configure(text="Model: Not saved", text_color="gray")


# =============================================================================
# BASE PAGE CLASS
# =============================================================================

class BasePage(ctk.CTkFrame):
    """Base class for all pages."""

    def __init__(self, parent):
        super().__init__(parent, fg_color="transparent")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

    def on_show(self):
        """Called when page is shown."""
        pass

    def on_hide(self):
        """Called when page is hidden."""
        pass

    def create_header(self, title: str, subtitle: str = ""):
        """Create a page header."""
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 20))

        title_label = ctk.CTkLabel(
            header_frame, text=title,
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title_label.pack(anchor="w")

        if subtitle:
            subtitle_label = ctk.CTkLabel(
                header_frame, text=subtitle,
                font=ctk.CTkFont(size=14),
                text_color="gray"
            )
            subtitle_label.pack(anchor="w")

        return header_frame


# =============================================================================
# DATA COLLECTION PAGE
# =============================================================================

class CollectionPage(BasePage):
    """Data collection page with live EMG visualization."""

    def __init__(self, parent):
        super().__init__(parent)

        self.create_header(
            "Data Collection",
            "Collect labeled EMG data with timed gesture prompts"
        )

        # Collection state (MUST be initialized BEFORE setup_controls)
        self.is_collecting = False
        self.is_connected = False
        self.using_real_hardware = False
        self.stream = None
        self.parser = None
        self.windower = None
        self.scheduler = None
        self.collected_windows = []
        self.collected_labels = []
        self.sample_buffer = []
        self.collection_thread = None
        self.data_queue = queue.Queue()

        # Main content area
        self.content = ctk.CTkFrame(self)
        self.content.grid(row=1, column=0, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_columnconfigure(1, weight=2)
        self.content.grid_rowconfigure(0, weight=1)

        # Left panel - Controls
        self.controls_panel = ctk.CTkFrame(self.content)
        self.controls_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=0)
        self.setup_controls()

        # Right panel - Live plot and prompt
        self.plot_panel = ctk.CTkFrame(self.content)
        self.plot_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=0)
        self.setup_plot()

    def setup_controls(self):
        """Setup the control panel."""
        # User ID
        user_frame = ctk.CTkFrame(self.controls_panel, fg_color="transparent")
        user_frame.pack(fill="x", padx=20, pady=(20, 10))

        ctk.CTkLabel(user_frame, text="User ID:", font=ctk.CTkFont(size=14)).pack(anchor="w")
        self.user_id_entry = ctk.CTkEntry(user_frame, placeholder_text="user_001")
        self.user_id_entry.pack(fill="x", pady=(5, 0))
        self.user_id_entry.insert(0, USER_ID)

        # Data Source selection
        source_frame = ctk.CTkFrame(self.controls_panel, fg_color="transparent")
        source_frame.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(source_frame, text="Data Source:", font=ctk.CTkFont(size=14)).pack(anchor="w")

        self.source_var = ctk.StringVar(value="simulated")

        radio_frame = ctk.CTkFrame(source_frame, fg_color="transparent")
        radio_frame.pack(fill="x", pady=(5, 0))

        self.sim_radio = ctk.CTkRadioButton(
            radio_frame, text="Simulated", variable=self.source_var, value="simulated",
            command=self._on_source_change
        )
        self.sim_radio.pack(side="left", padx=(0, 20))

        self.real_radio = ctk.CTkRadioButton(
            radio_frame, text="Real ESP32", variable=self.source_var, value="real",
            command=self._on_source_change
        )
        self.real_radio.pack(side="left")

        # Port selection (initially hidden, shown when "Real ESP32" selected)
        self.port_frame = ctk.CTkFrame(source_frame, fg_color="transparent")
        # Don't pack yet - _on_source_change will handle visibility

        port_select_frame = ctk.CTkFrame(self.port_frame, fg_color="transparent")
        port_select_frame.pack(fill="x", pady=(5, 0))

        ctk.CTkLabel(port_select_frame, text="Port:").pack(side="left")

        self.port_var = ctk.StringVar(value="Auto-detect")
        self.port_dropdown = ctk.CTkOptionMenu(
            port_select_frame, variable=self.port_var,
            values=["Auto-detect"], width=150
        )
        self.port_dropdown.pack(side="left", padx=(10, 5))

        self.refresh_ports_btn = ctk.CTkButton(
            port_select_frame, text="⟳", width=30,
            command=self._refresh_ports
        )
        self.refresh_ports_btn.pack(side="left")

        # Connection status and button
        connect_frame = ctk.CTkFrame(self.port_frame, fg_color="transparent")
        connect_frame.pack(fill="x", pady=(5, 0))

        self.connect_button = ctk.CTkButton(
            connect_frame, text="Connect",
            width=100, height=28,
            command=self._toggle_connection,
            state="disabled"  # Disabled until "Real ESP32" selected
        )
        self.connect_button.pack(side="left", padx=(0, 10))

        self.connection_status = ctk.CTkLabel(
            connect_frame, text="● Disconnected",
            font=ctk.CTkFont(size=11), text_color="gray"
        )
        self.connection_status.pack(side="left")

        # Gesture selection
        gesture_frame = ctk.CTkFrame(self.controls_panel, fg_color="transparent")
        gesture_frame.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(gesture_frame, text="Gestures:", font=ctk.CTkFont(size=14)).pack(anchor="w")

        self.gesture_vars = {}
        available_gestures = ["open", "fist", "hook_em", "thumbs_up"]

        for gesture in available_gestures:
            var = ctk.BooleanVar(value=True)  # All selected by default
            cb = ctk.CTkCheckBox(gesture_frame, text=gesture.replace("_", " ").title(), variable=var)
            cb.pack(anchor="w", pady=2)
            self.gesture_vars[gesture] = var

        # Settings
        settings_frame = ctk.CTkFrame(self.controls_panel, fg_color="transparent")
        settings_frame.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(settings_frame, text="Settings:", font=ctk.CTkFont(size=14)).pack(anchor="w")

        # Hold duration
        hold_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        hold_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(hold_frame, text="Hold (sec):").pack(side="left")
        self.hold_slider = ctk.CTkSlider(hold_frame, from_=1, to=5, number_of_steps=8)
        self.hold_slider.set(GESTURE_HOLD_SEC)
        self.hold_slider.pack(side="left", fill="x", expand=True, padx=10)
        self.hold_label = ctk.CTkLabel(hold_frame, text=f"{GESTURE_HOLD_SEC:.1f}")
        self.hold_label.pack(side="right")
        self.hold_slider.configure(command=lambda v: self.hold_label.configure(text=f"{v:.1f}"))

        # Reps
        reps_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        reps_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(reps_frame, text="Reps:").pack(side="left")
        self.reps_slider = ctk.CTkSlider(reps_frame, from_=1, to=5, number_of_steps=4)
        self.reps_slider.set(REPS_PER_GESTURE)
        self.reps_slider.pack(side="left", fill="x", expand=True, padx=10)
        self.reps_label = ctk.CTkLabel(reps_frame, text=f"{REPS_PER_GESTURE}")
        self.reps_label.pack(side="right")
        self.reps_slider.configure(command=lambda v: self.reps_label.configure(text=f"{int(v)}"))

        # Buttons
        button_frame = ctk.CTkFrame(self.controls_panel, fg_color="transparent")
        button_frame.pack(fill="x", padx=20, pady=20)

        self.start_button = ctk.CTkButton(
            button_frame, text="Start Collection",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=50,
            command=self.toggle_collection
        )
        self.start_button.pack(fill="x", pady=5)

        self.save_button = ctk.CTkButton(
            button_frame, text="Save Session",
            font=ctk.CTkFont(size=14),
            height=40,
            state="disabled",
            command=self.save_session
        )
        self.save_button.pack(fill="x", pady=5)

        # Progress
        progress_frame = ctk.CTkFrame(self.controls_panel, fg_color="transparent")
        progress_frame.pack(fill="x", padx=20, pady=10)

        self.progress_bar = ctk.CTkProgressBar(progress_frame)
        self.progress_bar.pack(fill="x", pady=5)
        self.progress_bar.set(0)

        self.status_label = ctk.CTkLabel(
            progress_frame, text="Ready to collect",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack()

        self.window_count_label = ctk.CTkLabel(
            progress_frame, text="Windows: 0",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.window_count_label.pack()

    def setup_plot(self):
        """Setup the live plot area."""
        # Gesture prompt display
        self.prompt_frame = ctk.CTkFrame(self.plot_panel)
        self.prompt_frame.pack(fill="x", padx=20, pady=20)

        self.prompt_label = ctk.CTkLabel(
            self.prompt_frame, text="READY",
            font=ctk.CTkFont(size=48, weight="bold"),
            text_color="gray",
            width=500,  # Fixed width to prevent resizing glitches
        )
        self.prompt_label.pack(pady=30)

        self.countdown_label = ctk.CTkLabel(
            self.prompt_frame, text="",
            font=ctk.CTkFont(size=18)
        )
        self.countdown_label.pack()

        # Matplotlib figure for live EMG
        self.fig = Figure(figsize=(8, 5), dpi=100, facecolor='#2b2b2b')
        self.axes = []

        for i in range(NUM_CHANNELS):
            ax = self.fig.add_subplot(NUM_CHANNELS, 1, i + 1)
            ax.set_facecolor('#2b2b2b')
            ax.tick_params(colors='white')
            ax.set_ylabel(f'Ch{i}', color='white', fontsize=10)
            ax.set_xlim(0, 500)
            ax.set_ylim(0, 1024)
            ax.grid(True, alpha=0.3)
            for spine in ax.spines.values():
                spine.set_color('white')
            self.axes.append(ax)

        self.axes[-1].set_xlabel('Samples', color='white')
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_panel)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=(0, 20))

        # Initialize plot lines
        self.plot_lines = []
        self.plot_data = [np.zeros(500) for _ in range(NUM_CHANNELS)]

        for i, ax in enumerate(self.axes):
            line, = ax.plot(self.plot_data[i], color='#00ff88', linewidth=1)
            self.plot_lines.append(line)

    def toggle_collection(self):
        """Start or stop collection."""
        print("\n" + "="*80)
        print("[DEBUG] toggle_collection() called")
        print(f"[DEBUG] Current state:")
        print(f"  - is_collecting: {self.is_collecting}")
        print(f"  - is_connected: {self.is_connected}")
        print(f"  - using_real_hardware: {self.using_real_hardware}")
        print(f"  - source_var: {self.source_var.get()}")
        print(f"  - stream exists: {self.stream is not None}")
        if self.stream:
            if hasattr(self.stream, 'state'):
                print(f"  - stream.state: {self.stream.state}")
        print(f"  - button text: {self.start_button.cget('text')}")
        print(f"  - button state: {self.start_button.cget('state')}")

        # Prevent rapid double-clicks from interfering
        if hasattr(self, '_toggling') and self._toggling:
            print("[DEBUG] BLOCKED: Already toggling (debounce)")
            print("="*80 + "\n")
            return

        self._toggling = True
        try:
            if self.is_collecting:
                print("[DEBUG] Branch: STOPPING collection")
                self.stop_collection()
            else:
                print("[DEBUG] Branch: STARTING collection")
                self.start_collection()
        finally:
            # Reset flag after brief delay to prevent immediate re-trigger
            self.after(100, lambda: setattr(self, '_toggling', False))

    def start_collection(self):
        """Start data collection."""
        print("[DEBUG] start_collection() entered")

        # CRITICAL: Drain any stale messages from previous sessions FIRST
        # This prevents old 'done' messages from stopping the new session
        stale_count = 0
        try:
            while True:
                msg = self.data_queue.get_nowait()
                stale_count += 1
                print(f"[DEBUG] Drained stale message: {msg[0]}")
        except queue.Empty:
            pass
        if stale_count > 0:
            print(f"[DEBUG] Cleared {stale_count} stale message(s) from queue")

        # Get selected gestures
        gestures = [g for g, var in self.gesture_vars.items() if var.get()]
        print(f"[DEBUG] Selected gestures: {gestures}")
        if not gestures:
            print("[DEBUG] EXIT: No gestures selected")
            messagebox.showwarning("No Gestures", "Please select at least one gesture.")
            return

        # Determine data source and create appropriate stream
        self.using_real_hardware = (self.source_var.get() == "real")
        print(f"[DEBUG] using_real_hardware set to: {self.using_real_hardware}")

        if self.using_real_hardware:
            print("[DEBUG] Real hardware path")
            # Must be connected for real hardware
            print(f"[DEBUG] Checking connection: is_connected={self.is_connected}, stream exists={self.stream is not None}")
            if not self.is_connected or not self.stream:
                print("[DEBUG] EXIT: Not connected to device")
                messagebox.showerror("Not Connected", "Please connect to the ESP32 first.")
                return

            # Send start command to begin streaming
            print("[DEBUG] Calling stream.start()...")
            try:
                self.stream.start()
                print("[DEBUG] stream.start() succeeded")
            except Exception as e:
                print(f"[DEBUG] stream.start() FAILED: {e}")
                # Reset stream state if start failed
                if self.stream:
                    try:
                        print("[DEBUG] Attempting stream.stop() to reset state...")
                        self.stream.stop()  # Try to return to CONNECTED state
                        print("[DEBUG] stream.stop() succeeded")
                    except Exception as e2:
                        print(f"[DEBUG] stream.stop() FAILED: {e2}")
                messagebox.showerror("Start Error", f"Failed to start streaming:\n{e}")
                print("[DEBUG] EXIT: Stream start error")
                return
        else:
            print("[DEBUG] Simulated stream path")
            # Simulated stream (gesture-aware for realistic testing)
            self.stream = GestureAwareEMGStream(num_channels=NUM_CHANNELS, sample_rate=SAMPLING_RATE_HZ)
            print("[DEBUG] Created GestureAwareEMGStream")
            self.stream.start()  # Start the background data generation thread
            print("[DEBUG] Started simulated stream")

        # Initialize parser and windower
        self.parser = EMGParser(num_channels=NUM_CHANNELS)
        self.windower = Windower(
            window_size_ms=WINDOW_SIZE_MS,
            sample_rate=SAMPLING_RATE_HZ,
            overlap=WINDOW_OVERLAP
        )

        self.scheduler = PromptScheduler(
            gestures=gestures,
            hold_sec=self.hold_slider.get(),
            rest_sec=REST_BETWEEN_SEC,
            reps=int(self.reps_slider.get())
        )

        # Reset state
        self.collected_windows = []
        self.collected_labels = []
        self.sample_buffer = []
        print("[DEBUG] Reset collection state")

        # Mark as collecting
        self.is_collecting = True
        print("[DEBUG] Set is_collecting = True")

        # Update UI
        self.start_button.configure(text="Stop Collection", fg_color="red")
        self.save_button.configure(state="disabled")
        self.status_label.configure(text="Starting...")
        print("[DEBUG] Updated UI - button now shows 'Stop Collection'")

        # Disable source selection and connection during collection
        self.sim_radio.configure(state="disabled")
        self.real_radio.configure(state="disabled")
        if self.using_real_hardware:
            self.connect_button.configure(state="disabled")
        print("[DEBUG] Disabled source/connection controls")

        # Start collection thread
        self.collection_thread = threading.Thread(target=self.collection_loop, daemon=True)
        self.collection_thread.start()
        print("[DEBUG] Started collection thread")

        # Start UI update loop
        self.update_collection_ui()
        print("[DEBUG] start_collection() completed successfully")
        print("="*80 + "\n")

    def stop_collection(self):
        """Stop data collection."""
        print("[DEBUG] stop_collection() called")
        print(f"[DEBUG] Was collecting: {self.is_collecting}")
        self.is_collecting = False

        # Safe cleanup - stream might already be in error state
        try:
            if self.stream:
                if self.using_real_hardware:
                    print("[DEBUG] Calling stream.stop() for real hardware")
                    # Send stop command (returns to CONNECTED state)
                    self.stream.stop()
                    print("[DEBUG] stream.stop() completed")
                else:
                    print("[DEBUG] Stopping simulated stream")
                    # For simulated stream, just stop it
                    self.stream.stop()
                    self.stream = None
                    print("[DEBUG] Simulated stream stopped and cleared")
        except Exception as e:
            print(f"[DEBUG] Exception during stream cleanup: {e}")
            pass  # Ignore cleanup errors

        # Drain any pending messages from queue to prevent stale data
        try:
            while True:
                self.data_queue.get_nowait()
        except queue.Empty:
            pass

        self.start_button.configure(text="Start Collection", fg_color=["#3B8ED0", "#1F6AA5"])
        self.status_label.configure(text=f"Collected {len(self.collected_windows)} windows")
        self.prompt_label.configure(text="DONE", text_color="green")
        self.countdown_label.configure(text="")
        print("[DEBUG] UI reset - button shows 'Start Collection'")

        # Re-enable source selection and connection button
        self.sim_radio.configure(state="normal")
        self.real_radio.configure(state="normal")
        if self.using_real_hardware:
            self.connect_button.configure(state="normal")
            # Still connected, just not streaming
            if self.is_connected:
                device_name = self.stream.device_info.get('device', 'ESP32') if self.stream and self.stream.device_info else 'ESP32'
                self._update_connection_status("green", f"Connected ({device_name})")

        if self.collected_windows:
            self.save_button.configure(state="normal")

        print("[DEBUG] stop_collection() completed")
        print("="*80 + "\n")

    def collection_loop(self):
        """Background collection loop."""
        # Stream is already started (either via handshake for real HW or created for simulated)
        # Just mark as ready
        if self.using_real_hardware:
            self.data_queue.put(('connection_status', ('green', 'Streaming')))

        self.scheduler.start_session()

        last_prompt = None
        last_ui_update = time.perf_counter()
        last_plot_update = time.perf_counter()
        last_data_time = time.perf_counter()  # Track last received data for timeout detection
        sample_batch = []  # Batch samples for plotting
        timeout_warning_sent = False

        while self.is_collecting and not self.scheduler.is_session_complete():
            # Get current prompt
            prompt = self.scheduler.get_current_prompt()
            current_time = time.perf_counter()

            if prompt:
                # Update simulated stream gesture (only for GestureAwareEMGStream)
                if hasattr(self.stream, 'set_gesture'):
                    self.stream.set_gesture(prompt.gesture_name)

                # Calculate time remaining in current gesture
                elapsed_in_session = self.scheduler.get_elapsed_time()
                elapsed_in_gesture = elapsed_in_session - prompt.start_time
                time_remaining_in_gesture = prompt.duration_sec - elapsed_in_gesture

                # Find the next gesture (for "upcoming" display)
                current_prompt_idx = self.scheduler.schedule.prompts.index(prompt)
                next_gesture = None
                if current_prompt_idx + 1 < len(self.scheduler.schedule.prompts):
                    next_prompt = self.scheduler.schedule.prompts[current_prompt_idx + 1]
                    if next_prompt.gesture_name != "rest":
                        next_gesture = next_prompt.gesture_name

                # Send prompt update to UI (throttled to every 200ms for smoother text)
                if current_time - last_ui_update > 0.2:
                    # Send current gesture, countdown, and upcoming gesture
                    self.data_queue.put(('prompt_with_countdown', (
                        prompt.gesture_name,
                        time_remaining_in_gesture,
                        next_gesture
                    )))

                    # Send overall progress
                    progress = elapsed_in_session / self.scheduler.schedule.total_duration
                    self.data_queue.put(('progress', progress))
                    last_ui_update = current_time

                    last_prompt = prompt.gesture_name

            # Read and process data
            try:
                line = self.stream.readline()
            except Exception as e:
                # Only report error if we didn't intentionally stop
                if self.is_collecting:
                    self.data_queue.put(('error', f"Serial read error: {e}"))
                break

            if line:
                last_data_time = current_time  # Reset timeout counter
                timeout_warning_sent = False
                sample = self.parser.parse_line(line)
                if sample:
                    # Batch samples for plotting (don't send every single one)
                    sample_batch.append(sample.channels)

                    # Send batched samples for plotting every 50ms (20 FPS)
                    if current_time - last_plot_update > 0.05:
                        if sample_batch:
                            self.data_queue.put(('samples_batch', sample_batch))
                            sample_batch = []
                            last_plot_update = current_time

                    # Try to form a window
                    window = self.windower.add_sample(sample)
                    if window:
                        label = self.scheduler.get_label_for_time(window.start_time)
                        self.collected_windows.append(window)
                        self.collected_labels.append(label)
                        self.data_queue.put(('window_count', len(self.collected_windows)))
            else:
                # Check for data timeout (only relevant for real hardware)
                if self.using_real_hardware and (current_time - last_data_time > 3.0):
                    if not timeout_warning_sent:
                        self.data_queue.put(('warning', 'No data received - check ESP32 connection'))
                        self.data_queue.put(('connection_status', ('orange', 'No data')))
                        timeout_warning_sent = True

        # Collection complete
        self.data_queue.put(('done', None))

    def update_collection_ui(self):
        """Update UI from collection thread data."""
        needs_redraw = False

        try:
            # Process up to 10 messages per update cycle to prevent backlog
            for _ in range(10):
                msg_type, data = self.data_queue.get_nowait()

                if msg_type == 'prompt_with_countdown':
                    gesture_name, time_remaining, next_gesture = data
                    countdown_int = int(np.ceil(time_remaining))

                    if gesture_name == "rest" and next_gesture:
                        # During rest, show upcoming gesture
                        next_display = next_gesture.upper().replace("_", " ")
                        color = get_gesture_color(next_gesture)
                        display_text = f"{next_display} in {countdown_int}"
                    else:
                        # During gesture, show current gesture (user is holding it)
                        gesture_display = gesture_name.upper().replace("_", " ")
                        color = get_gesture_color(gesture_name)
                        if countdown_int > 0:
                            display_text = f"{gesture_display}  {countdown_int}"
                        else:
                            display_text = gesture_display

                    self.prompt_label.configure(text=display_text, text_color=color)

                elif msg_type == 'progress':
                    self.progress_bar.set(data)
                    remaining = self.scheduler.schedule.total_duration * (1 - data)
                    self.countdown_label.configure(text=f"Total: {remaining:.1f}s remaining")

                elif msg_type == 'samples_batch':
                    # Update plot data with batch of samples
                    for sample in data:
                        for i, val in enumerate(sample):
                            self.plot_data[i] = np.roll(self.plot_data[i], -1)
                            self.plot_data[i][-1] = val

                    # Update plot lines once per batch
                    for i in range(len(self.plot_lines)):
                        self.plot_lines[i].set_ydata(self.plot_data[i])
                    needs_redraw = True

                elif msg_type == 'window_count':
                    self.window_count_label.configure(text=f"Windows: {data}")

                elif msg_type == 'error':
                    # Show error and stop collection
                    self.status_label.configure(text=f"Error: {data}", text_color="red")
                    if self.using_real_hardware:
                        self._update_connection_status("red", "Disconnected")
                    messagebox.showerror("Collection Error", data)
                    self.stop_collection()
                    return

                elif msg_type == 'warning':
                    # Show warning but continue
                    self.status_label.configure(text=f"Warning: {data}", text_color="orange")

                elif msg_type == 'connection_status':
                    # Update connection indicator
                    color, text = data
                    self._update_connection_status(color, text)

                elif msg_type == 'done':
                    self.stop_collection()
                    return

        except queue.Empty:
            pass

        # Only redraw once per update cycle
        if needs_redraw:
            self.canvas.draw_idle()

        if self.is_collecting:
            self.after(50, self.update_collection_ui)

    def save_session(self):
        """Save the collected session."""
        if not self.collected_windows:
            messagebox.showwarning("No Data", "No data to save!")
            return

        user_id = self.user_id_entry.get() or USER_ID
        gestures = [g for g, var in self.gesture_vars.items() if var.get()]

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

        filepath = storage.save_session(self.collected_windows, self.collected_labels, metadata)

        messagebox.showinfo("Saved", f"Session saved!\n\nID: {session_id}\nWindows: {len(self.collected_windows)}")

        # Update sidebar
        app = self.winfo_toplevel()
        if isinstance(app, EMGApp):
            app.sidebar.update_status()

        # Reset for next collection
        self.collected_windows = []
        self.collected_labels = []
        self.save_button.configure(state="disabled")
        self.status_label.configure(text="Ready to collect")
        self.window_count_label.configure(text="Windows: 0")
        self.progress_bar.set(0)
        self.prompt_label.configure(text="READY", text_color="gray")

    def _on_source_change(self):
        """Show/hide port selection based on data source."""
        print("\n" + "="*80)
        print("[DEBUG] _on_source_change() called")
        print(f"[DEBUG] Before cleanup:")
        print(f"  - is_connected: {self.is_connected}")
        print(f"  - is_collecting: {self.is_collecting}")
        print(f"  - stream exists: {self.stream is not None}")
        print(f"  - source_var changing to: {self.source_var.get()}")

        # Clean up any existing connection/stream when switching modes
        if self.is_connected and self.stream:
            print("[DEBUG] Disconnecting existing stream...")
            try:
                self.stream.disconnect()
                print("[DEBUG] Stream disconnected successfully")
            except Exception as e:
                print(f"[DEBUG] Stream disconnect failed: {e}")

        self.is_connected = False
        self.stream = None
        print("[DEBUG] Cleared is_connected and stream")
        print(f"[DEBUG] NOTE: is_collecting remains: {self.is_collecting}")

        if self.source_var.get() == "real":
            print("[DEBUG] Configuring for REAL hardware mode")
            self.port_frame.pack(fill="x", pady=(5, 0))
            self._refresh_ports()
            self.connect_button.configure(text="Connect", state="normal")
            self.start_button.configure(state="disabled")  # Must connect first
            self._update_connection_status("gray", "Disconnected")
            print("[DEBUG] Start button DISABLED (must connect first)")
        else:
            print("[DEBUG] Configuring for SIMULATED mode")
            self.port_frame.pack_forget()
            self._update_connection_status("gray", "Not using hardware")
            self.connect_button.configure(state="disabled")
            self.start_button.configure(state="normal")  # Simulated mode doesn't need connect
            print("[DEBUG] Start button ENABLED (no connection needed)")
        print("="*80 + "\n")

    def _refresh_ports(self):
        """Scan and populate available serial ports."""
        ports = serial.tools.list_ports.comports()
        port_names = ["Auto-detect"] + [p.device for p in ports]

        # Update dropdown values
        self.port_dropdown.configure(values=port_names)

        # Show port info
        if ports:
            self._update_connection_status("orange", f"Found {len(ports)} port(s)")
        else:
            self._update_connection_status("red", "No ports found")

    def _get_serial_port(self):
        """Get selected port, or None for auto-detect."""
        port = self.port_var.get()
        return None if port == "Auto-detect" else port

    def _update_connection_status(self, color: str, text: str):
        """Update the connection status indicator."""
        self.connection_status.configure(text=f"● {text}", text_color=color)

    def _toggle_connection(self):
        """Connect or disconnect from ESP32."""
        if self.is_connected:
            self._disconnect_device()
        else:
            self._connect_device()

    def _connect_device(self):
        """Connect to ESP32 with handshake."""
        print("\n" + "="*80)
        print("[DEBUG] _connect_device() called")
        port = self._get_serial_port()
        print(f"[DEBUG] Port: {port}")

        try:
            # Update UI to show connecting
            self._update_connection_status("orange", "Connecting...")
            self.connect_button.configure(state="disabled")
            self.update()  # Force UI update
            print("[DEBUG] UI updated - showing 'Connecting...'")

            # Create stream and connect
            self.stream = RealSerialStream(port=port)
            print("[DEBUG] Created RealSerialStream")
            device_info = self.stream.connect(timeout=5.0)
            print(f"[DEBUG] Connection successful: {device_info}")

            # Success!
            self.is_connected = True
            print("[DEBUG] Set is_connected = True")
            self._update_connection_status("green", f"Connected ({device_info.get('device', 'ESP32')})")
            self.connect_button.configure(text="Disconnect", state="normal")
            self.start_button.configure(state="normal")
            print("[DEBUG] Start button ENABLED")
            print(f"[DEBUG] Stream state: {self.stream.state}")
            print("="*80 + "\n")

        except TimeoutError as e:
            messagebox.showerror(
                "Connection Timeout",
                f"Device did not respond within 5 seconds.\n\n"
                f"Check that:\n"
                f"• ESP32 is powered on\n"
                f"• Correct firmware is flashed\n"
                f"• USB cable is properly connected"
            )
            self._update_connection_status("red", "Timeout")
            self.connect_button.configure(state="normal")
            if self.stream:
                try:
                    self.stream.disconnect()
                except:
                    pass
                self.stream = None

        except Exception as e:
            error_msg = f"Failed to connect:\n{e}"
            if "Permission denied" in str(e) or "Resource busy" in str(e):
                error_msg += "\n\nThe port may still be in use. Wait a few seconds and try again."
            elif "FileNotFoundError" in str(type(e).__name__):
                error_msg += f"\n\nPort not found. Try refreshing the port list."

            messagebox.showerror("Connection Error", error_msg)
            self._update_connection_status("red", "Failed")
            self.connect_button.configure(state="normal")
            if self.stream:
                try:
                    self.stream.disconnect()
                except:
                    pass
                self.stream = None

    def _disconnect_device(self):
        """Disconnect from ESP32."""
        try:
            if self.stream:
                self.stream.disconnect()
                # Give OS time to release the port
                time.sleep(0.5)

            self.is_connected = False
            self.stream = None
            self._update_connection_status("gray", "Disconnected")
            self.connect_button.configure(text="Connect")
            self.start_button.configure(state="disabled")

        except Exception as e:
            messagebox.showwarning("Disconnect Warning", f"Error during disconnect: {e}")
            # Still mark as disconnected even if there was an error
            self.is_connected = False
            self.stream = None
            self._update_connection_status("gray", "Disconnected")
            self.connect_button.configure(text="Connect")
            self.start_button.configure(state="disabled")

    def on_hide(self):
        """Stop collection when leaving page."""
        if self.is_collecting:
            self.stop_collection()

    def stop(self):
        """Stop everything."""
        self.is_collecting = False
        if self.stream:
            self.stream.stop()


# =============================================================================
# INSPECT SESSIONS PAGE
# =============================================================================

class InspectPage(BasePage):
    """Page for inspecting saved sessions."""

    def __init__(self, parent):
        super().__init__(parent)

        self.create_header(
            "Inspect Sessions",
            "View saved session data and features"
        )

        # Content
        self.content = ctk.CTkFrame(self)
        self.content.grid(row=1, column=0, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_columnconfigure(1, weight=3)
        self.content.grid_rowconfigure(0, weight=1)

        # Left panel - Session list
        self.list_panel = ctk.CTkFrame(self.content)
        self.list_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        ctk.CTkLabel(self.list_panel, text="Sessions", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

        self.session_listbox = ctk.CTkScrollableFrame(self.list_panel)
        self.session_listbox.pack(fill="both", expand=True, padx=10, pady=10)

        self.refresh_button = ctk.CTkButton(self.list_panel, text="Refresh", command=self.load_sessions)
        self.refresh_button.pack(pady=10)

        # Right panel - Details
        self.details_panel = ctk.CTkFrame(self.content)
        self.details_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        self.details_label = ctk.CTkLabel(
            self.details_panel,
            text="Select a session to view details",
            font=ctk.CTkFont(size=14)
        )
        self.details_label.pack(pady=20)

        # Plot area
        self.fig = None
        self.canvas = None

        self.session_buttons = []

    def on_show(self):
        """Load sessions when page is shown."""
        self.load_sessions()

    def load_sessions(self):
        """Load and display available sessions."""
        # Clear existing buttons
        for btn in self.session_buttons:
            btn.destroy()
        self.session_buttons = []

        storage = SessionStorage()
        sessions = storage.list_sessions()

        if not sessions:
            label = ctk.CTkLabel(self.session_listbox, text="No sessions found")
            label.pack(pady=10)
            self.session_buttons.append(label)
            return

        for session_id in sessions:
            info = storage.get_session_info(session_id)
            btn_text = f"{session_id}\n{info['num_windows']} windows"

            btn = ctk.CTkButton(
                self.session_listbox,
                text=btn_text,
                font=ctk.CTkFont(size=12),
                height=60,
                anchor="w",
                command=lambda s=session_id: self.show_session(s)
            )
            btn.pack(fill="x", pady=5)
            self.session_buttons.append(btn)

    def show_session(self, session_id: str):
        """Display session details and plot."""
        storage = SessionStorage()

        try:
            X, y, label_names = storage.load_for_training(session_id)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load session: {e}")
            return

        # Clear previous plot
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        # Create info text
        info = storage.get_session_info(session_id)
        info_text = f"""Session: {session_id}
User: {info['user_id']}
Time: {info['timestamp']}
Windows: {X.shape[0]}
Samples/window: {X.shape[1]}
Channels: {X.shape[2]}
Gestures: {', '.join(label_names)}"""

        self.details_label.configure(text=info_text)

        # Create plot
        self.fig = Figure(figsize=(10, 6), dpi=100, facecolor='#2b2b2b')

        # Plot raw signal for each channel
        for ch in range(min(X.shape[2], 4)):
            ax = self.fig.add_subplot(2, 2, ch + 1)
            ax.set_facecolor('#2b2b2b')
            ax.tick_params(colors='white')

            signal = X[:, :, ch].flatten()
            signal_centered = signal - signal.mean()
            ax.plot(signal_centered[:2000], color='#00ff88', linewidth=0.5)
            ax.set_title(f'Channel {ch}', color='white', fontsize=10)
            ax.set_ylabel('Amplitude', color='white', fontsize=8)
            ax.grid(True, alpha=0.3)

            for spine in ax.spines.values():
                spine.set_color('white')

        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.details_panel)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=20)


# =============================================================================
# TRAINING PAGE
# =============================================================================

class TrainingPage(BasePage):
    """Page for training the classifier."""

    def __init__(self, parent):
        super().__init__(parent)

        self.create_header(
            "Train Classifier",
            "Train LDA model on all collected sessions"
        )

        # Content
        self.content = ctk.CTkFrame(self)
        self.content.grid(row=1, column=0, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=1)

        # Sessions info
        self.info_frame = ctk.CTkFrame(self.content)
        self.info_frame.pack(fill="x", padx=20, pady=20)

        self.sessions_label = ctk.CTkLabel(
            self.info_frame,
            text="Loading sessions...",
            font=ctk.CTkFont(size=14)
        )
        self.sessions_label.pack(pady=10)

        # Train button
        self.train_button = ctk.CTkButton(
            self.content,
            text="Train on All Sessions",
            font=ctk.CTkFont(size=18, weight="bold"),
            height=60,
            command=self.train_model
        )
        self.train_button.pack(pady=20)

        # Progress
        self.progress_bar = ctk.CTkProgressBar(self.content, width=400)
        self.progress_bar.pack(pady=10)
        self.progress_bar.set(0)

        self.status_label = ctk.CTkLabel(self.content, text="", font=ctk.CTkFont(size=12))
        self.status_label.pack()

        # Results
        self.results_frame = ctk.CTkFrame(self.content)
        self.results_frame.pack(fill="both", expand=True, padx=20, pady=20)

        self.results_text = ctk.CTkTextbox(self.results_frame, font=ctk.CTkFont(family="Courier", size=12))
        self.results_text.pack(fill="both", expand=True, padx=10, pady=10)

        self.classifier = None

    def on_show(self):
        """Update session info when shown."""
        self.update_session_info()

    def update_session_info(self):
        """Update the sessions information display."""
        storage = SessionStorage()
        sessions = storage.list_sessions()

        if not sessions:
            self.sessions_label.configure(text="No sessions found. Collect data first!")
            self.train_button.configure(state="disabled")
            return

        total_windows = 0
        info_lines = [f"Found {len(sessions)} session(s):\n"]

        for session_id in sessions:
            info = storage.get_session_info(session_id)
            info_lines.append(f"  - {session_id}: {info['num_windows']} windows")
            total_windows += info['num_windows']

        info_lines.append(f"\nTotal: {total_windows} windows")

        self.sessions_label.configure(text="\n".join(info_lines))
        self.train_button.configure(state="normal")

    def train_model(self):
        """Train the model on all sessions."""
        self.train_button.configure(state="disabled")
        self.results_text.delete("1.0", "end")
        self.progress_bar.set(0)
        self.status_label.configure(text="Loading data...")

        # Run in thread to not block UI
        thread = threading.Thread(target=self._train_thread, daemon=True)
        thread.start()

    def _train_thread(self):
        """Training thread."""
        try:
            storage = SessionStorage()

            # Load data
            self.after(0, lambda: self.status_label.configure(text="Loading all sessions..."))
            self.after(0, lambda: self.progress_bar.set(0.2))

            X, y, label_names, loaded_sessions = storage.load_all_for_training()

            self.after(0, lambda: self._log(f"Loaded {X.shape[0]} windows from {len(loaded_sessions)} sessions"))
            self.after(0, lambda: self._log(f"Labels: {label_names}\n"))

            # Train
            self.after(0, lambda: self.status_label.configure(text="Training classifier..."))
            self.after(0, lambda: self.progress_bar.set(0.5))

            self.classifier = EMGClassifier()
            self.classifier.train(X, y, label_names)

            self.after(0, lambda: self._log("Training complete!\n"))

            # Cross-validation
            self.after(0, lambda: self.status_label.configure(text="Running cross-validation..."))
            self.after(0, lambda: self.progress_bar.set(0.7))

            cv_scores = self.classifier.cross_validate(X, y, cv=5)

            self.after(0, lambda: self._log(f"Cross-validation scores: {cv_scores.round(3)}"))
            self.after(0, lambda: self._log(f"Mean accuracy: {cv_scores.mean()*100:.1f}% (+/- {cv_scores.std()*100:.1f}%)\n"))

            # Feature importance
            self.after(0, lambda: self._log("Feature importance (top 8):"))
            importance = self.classifier.get_feature_importance()
            for i, (name, score) in enumerate(list(importance.items())[:8]):
                self.after(0, lambda n=name, s=score: self._log(f"  {n}: {s:.3f}"))

            # Save model
            self.after(0, lambda: self.status_label.configure(text="Saving model..."))
            self.after(0, lambda: self.progress_bar.set(0.9))

            model_path = self.classifier.save(EMGClassifier.get_default_model_path())

            self.after(0, lambda: self._log(f"\nModel saved to: {model_path}"))
            self.after(0, lambda: self.progress_bar.set(1.0))
            self.after(0, lambda: self.status_label.configure(text="Training complete!"))

            # Update sidebar
            self.after(0, lambda: self._update_sidebar())

        except Exception as e:
            self.after(0, lambda: self._log(f"\nError: {e}"))
            self.after(0, lambda: self.status_label.configure(text="Training failed!"))

        finally:
            self.after(0, lambda: self.train_button.configure(state="normal"))

    def _log(self, text: str):
        """Add text to results."""
        self.results_text.insert("end", text + "\n")
        self.results_text.see("end")

    def _update_sidebar(self):
        """Safely update the sidebar."""
        app = self.winfo_toplevel()
        if isinstance(app, EMGApp):
            app.sidebar.update_status()


# =============================================================================
# LIVE PREDICTION PAGE
# =============================================================================

class PredictionPage(BasePage):
    """Page for live prediction demo."""

    def __init__(self, parent):
        super().__init__(parent)

        self.create_header(
            "Live Prediction",
            "Real-time gesture classification"
        )

        # State (MUST be initialized BEFORE creating UI elements)
        self.is_predicting = False
        self.is_connected = False
        self.using_real_hardware = False
        self.classifier = None
        self.smoother = None
        self.stream = None
        self.data_queue = queue.Queue()

        # Content
        self.content = ctk.CTkFrame(self)
        self.content.grid(row=1, column=0, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(1, weight=1)

        # Model status
        self.status_frame = ctk.CTkFrame(self.content)
        self.status_frame.pack(fill="x", padx=20, pady=20)

        self.model_label = ctk.CTkLabel(
            self.status_frame,
            text="Checking for saved model...",
            font=ctk.CTkFont(size=14)
        )
        self.model_label.pack(pady=10)

        # Data Source selection
        source_frame = ctk.CTkFrame(self.status_frame, fg_color="transparent")
        source_frame.pack(fill="x", pady=(10, 0))

        ctk.CTkLabel(source_frame, text="Data Source:", font=ctk.CTkFont(size=14)).pack(anchor="w")

        self.source_var = ctk.StringVar(value="simulated")

        radio_frame = ctk.CTkFrame(source_frame, fg_color="transparent")
        radio_frame.pack(fill="x", pady=(5, 0))

        self.sim_radio = ctk.CTkRadioButton(
            radio_frame, text="Simulated", variable=self.source_var, value="simulated",
            command=self._on_source_change
        )
        self.sim_radio.pack(side="left", padx=(0, 20))

        self.real_radio = ctk.CTkRadioButton(
            radio_frame, text="Real ESP32", variable=self.source_var, value="real",
            command=self._on_source_change
        )
        self.real_radio.pack(side="left")

        # Port selection (initially hidden)
        self.port_frame = ctk.CTkFrame(source_frame, fg_color="transparent")

        port_select_frame = ctk.CTkFrame(self.port_frame, fg_color="transparent")
        port_select_frame.pack(fill="x", pady=(5, 0))

        ctk.CTkLabel(port_select_frame, text="Port:").pack(side="left")

        self.port_var = ctk.StringVar(value="Auto-detect")
        self.port_dropdown = ctk.CTkOptionMenu(
            port_select_frame, variable=self.port_var,
            values=["Auto-detect"], width=150
        )
        self.port_dropdown.pack(side="left", padx=(10, 5))

        self.refresh_ports_btn = ctk.CTkButton(
            port_select_frame, text="⟳", width=30,
            command=self._refresh_ports
        )
        self.refresh_ports_btn.pack(side="left")

        # Connection status and button
        connect_frame = ctk.CTkFrame(self.port_frame, fg_color="transparent")
        connect_frame.pack(fill="x", pady=(5, 0))

        self.connect_button = ctk.CTkButton(
            connect_frame, text="Connect",
            width=100, height=28,
            command=self._toggle_connection,
            state="disabled"  # Disabled until "Real ESP32" selected
        )
        self.connect_button.pack(side="left", padx=(0, 10))

        self.connection_status = ctk.CTkLabel(
            connect_frame, text="● Disconnected",
            font=ctk.CTkFont(size=11), text_color="gray"
        )
        self.connection_status.pack(side="left")

        # Start button
        self.start_button = ctk.CTkButton(
            self.content,
            text="Start Prediction",
            font=ctk.CTkFont(size=18, weight="bold"),
            height=60,
            command=self.toggle_prediction
        )
        self.start_button.pack(pady=20)

        # Prediction display
        self.prediction_frame = ctk.CTkFrame(self.content)
        self.prediction_frame.pack(fill="both", expand=True, padx=20, pady=20)

        self.prediction_label = ctk.CTkLabel(
            self.prediction_frame,
            text="---",
            font=ctk.CTkFont(size=72, weight="bold")
        )
        self.prediction_label.pack(pady=30)

        self.confidence_bar = ctk.CTkProgressBar(self.prediction_frame, width=400, height=30)
        self.confidence_bar.pack(pady=10)
        self.confidence_bar.set(0)

        self.confidence_label = ctk.CTkLabel(
            self.prediction_frame,
            text="Confidence: ---%",
            font=ctk.CTkFont(size=18)
        )
        self.confidence_label.pack()

        # Simulated gesture indicator
        self.sim_label = ctk.CTkLabel(
            self.prediction_frame,
            text="",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        self.sim_label.pack(pady=10)

        # Smoothing info display
        self.smoothing_info_label = ctk.CTkLabel(
            self.prediction_frame,
            text="Smoothing: EMA(0.7) + Majority(5) + Debounce(3)",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.smoothing_info_label.pack()

        self.raw_label = ctk.CTkLabel(
            self.prediction_frame,
            text="",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.raw_label.pack(pady=5)

    def on_show(self):
        """Check model status when shown."""
        self.check_model()

    def check_model(self):
        """Check if a saved model exists."""
        model_path = EMGClassifier.get_default_model_path()

        if model_path.exists():
            self.model_label.configure(
                text=f"Saved model found: {model_path.name}",
                text_color="green"
            )
            self.start_button.configure(state="normal")
        else:
            self.model_label.configure(
                text="No saved model. Train a model first (Option 3).",
                text_color="orange"
            )
            self.start_button.configure(state="disabled")

    def toggle_prediction(self):
        """Start or stop prediction."""
        # Prevent rapid double-clicks from interfering
        if hasattr(self, '_toggling') and self._toggling:
            return

        self._toggling = True
        try:
            if self.is_predicting:
                self.stop_prediction()
            else:
                self.start_prediction()
        finally:
            # Reset flag after brief delay to prevent immediate re-trigger
            self.after(100, lambda: setattr(self, '_toggling', False))

    def start_prediction(self):
        """Start live prediction."""
        # CRITICAL: Drain any stale messages from previous sessions FIRST
        # This prevents old 'done' messages from stopping the new session
        try:
            while True:
                self.data_queue.get_nowait()
        except queue.Empty:
            pass

        # Load model
        try:
            self.classifier = EMGClassifier.load(EMGClassifier.get_default_model_path())
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            return

        # Determine data source
        self.using_real_hardware = (self.source_var.get() == "real")

        # For real hardware, must be connected
        if self.using_real_hardware:
            if not self.is_connected or not self.stream:
                messagebox.showerror("Not Connected", "Please connect to the ESP32 first.")
                return

            # Send start command to begin streaming
            try:
                self.stream.start()
            except Exception as e:
                messagebox.showerror("Start Error", f"Failed to start streaming:\n{e}")
                return

        # Create prediction smoother
        self.smoother = PredictionSmoother(
            label_names=self.classifier.label_names,
            probability_smoothing=0.7,   # Higher = more smoothing
            majority_vote_window=5,      # Past predictions to consider
            debounce_count=3,            # Consecutive same predictions to change output
        )

        self.is_predicting = True
        self.start_button.configure(text="Stop", fg_color="red")

        # Disable source selection and connection during prediction
        self.sim_radio.configure(state="disabled")
        self.real_radio.configure(state="disabled")
        if self.using_real_hardware:
            self.connect_button.configure(state="disabled")

        # Start prediction thread
        thread = threading.Thread(target=self._prediction_thread, daemon=True)
        thread.start()

        # Start UI update
        self.update_prediction_ui()

    def stop_prediction(self):
        """Stop live prediction."""
        self.is_predicting = False

        # Safe cleanup - stream might already be in error state
        try:
            if self.stream:
                if self.using_real_hardware:
                    # Send stop command (returns to CONNECTED state)
                    self.stream.stop()
                else:
                    # For simulated stream, just stop it
                    self.stream.stop()
                    self.stream = None
        except Exception:
            pass  # Ignore cleanup errors

        self.start_button.configure(text="Start Prediction", fg_color=["#3B8ED0", "#1F6AA5"])
        self.prediction_label.configure(text="---", text_color="white")
        self.confidence_bar.set(0)
        self.confidence_label.configure(text="Confidence: ---%")
        self.sim_label.configure(text="")
        self.raw_label.configure(text="", text_color="gray")

        # Re-enable source selection and connection button
        self.sim_radio.configure(state="normal")
        self.real_radio.configure(state="normal")
        if self.using_real_hardware:
            self.connect_button.configure(state="normal")
            # Still connected, just not streaming
            if self.is_connected:
                device_name = self.stream.device_info.get('device', 'ESP32') if self.stream and self.stream.device_info else 'ESP32'
                self._update_connection_status("green", f"Connected ({device_name})")

    def _on_source_change(self):
        """Show/hide port selection based on data source."""
        # Clean up any existing connection/stream when switching modes
        if self.is_connected and self.stream:
            try:
                self.stream.disconnect()
            except:
                pass

        self.is_connected = False
        self.stream = None

        if self.source_var.get() == "real":
            self.port_frame.pack(fill="x", pady=(5, 0))
            self._refresh_ports()
            self.connect_button.configure(text="Connect", state="normal")
            self._update_connection_status("gray", "Disconnected")
            # Start button will be enabled after connection
        else:
            self.port_frame.pack_forget()
            self._update_connection_status("gray", "Not using hardware")
            self.connect_button.configure(state="disabled")

    def _refresh_ports(self):
        """Scan and populate available serial ports."""
        ports = serial.tools.list_ports.comports()
        port_names = ["Auto-detect"] + [p.device for p in ports]
        self.port_dropdown.configure(values=port_names)

        if ports:
            self._update_connection_status("orange", f"Found {len(ports)} port(s)")
        else:
            self._update_connection_status("red", "No ports found")

    def _get_serial_port(self):
        """Get selected port, or None for auto-detect."""
        port = self.port_var.get()
        return None if port == "Auto-detect" else port

    def _update_connection_status(self, color: str, text: str):
        """Update the connection status indicator."""
        self.connection_status.configure(text=f"● {text}", text_color=color)

    def _toggle_connection(self):
        """Connect or disconnect from ESP32."""
        if self.is_connected:
            self._disconnect_device()
        else:
            self._connect_device()

    def _connect_device(self):
        """Connect to ESP32 with handshake."""
        port = self._get_serial_port()

        try:
            # Update UI to show connecting
            self._update_connection_status("orange", "Connecting...")
            self.connect_button.configure(state="disabled")
            self.update()  # Force UI update

            # Create stream and connect
            self.stream = RealSerialStream(port=port)
            device_info = self.stream.connect(timeout=5.0)

            # Success!
            self.is_connected = True
            self._update_connection_status("green", f"Connected ({device_info.get('device', 'ESP32')})")
            self.connect_button.configure(text="Disconnect", state="normal")

        except TimeoutError as e:
            messagebox.showerror(
                "Connection Timeout",
                f"Device did not respond within 5 seconds.\n\n"
                f"Check that:\n"
                f"• ESP32 is powered on\n"
                f"• Correct firmware is flashed\n"
                f"• USB cable is properly connected"
            )
            self._update_connection_status("red", "Timeout")
            self.connect_button.configure(state="normal")
            if self.stream:
                try:
                    self.stream.disconnect()
                except:
                    pass
                self.stream = None

        except Exception as e:
            error_msg = f"Failed to connect:\n{e}"
            if "Permission denied" in str(e) or "Resource busy" in str(e):
                error_msg += "\n\nThe port may still be in use. Wait a few seconds and try again."
            elif "FileNotFoundError" in str(type(e).__name__):
                error_msg += f"\n\nPort not found. Try refreshing the port list."

            messagebox.showerror("Connection Error", error_msg)
            self._update_connection_status("red", "Failed")
            self.connect_button.configure(state="normal")
            if self.stream:
                try:
                    self.stream.disconnect()
                except:
                    pass
                self.stream = None

    def _disconnect_device(self):
        """Disconnect from ESP32."""
        try:
            if self.stream:
                self.stream.disconnect()
                # Give OS time to release the port
                time.sleep(0.5)

            self.is_connected = False
            self.stream = None
            self._update_connection_status("gray", "Disconnected")
            self.connect_button.configure(text="Connect")

        except Exception as e:
            messagebox.showwarning("Disconnect Warning", f"Error during disconnect: {e}")
            # Still mark as disconnected even if there was an error
            self.is_connected = False
            self.stream = None
            self._update_connection_status("gray", "Disconnected")
            self.connect_button.configure(text="Connect")

    def _prediction_thread(self):
        """Background prediction thread."""
        # For simulated mode, create new stream
        if not self.using_real_hardware:
            self.stream = GestureAwareEMGStream(num_channels=NUM_CHANNELS, sample_rate=SAMPLING_RATE_HZ)

        # Stream is already started (either via handshake for real HW or will be started for simulated)
        parser = EMGParser(num_channels=NUM_CHANNELS)
        windower = Windower(window_size_ms=WINDOW_SIZE_MS, sample_rate=SAMPLING_RATE_HZ, overlap=0.0)

        # Simulated gesture cycling (only for simulated mode)
        gesture_cycle = ["rest", "open", "fist", "hook_em", "thumbs_up"]
        gesture_idx = 0
        gesture_duration = 2.5
        gesture_start = time.perf_counter()
        current_gesture = gesture_cycle[0]

        # Start simulated stream if needed
        if not self.using_real_hardware:
            try:
                if hasattr(self.stream, 'set_gesture'):
                    self.stream.set_gesture(current_gesture)
                self.stream.start()
            except Exception as e:
                self.data_queue.put(('error', f"Failed to start simulated stream: {e}"))
                return
        else:
            # Real hardware is already streaming
            self.data_queue.put(('connection_status', ('green', 'Streaming')))

        while self.is_predicting:
            # Change simulated gesture periodically (only for simulated mode)
            if hasattr(self.stream, 'set_gesture'):
                elapsed = time.perf_counter() - gesture_start
                if elapsed > gesture_duration:
                    gesture_idx = (gesture_idx + 1) % len(gesture_cycle)
                    gesture_start = time.perf_counter()
                    current_gesture = gesture_cycle[gesture_idx]
                    self.stream.set_gesture(current_gesture)
                    self.data_queue.put(('sim_gesture', current_gesture))

            # Read and process
            try:
                line = self.stream.readline()
            except Exception as e:
                # Only report error if we didn't intentionally stop
                if self.is_predicting:
                    self.data_queue.put(('error', f"Serial read error: {e}"))
                break

            if line:
                sample = parser.parse_line(line)
                if sample:
                    window = windower.add_sample(sample)
                    if window:
                        # Get raw prediction
                        window_data = window.to_numpy()
                        raw_label, proba = self.classifier.predict(window_data)
                        raw_confidence = max(proba) * 100

                        # Apply smoothing
                        smoothed_label, smoothed_conf, debug = self.smoother.update(raw_label, proba)
                        smoothed_confidence = smoothed_conf * 100

                        # Send both raw and smoothed to UI
                        self.data_queue.put(('prediction', (
                            smoothed_label,     # The stable output
                            smoothed_confidence,
                            raw_label,          # The raw (possibly twitchy) output
                            raw_confidence,
                        )))

        # Safe cleanup - stream might already be stopped
        try:
            if self.stream:
                self.stream.stop()
        except Exception:
            pass  # Ignore cleanup errors

    def update_prediction_ui(self):
        """Update UI from prediction thread."""
        try:
            while True:
                msg_type, data = self.data_queue.get_nowait()

                if msg_type == 'prediction':
                    smoothed_label, smoothed_conf, raw_label, raw_conf = data

                    # Display smoothed (stable) prediction
                    display_label = smoothed_label.upper().replace("_", " ")
                    color = get_gesture_color(smoothed_label)

                    self.prediction_label.configure(text=display_label, text_color=color)
                    self.confidence_bar.set(smoothed_conf / 100)
                    self.confidence_label.configure(text=f"Confidence: {smoothed_conf:.1f}%")

                    # Show raw prediction for comparison (grayed out)
                    raw_display = raw_label.upper().replace("_", " ")
                    if raw_label != smoothed_label:
                        # Raw differs from smoothed - show it was filtered
                        self.raw_label.configure(
                            text=f"Raw: {raw_display} ({raw_conf:.0f}%) → filtered",
                            text_color="orange"
                        )
                    else:
                        self.raw_label.configure(
                            text=f"Raw: {raw_display} ({raw_conf:.0f}%)",
                            text_color="gray"
                        )

                elif msg_type == 'sim_gesture':
                    self.sim_label.configure(text=f"[Simulating: {data}]")

                elif msg_type == 'error':
                    # Show error and stop prediction
                    if self.using_real_hardware:
                        self._update_connection_status("red", "Disconnected")
                    messagebox.showerror("Prediction Error", data)
                    self.stop_prediction()
                    return

                elif msg_type == 'connection_status':
                    # Update connection indicator
                    color, text = data
                    self._update_connection_status(color, text)
                    # Also update sim_label to indicate real hardware
                    if text == "Connected":
                        self.sim_label.configure(text="[Real ESP32 Hardware]")

        except queue.Empty:
            pass

        if self.is_predicting:
            self.after(50, self.update_prediction_ui)

    def on_hide(self):
        """Stop when leaving page."""
        if self.is_predicting:
            self.stop_prediction()

    def stop(self):
        """Stop everything."""
        self.is_predicting = False
        if self.stream:
            self.stream.stop()


# =============================================================================
# VISUALIZATION PAGE
# =============================================================================

class VisualizationPage(BasePage):
    """Page for LDA visualization."""

    def __init__(self, parent):
        super().__init__(parent)

        self.create_header(
            "LDA Visualization",
            "Visualize decision boundaries and feature space"
        )

        # Content
        self.content = ctk.CTkFrame(self)
        self.content.grid(row=1, column=0, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(1, weight=1)

        # Controls
        self.controls = ctk.CTkFrame(self.content)
        self.controls.pack(fill="x", padx=20, pady=20)

        self.generate_button = ctk.CTkButton(
            self.controls,
            text="Generate Visualizations",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=50,
            command=self.generate_plots
        )
        self.generate_button.pack(side="left", padx=10)

        self.status_label = ctk.CTkLabel(self.controls, text="", font=ctk.CTkFont(size=12))
        self.status_label.pack(side="left", padx=20)

        # Plot area
        self.plot_frame = ctk.CTkFrame(self.content)
        self.plot_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        self.canvas = None

    def generate_plots(self):
        """Generate LDA visualization plots."""
        storage = SessionStorage()
        sessions = storage.list_sessions()

        if not sessions:
            messagebox.showwarning("No Data", "No sessions found. Collect data first!")
            return

        self.status_label.configure(text="Loading data...")
        self.generate_button.configure(state="disabled")

        # Run in thread
        thread = threading.Thread(target=self._generate_thread, daemon=True)
        thread.start()

    def _generate_thread(self):
        """Generate plots in background."""
        try:
            storage = SessionStorage()
            X, y, label_names, _ = storage.load_all_for_training()

            self.after(0, lambda: self.status_label.configure(text="Extracting features..."))

            # Extract features and train LDA
            extractor = EMGFeatureExtractor()
            X_features = extractor.extract_features_batch(X)

            lda = LinearDiscriminantAnalysis()
            lda.fit(X_features, y)
            X_lda = lda.transform(X_features)

            self.after(0, lambda: self.status_label.configure(text="Creating plots..."))

            n_classes = len(label_names)
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_classes))

            # Create figure
            fig = Figure(figsize=(12, 5), dpi=100, facecolor='#2b2b2b')

            # Plot 1: LDA Feature Space with Decision Boundaries
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.set_facecolor('#2b2b2b')
            ax1.tick_params(colors='white')

            # Create mesh grid for decision boundaries
            if X_lda.shape[1] >= 2:
                x_min, x_max = X_lda[:, 0].min() - 1, X_lda[:, 0].max() + 1
                y_min, y_max = X_lda[:, 1].min() - 1, X_lda[:, 1].max() + 1

                xx, yy = np.meshgrid(
                    np.linspace(x_min, x_max, 200),
                    np.linspace(y_min, y_max, 200)
                )

                # Train a classifier on the 2D LDA space for visualization
                lda_2d = LinearDiscriminantAnalysis()
                lda_2d.fit(X_lda[:, :2], y)
                Z = lda_2d.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)

                # Plot decision regions (filled contours)
                ax1.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(-0.5, n_classes, 1),
                            colors=[colors[i] for i in range(n_classes)])

                # Plot decision boundaries (lines)
                ax1.contour(xx, yy, Z, colors='white', linewidths=1.5, alpha=0.8)

            # Plot data points
            for i, label in enumerate(label_names):
                mask = y == i
                ax1.scatter(
                    X_lda[mask, 0],
                    X_lda[mask, 1] if X_lda.shape[1] > 1 else np.zeros(mask.sum()),
                    c=[colors[i]], label=label, s=50, alpha=0.9,
                    edgecolors='white', linewidth=0.5
                )

            ax1.set_xlabel('LDA Component 1', color='white')
            ax1.set_ylabel('LDA Component 2', color='white')
            ax1.set_title('LDA Decision Boundaries', color='white', fontsize=14)
            ax1.legend(facecolor='#2b2b2b', labelcolor='white', loc='upper right')
            for spine in ax1.spines.values():
                spine.set_color('white')

            # Plot 2: Class distributions
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.set_facecolor('#2b2b2b')
            ax2.tick_params(colors='white')

            for i, label in enumerate(label_names):
                mask = y == i
                ax2.hist(X_lda[mask, 0], bins=20, alpha=0.6, label=label, color=colors[i])

            ax2.set_xlabel('LDA Component 1', color='white')
            ax2.set_ylabel('Count', color='white')
            ax2.set_title('Class Distributions', color='white', fontsize=14)
            ax2.legend(facecolor='#2b2b2b', labelcolor='white')
            for spine in ax2.spines.values():
                spine.set_color('white')

            fig.tight_layout()

            # Display in GUI
            self.after(0, lambda: self._show_plot(fig))
            self.after(0, lambda: self.status_label.configure(text="Done!"))

        except Exception as e:
            self.after(0, lambda: self.status_label.configure(text=f"Error: {e}"))

        finally:
            self.after(0, lambda: self.generate_button.configure(state="normal"))

    def _show_plot(self, fig):
        """Show the plot in the GUI."""
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    app = EMGApp()
    app.mainloop()
