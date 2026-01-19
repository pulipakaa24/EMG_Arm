"""
@file serial_stream.py
@brief Real serial stream for reading EMG data from ESP32.

This module provides a serial communication interface for receiving
EMG data from the ESP32 microcontroller over USB. It implements the
same interface as SimulatedEMGStream, making it a drop-in replacement.

@section usage Usage Example
@code{.py}
    from serial_stream import RealSerialStream

    # Create stream (auto-detects port, or specify manually)
    stream = RealSerialStream(port='COM3')
    stream.start()

    # Read data (same interface as SimulatedEMGStream)
    while True:
        line = stream.readline()
        if line:
            print(line)  # "1234,512,489,501,523"

    stream.stop()
@endcode

@section format Data Format
    The ESP32 sends data as CSV lines:
    "timestamp_ms,ch0,ch1,ch2,ch3\\n"

    Example:
    "12345,512,489,501,523\\n"

@author Bucky Arm Project
"""

import serial
import serial.tools.list_ports
from typing import Optional, List


class RealSerialStream:
    """
    @brief Reads EMG data from ESP32 over USB serial.

    This class provides the same interface as SimulatedEMGStream:
        - start()    : Open serial connection
        - stop()     : Close serial connection
        - readline() : Read one line of data

    This allows it to be used as a drop-in replacement for testing
    with real hardware instead of simulated data.

    @note Requires pyserial: pip install pyserial
    """

    def __init__(self, port: str = None, baud_rate: int = 115200, timeout: float = 1.0):
        """
        @brief Initialize the serial stream.

        @param port      Serial port name (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux).
                         If None, will attempt to auto-detect the ESP32.
        @param baud_rate Communication speed in bits per second. Default 115200 matches ESP32.
        @param timeout   Read timeout in seconds. Returns None if no data within this time.
        """
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.serial: Optional[serial.Serial] = None
        self.running = False

    def start(self) -> None:
        """
        @brief Open the serial connection to the ESP32.

        If no port was specified in __init__, attempts to auto-detect
        the ESP32 by looking for common USB-UART chip identifiers.

        @throws RuntimeError If no port specified and auto-detect fails.
        @throws RuntimeError If unable to open the serial port.
        """
        # Auto-detect port if not specified
        if self.port is None:
            self.port = self._auto_detect_port()

        if self.port is None:
            raise RuntimeError(
                "No serial port specified and auto-detect failed.\n"
                "Use RealSerialStream.list_ports() to see available ports."
            )

        # Open serial connection
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=self.timeout
            )
            self.running = True

            # Clear any stale data in the buffer
            self.serial.reset_input_buffer()

            print(f"[SERIAL] Connected to {self.port} at {self.baud_rate} baud")

        except serial.SerialException as e:
            raise RuntimeError(f"Failed to open {self.port}: {e}")

    def stop(self) -> None:
        """
        @brief Close the serial connection.

        Safe to call even if not connected.
        """
        self.running = False

        if self.serial and self.serial.is_open:
            self.serial.close()
            print(f"[SERIAL] Disconnected from {self.port}")

    def readline(self) -> Optional[str]:
        """
        @brief Read one line of data from the ESP32.

        Blocks until a complete line is received or timeout occurs.
        This matches the interface of SimulatedEMGStream.readline().

        @return Line string including newline, or None if timeout/error.

        @note Lines from ESP32 are in format: "timestamp_ms,ch0,ch1,ch2,ch3\\n"
        """
        if not self.serial or not self.serial.is_open:
            return None

        try:
            line_bytes = self.serial.readline()
            if line_bytes:
                return line_bytes.decode('utf-8', errors='ignore')
            return None

        except serial.SerialException:
            return None

    def _auto_detect_port(self) -> Optional[str]:
        """
        @brief Attempt to auto-detect the ESP32 serial port.

        Looks for common USB-UART bridge chips used on ESP32 dev boards:
            - CP210x (Silicon Labs)
            - CH340 (WCH)
            - FTDI

        @return Port name if found, None otherwise.
        """
        ports = serial.tools.list_ports.comports()

        # Known USB-UART chip identifiers
        known_chips = ['cp210', 'ch340', 'ftdi', 'usb-serial', 'usb serial']

        for port in ports:
            description_lower = port.description.lower()
            if any(chip in description_lower for chip in known_chips):
                print(f"[SERIAL] Auto-detected ESP32 on {port.device}")
                return port.device

        # Fallback: use first available port
        if ports:
            print(f"[SERIAL] No ESP32 detected, using first port: {ports[0].device}")
            return ports[0].device

        return None

    @staticmethod
    def list_ports() -> List[str]:
        """
        @brief List all available serial ports on the system.

        Useful for finding the correct port name for your ESP32.

        @return List of port names (e.g., ['COM3', 'COM4'] on Windows).
        """
        ports = serial.tools.list_ports.comports()

        if not ports:
            print("No serial ports found.")
            return []

        print("\nAvailable serial ports:")
        print("-" * 50)

        for port in ports:
            print(f"  {port.device}")
            print(f"      Description: {port.description}")
            print()

        return [p.device for p in ports]


# =============================================================================
# Standalone Test
# =============================================================================

if __name__ == "__main__":
    """
    @brief Quick test to verify ESP32 serial communication.

    Run this file directly to test:
        python serial_stream.py [port]

    If port is not specified, auto-detection is attempted.
    """
    import sys

    print("=" * 50)
    print("  ESP32 Serial Stream Test")
    print("=" * 50)
    print()

    # Show available ports
    ports = RealSerialStream.list_ports()

    if not ports:
        print("No ports found. Is the ESP32 plugged in?")
        sys.exit(1)

    # Determine which port to use
    if len(sys.argv) > 1:
        port = sys.argv[1]
        print(f"Using specified port: {port}")
    else:
        port = None  # Will auto-detect
        print("No port specified, will auto-detect.")

    print()
    print("Starting stream... (Ctrl+C to stop)")
    print("-" * 50)

    # Create and start stream
    stream = RealSerialStream(port=port)

    try:
        stream.start()

        sample_count = 0

        while True:
            line = stream.readline()

            if line:
                line = line.strip()

                # Check if this is a data line (starts with digit = timestamp)
                if line and line[0].isdigit():
                    sample_count += 1

                    # Print every 500th sample to avoid flooding terminal
                    if sample_count % 500 == 0:
                        print(f"  [{sample_count:6d} samples] Latest: {line}")

                else:
                    # Print startup/info messages from ESP32
                    print(f"  {line}")

    except KeyboardInterrupt:
        print("\n")
        print("-" * 50)
        print("Stopped by user (Ctrl+C)")

    finally:
        stream.stop()
        print(f"Total samples received: {sample_count}")
