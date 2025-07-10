import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import pyaudio
from scipy.signal import get_window
from matplotlib.ticker import LogLocator, ScalarFormatter
import random
from scipy.ndimage import gaussian_filter1d # Import for smoothing

# Import pyi_splash for splash screen control
try:
    import pyi_splash
except ImportError:
    # Handle the case where pyi_splash is not available (e.g., when running directly, not as an EXE)
    pyi_splash = None

# --- Configuration Constants ---
CHUNK = 1024  # Number of audio samples per frame
FORMAT = pyaudio.paInt16  # Audio format (16-bit integers)
RATE = 44100  # Sample rate (samples per second)
# CHANNELS constant is removed; actual channels will be detected from device

# Debugging flag: Set to True to enable print statements, False to disable
DEBUG_MODE = False

# --- Audio Stream Class ---
class AudioStream:
    """
    Manages the PyAudio stream for microphone input.
    """
    def __init__(self, chunk_size=CHUNK, format=FORMAT, rate=RATE):
        self.chunk_size = chunk_size
        self.format = format
        self.rate = rate
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.current_input_device_info = None # Store info about the currently active input device

    @staticmethod
    def list_devices(input_only=False, output_only=False):
        """
        Lists all available audio input/output devices.
        Returns a dictionary mapping device names to their indices.
        """
        p = pyaudio.PyAudio()
        info = p.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        devices = {}
        for i in range(0, num_devices):
            device_info = p.get_device_info_by_host_api_device_index(0, i)
            is_input = device_info.get('maxInputChannels') > 0
            is_output = device_info.get('maxOutputChannels') > 0

            if (input_only and is_input) or \
               (output_only and is_output) or \
               (not input_only and not output_only and (is_input or is_output)):
                devices[device_info.get('name')] = i
        p.terminate()
        return devices

    def open_stream(self, input_device_index=None):
        """Opens the audio input stream with a specified device index."""
        self.close_stream() # Ensure any existing stream is closed first
        try:
            # Get device info to determine max input channels
            if input_device_index is None:
                device_info = self.p.get_default_input_device_info()
                input_device_index = device_info['index']
            else:
                device_info = self.p.get_device_info_by_index(input_device_index)

            channels_to_open = device_info.get('maxInputChannels', 1) # Default to 1 if not specified

            self.stream = self.p.open(format=self.format,
                                      channels=channels_to_open, # Use device's max channels
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer=self.chunk_size,
                                      input_device_index=input_device_index)
            self.current_input_device_info = device_info
            if DEBUG_MODE:
                print(f"Audio input stream opened successfully for device: {self.current_input_device_info['name']} with {channels_to_open} channels.")
        except Exception as e:
            if DEBUG_MODE:
                print(f"Error opening audio input stream: {e}")
            self.stream = None
            self.current_input_device_info = None

    def read_chunk(self):
        """Reads a chunk of audio data from the stream."""
        # Determine the number of channels from the stored device info
        num_channels = self.current_input_device_info.get('maxInputChannels', 1) if self.current_input_device_info else 1

        if self.stream and self.stream.is_active():
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio_array = np.frombuffer(data, dtype=np.int16)

                if num_channels == 2:
                    # Reshape interleaved stereo data into (samples, channels)
                    audio_array = audio_array.reshape(-1, 2)
                return audio_array
            except Exception as e:
                # This can happen if the device is disconnected or stream errors
                if DEBUG_MODE:
                    print(f"Error reading audio chunk: {e}")
                # Return appropriate zero array based on expected channels
                if num_channels == 2:
                    return np.zeros((self.chunk_size, 2), dtype=np.int16)
                else:
                    return np.zeros(self.chunk_size, dtype=np.int16)
        # Return appropriate zero array if stream is not active
        if num_channels == 2:
            return np.zeros((self.chunk_size, 2), dtype=np.int16)
        else:
            return np.zeros(self.chunk_size, dtype=np.int16)


    def close_stream(self):
        """Closes the audio stream."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            if DEBUG_MODE:
                print("Audio stream closed.")
        # Do not terminate self.p here, as it's used for listing devices
        # self.p.terminate() # This was causing issues if called repeatedly

# --- Spectrum Analyzer Core Logic ---
class SpectrumAnalyzer:
    """
    Handles FFT computation, windowing, and averaging.
    """
    def __init__(self, rate=RATE, chunk_size=CHUNK):
        self.rate = rate
        self.chunk_size = chunk_size
        self.window_type = 'hann'
        self.average_samples = 10 # Default average samples
        self.fft_buffer = {}  # Buffer to store recent FFT results for averaging, keyed by channel ('left', 'right', 'mono_sum')
        self.smoothing_factor = 3.0 # Default smoothing factor

        # Pre-calculate frequencies
        self.frequencies = np.fft.rfftfreq(self.chunk_size, d=1/self.rate) # Corrected d=1/self.rate

    def set_window_type(self, window_name):
        """Sets the window function type."""
        self.window_type = window_name

    def set_average_samples(self, num_samples):
        """Sets the number of samples to average."""
        self.average_samples = int(num_samples)

    def set_smoothing_factor(self, factor):
        """Sets the smoothing factor (sigma for Gaussian filter)."""
        self.smoothing_factor = max(0.0, float(factor)) # Ensure non-negative

    def compute_spectrum(self, audio_data_mono):
        """
        Takes raw mono audio data, applies window, performs FFT, and returns magnitude spectrum (in dB).
        This method now expects a 1D (mono) numpy array.
        """
        if len(audio_data_mono) != self.chunk_size:
            # Pad or truncate if audio_data size doesn't match chunk_size
            if len(audio_data_mono) < self.chunk_size:
                audio_data_mono = np.pad(audio_data_mono, (0, self.chunk_size - len(audio_data_mono)), 'constant')
            else:
                audio_data_mono = audio_data_mono[:self.chunk_size]

        # Apply window function
        window = get_window(self.window_type, self.chunk_size)
        windowed_data = audio_data_mono * window

        # Perform FFT
        yf = np.fft.rfft(windowed_data)
        magnitude = np.abs(yf) * 2 / self.chunk_size # Normalize magnitude

        # Convert to dB (avoid log of zero)
        # Add a small epsilon to avoid log(0)
        magnitude_db = 20 * np.log10(magnitude + 1e-10)

        return magnitude_db

    def get_averaged_and_smoothed_spectrum(self, current_spectrum_db, channel_key):
        """
        Adds current spectrum to buffer for a specific channel, averages, and returns smoothed spectrum.
        """
        if channel_key not in self.fft_buffer:
            self.fft_buffer[channel_key] = []

        self.fft_buffer[channel_key].append(current_spectrum_db)
        # Keep only the last 'average_samples' spectra
        if len(self.fft_buffer[channel_key]) > self.average_samples:
            self.fft_buffer[channel_key].pop(0)

        if not self.fft_buffer[channel_key]:
            return np.zeros_like(current_spectrum_db)

        # Average the spectra in the buffer
        averaged_spectrum = np.mean(self.fft_buffer[channel_key], axis=0)

        # Apply Gaussian smoothing if factor is greater than 0
        if self.smoothing_factor > 0:
            averaged_spectrum = gaussian_filter1d(averaged_spectrum, sigma=self.smoothing_factor)

        return averaged_spectrum

    def clear_fft_buffers(self):
        """Clears all stored FFT buffers."""
        self.fft_buffer = {}


# --- Main Application Class ---
class SpectrumAnalyzerApp:
    """
    Tkinter GUI for the real-time audio spectrum analyzer.
    """
    def __init__(self, master):
        self.master = master
        master.title("Real-time Audio Spectrum Analyzer")
        master.protocol("WM_DELETE_WINDOW", self.on_closing) # Handle window close event

        self.audio_stream = AudioStream()
        self.analyzer = SpectrumAnalyzer() # Analyzer now has default average_samples and smoothing_factor

        # List to store captured waveform data and their plot lines
        # Each item will be a dictionary: {'id': int, 'label': str, 'raw_audio_data': np.array, 'channel_view_at_capture': str, 'lines': [Line2D, ...], 'checkbox_var': tk.BooleanVar, 'ui_frame': ttk.Frame}
        self.captured_waveforms = []
        self.next_waveform_id = 0 # To assign unique IDs to captured waveforms

        # Define a list of distinct colors for captured waveforms
        self.capture_colors = [
            'red', 'green', 'purple', 'orange', 'cyan', 'magenta',
            'brown', 'pink', 'lime', 'teal', 'darkblue', 'gold'
        ]
        self.color_index = 0 # To cycle through colors

        # Get available audio input and output devices
        self.input_devices_map = AudioStream.list_devices(input_only=True)
        self.input_device_names = list(self.input_devices_map.keys())

        self.output_devices_map = AudioStream.list_devices(output_only=True)
        self.output_device_names = list(self.output_devices_map.keys())


        # Try to open default input device first
        default_input_device_index = None
        try:
            p = pyaudio.PyAudio()
            default_input_device_info = p.get_default_input_device_info()
            default_input_device_index = default_input_device_info['index']
            p.terminate()
        except Exception:
            # If no default input device, try the first available
            if self.input_device_names:
                default_input_device_index = self.input_devices_map[self.input_device_names[0]]

        self.audio_stream.open_stream(input_device_index=default_input_device_index)

        # Initialize channel view variables BEFORE setup_plot is called
        self.channel_view_var = tk.StringVar(self.master)
        self.channel_view_var.set("Stereo (L & R)") # Default to stereo view
        self.channel_view_options = ["Stereo (L & R)", "Left Channel", "Right Channel", "Mono (Sum)"]

        # Create the root content frame early, as it will be the parent for both plot and settings
        self.root_content_frame = ttk.Frame(self.master, padding="5") # Reduced padding
        self.root_content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Initialize the plot elements first, passing the correct master frame
        self.setup_plot(self.root_content_frame)

        # Then setup the UI elements, which now correctly references self.canvas_widget
        self.setup_ui()

        # Initialize FuncAnimation to None and create it in a dedicated method
        self.ani = None
        self._start_animation()

        # Bind the <Configure> event to the master window for resize handling
        self.master.bind("<Configure>", self.on_resize)

        # Close the PyInstaller splash screen once the main UI is set up
        if pyi_splash is not None:
            pyi_splash.close()


    def _start_animation(self):
        """Creates and starts the FuncAnimation."""
        if self.ani:
            self.ani.event_source.stop()
            # Set to None to explicitly clear the reference, allowing proper garbage collection
            self.ani = None

        self.ani = FuncAnimation(self.fig, self.update_plot, interval=50, blit=True, cache_frame_data=False)
        # Ensure background is fresh for blitting after any plot changes
        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)


    def setup_ui(self):
        """Sets up the Tkinter GUI elements."""
        # Create a frame to hold the plot on the left
        self.plot_panel_frame = ttk.Frame(self.root_content_frame)
        # Create a frame to hold the settings on the right
        self.settings_panel_frame = ttk.Frame(self.root_content_frame, padding="5", width=350)

        # Pack the settings panel first to ensure it gets its fixed width
        self.settings_panel_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=3, pady=3)
        self.settings_panel_frame.pack_propagate(False) # Prevent it from shrinking below its set width

        # Pack the plot panel to fill the remaining space
        self.plot_panel_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3, pady=3)

        # Pack the canvas widget into its plot panel frame
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)


        # Control Frame for Window Type, Average Samples, and Device Selection
        control_frame = ttk.LabelFrame(self.settings_panel_frame, text="Controls", padding="5") # Reduced padding
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=3, pady=3) # Reduced padx/pady

        # Input Device Selection
        ttk.Label(control_frame, text="Input Device:").grid(row=0, column=0, padx=2, pady=1, sticky="w") # Reduced padx/pady
        self.input_device_var = tk.StringVar(self.master)
        if self.audio_stream.current_input_device_info:
            self.input_device_var.set(self.audio_stream.current_input_device_info['name'])
        elif self.input_device_names:
            self.input_device_var.set(self.input_device_names[0]) # Set to first available if no default
        else:
            self.input_device_var.set("No Input Devices Found")

        self.input_device_menu = ttk.OptionMenu(control_frame, self.input_device_var, self.input_device_var.get(), *self.input_device_names, command=self.on_input_device_change)
        self.input_device_menu.config(width=25) # Reduced width
        self.input_device_menu.grid(row=0, column=1, padx=2, pady=1, sticky="ew") # Reduced padx/pady

        # Output Device Selection (for informational purposes/user's manual loopback setup)
        ttk.Label(control_frame, text="Output Device:").grid(row=1, column=0, padx=2, pady=1, sticky="w") # Reduced padx/pady
        self.output_device_var = tk.StringVar(self.master)
        # Try to get default output device
        default_output_device_name = "No Output Devices Found"
        try:
            p = pyaudio.PyAudio()
            default_output_device_info = p.get_default_output_device_info()
            default_output_device_name = default_output_device_info['name']
            p.terminate()
        except Exception:
            if self.output_device_names:
                default_output_device_name = self.output_device_names[0]

        self.output_device_var.set(default_output_device_name)
        self.output_device_menu = ttk.OptionMenu(control_frame, self.output_device_var, self.output_device_var.get(), *self.output_device_names, command=self.on_output_device_change)
        self.output_device_menu.config(width=25) # Reduced width
        self.output_device_menu.grid(row=1, column=1, padx=2, pady=1, sticky="ew") # Reduced padx/pady


        # Window Type Selection
        ttk.Label(control_frame, text="Window Type:").grid(row=2, column=0, padx=2, pady=1, sticky="w") # Reduced padx/pady
        self.window_var = tk.StringVar(self.master)
        self.window_var.set(self.analyzer.window_type) # Default value
        window_options = ['boxcar', 'triang', 'blackman', 'hamming', 'hann', 'bartlett', 'flattop', 'kaiser']
        self.window_menu = ttk.OptionMenu(control_frame, self.window_var, self.analyzer.window_type, *window_options, command=self.on_window_change)
        self.window_menu.grid(row=2, column=1, padx=2, pady=1, sticky="ew") # Reduced padx/pady

        # Average Samples Selection
        ttk.Label(control_frame, text="Average Samples:").grid(row=3, column=0, padx=2, pady=1, sticky="w") # Reduced padx/pady
        self.average_var = tk.IntVar(self.master)
        # Set default average samples to 10 and range from 1 to 20
        self.average_var.set(10)
        self.average_slider = ttk.Scale(control_frame, from_=1, to=20, orient="horizontal", # Range 1 to 20
                                        variable=self.average_var, command=self.on_average_change)
        self.average_slider.grid(row=3, column=1, padx=2, pady=1, sticky="ew") # Reduced padx/pady
        self.average_label = ttk.Label(control_frame, textvariable=self.average_var)
        self.average_label.grid(row=3, column=2, padx=2, pady=1, sticky="w") # Reduced padx/pady

        # Smoothing Factor Selection
        ttk.Label(control_frame, text="Smoothing Factor:").grid(row=4, column=0, padx=2, pady=1, sticky="w") # Reduced padx/pady
        self.smoothing_var = tk.DoubleVar(self.master)
        self.smoothing_var.set(3.0) # Default: 3.0
        self.smoothing_slider = ttk.Scale(control_frame, from_=0.0, to=5.0, orient="horizontal",
                                          variable=self.smoothing_var, command=self.on_smoothing_change)
        self.smoothing_slider.grid(row=4, column=1, padx=2, pady=1, sticky="ew") # Reduced padx/pady
        self.smoothing_label = ttk.Label(control_frame, textvariable=self.smoothing_var)
        self.smoothing_label.grid(row=4, column=2, padx=2, pady=1, sticky="w") # Reduced padx/pady

        # Channel View Selection
        ttk.Label(control_frame, text="Channel View:").grid(row=5, column=0, padx=2, pady=1, sticky="w") # Reduced padx/pady
        # self.channel_view_var and self.channel_view_options are now initialized in __init__
        self.channel_view_menu = ttk.OptionMenu(control_frame, self.channel_view_var, self.channel_view_var.get(), *self.channel_view_options, command=self.on_channel_view_change)
        self.channel_view_menu.config(width=15) # Reduced width
        self.channel_view_menu.grid(row=5, column=1, padx=2, pady=1, sticky="ew") # Reduced padx/pady


        # Initialize the analyzer with the new default values (these will be set by the UI vars)
        self.analyzer.set_average_samples(self.average_var.get())
        self.analyzer.set_smoothing_factor(self.smoothing_var.get())


        # Action Buttons Frame
        action_buttons_frame = ttk.Frame(control_frame)
        # Adjusted column and rowspan for action buttons
        action_buttons_frame.grid(row=0, column=2, rowspan=6, padx=5, pady=2, sticky="nsew") # Reduced padx/pady

        # Capture Button
        self.capture_button = ttk.Button(action_buttons_frame, text="Capture Spectrum", command=self.capture_spectrum)
        self.capture_button.pack(side=tk.TOP, fill=tk.X, pady=1) # Reduced pady

        # Clear All Captured Button
        self.clear_all_button = ttk.Button(action_buttons_frame, text="Clear All Captured", command=self.clear_all_captured)
        self.clear_all_button.pack(side=tk.TOP, fill=tk.X, pady=1) # Reduced pady

        control_frame.grid_columnconfigure(1, weight=1) # Make slider expand

        # Captured Waveforms List Frame
        self.captured_list_frame = ttk.LabelFrame(self.settings_panel_frame, text="Captured Waveforms", padding="5") # Reduced padding
        self.captured_list_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=3, pady=3) # Reduced padx/pady
        # Add a canvas and scrollbar for the captured waveforms list if it gets long
        self.captured_canvas = tk.Canvas(self.captured_list_frame)
        self.captured_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.captured_scrollbar = ttk.Scrollbar(self.captured_list_frame, orient="vertical", command=self.captured_canvas.yview)
        self.captured_scrollbar.pack(side=tk.RIGHT, fill="y")

        self.captured_canvas.configure(yscrollcommand=self.captured_scrollbar.set)
        self.captured_canvas.bind('<Configure>', lambda e: self.captured_canvas.configure(scrollregion = self.captured_canvas.bbox("all")))

        self.captured_inner_frame = ttk.Frame(self.captured_canvas)
        self.captured_canvas.create_window((0, 0), window=self.captured_inner_frame, anchor="nw")


    def setup_plot(self, master_frame): # Added master_frame argument
        """Sets up the Matplotlib figure and axes."""
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_title("Real-time Audio Spectrum")
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Magnitude (dB)")
        self.ax.set_xscale('log') # Logarithmic scale for frequency
        # Set initial fixed y-limits as requested: -60 dB to 80 dB
        self.ax.set_ylim(-60, 80)
        # Set initial fixed x-limits for the full audible spectrum
        self.ax.set_xlim(20, 20000)
        self.ax.grid(True, which="both", ls="-", alpha=0.7)

        # --- Custom X-axis Tick Divisions ---
        # Define major tick locations for clear labeling
        major_ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        self.ax.set_xticks(major_ticks)
        # Use LogLocator for minor ticks to automatically add subdivisions
        self.ax.xaxis.set_minor_locator(LogLocator(subs='auto'))
        # Ensure major tick labels are formatted as numbers (e.g., 1000 instead of 1e3)
        self.ax.xaxis.set_major_formatter(ScalarFormatter())
        # --- End Custom X-axis Tick Divisions ---

        # Initialize plot lines for real-time display
        # Left and Right channels for stereo, and a mono (sum) line
        self.line_left, = self.ax.plot(self.analyzer.frequencies, np.zeros_like(self.analyzer.frequencies), label="Real-time Left", color='blue', alpha=0.8)
        self.line_right, = self.ax.plot(self.analyzer.frequencies, np.zeros_like(self.analyzer.frequencies), label="Real-time Right", color='cyan', alpha=0.8)
        self.line_mono, = self.ax.plot(self.analyzer.frequencies, np.zeros_like(self.analyzer.frequencies), label="Real-time Mono (Sum)", color='blue', linestyle='-', alpha=0.8)
        self.line_mono.set_visible(False) # Mono line is hidden by default

        self.ax.legend()

        # Embed the plot in Tkinter, using the provided master_frame
        self.canvas = FigureCanvasTkAgg(self.fig, master=master_frame)
        self.canvas_widget = self.canvas.get_tk_widget()

        # Draw the canvas once to initialize the blitting background
        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)

        # Initial refresh of plot lines based on default channel view
        self._refresh_plot_lines()


    def _refresh_plot_lines(self):
        """
        Adjusts visibility of real-time plot lines and captured lines based on
        current channel view selection and input device capabilities.
        """
        current_view = self.channel_view_var.get()
        is_stereo_device = (self.audio_stream.current_input_device_info and
                            self.audio_stream.current_input_device_info.get('maxInputChannels', 1) == 2)

        # Manage real-time lines visibility
        if current_view == "Stereo (L & R)" and is_stereo_device:
            self.line_left.set_visible(True)
            self.line_right.set_visible(True)
            self.line_mono.set_visible(False)
        elif current_view == "Left Channel" and is_stereo_device:
            self.line_left.set_visible(True)
            self.line_right.set_visible(False)
            self.line_mono.set_visible(False)
        elif current_view == "Right Channel" and is_stereo_device:
            self.line_left.set_visible(False)
            self.line_right.set_visible(True)
            self.line_mono.set_visible(False)
        else: # Mono (Sum) selected, or mono device
            self.line_left.set_visible(False)
            self.line_right.set_visible(False)
            self.line_mono.set_visible(True)

        # Manage captured lines visibility (re-apply visibility based on stored channel_view_at_capture)
        for waveform_info in self.captured_waveforms:
            # Hide all lines for this captured waveform first
            for line in waveform_info['lines']:
                line.set_visible(False)

            # Then set visible based on its original capture mode and current checkbox state
            if waveform_info['checkbox_var'].get(): # Only show if checkbox is checked
                captured_view = waveform_info['channel_view_at_capture']
                captured_is_stereo = (waveform_info['raw_audio_data'].ndim == 2 and waveform_info['raw_audio_data'].shape[1] == 2)

                if captured_view == "Stereo (L & R)" and captured_is_stereo:
                    if len(waveform_info['lines']) >= 2:
                        waveform_info['lines'][0].set_visible(True) # Left
                        waveform_info['lines'][1].set_visible(True) # Right
                elif captured_view == "Left Channel" and captured_is_stereo:
                    if len(waveform_info['lines']) >= 1:
                        waveform_info['lines'][0].set_visible(True) # Left
                elif captured_view == "Right Channel" and captured_is_stereo:
                    if len(waveform_info['lines']) >= 2:
                        waveform_info['lines'][1].set_visible(True) # Right
                else: # Mono (Sum) or mono captured data
                    if len(waveform_info['lines']) >= 1:
                        waveform_info['lines'][0].set_visible(True) # Mono/Sum

        self.ax.legend()
        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)


    def update_plot(self, frame):
        """
        Callback function for FuncAnimation to update the plot in real-time.
        Uses fixed Y-axis limits and fixed X-axis limits (20Hz to 20kHz).
        Updates visibility of captured waveforms.
        Returns a list of artists to be redrawn for blitting.
        """
        # Restore the clean slate background
        self.canvas.restore_region(self.background)

        audio_data_raw = self.audio_stream.read_chunk() # This can now be 1D or 2D
        input_channels = self.audio_stream.current_input_device_info.get('maxInputChannels', 1) if self.audio_stream.current_input_device_info else 1

        artists_to_redraw = []

        # Process and update real-time lines based on current channel view
        current_view = self.channel_view_var.get()

        if input_channels == 2: # Stereo input device
            if current_view == "Stereo (L & R)":
                left_spectrum = self.analyzer.compute_spectrum(audio_data_raw[:, 0])
                right_spectrum = self.analyzer.compute_spectrum(audio_data_raw[:, 1])
                self.line_left.set_ydata(self.analyzer.get_averaged_and_smoothed_spectrum(left_spectrum, 'realtime_left'))
                self.line_right.set_ydata(self.analyzer.get_averaged_and_smoothed_spectrum(right_spectrum, 'realtime_right'))
                artists_to_redraw.extend([self.line_left, self.line_right])
            elif current_view == "Left Channel":
                left_spectrum = self.analyzer.compute_spectrum(audio_data_raw[:, 0])
                self.line_left.set_ydata(self.analyzer.get_averaged_and_smoothed_spectrum(left_spectrum, 'realtime_left'))
                artists_to_redraw.append(self.line_left)
            elif current_view == "Right Channel":
                right_spectrum = self.analyzer.compute_spectrum(audio_data_raw[:, 1])
                self.line_right.set_ydata(self.analyzer.get_averaged_and_smoothed_spectrum(right_spectrum, 'realtime_right'))
                artists_to_redraw.append(self.line_right)
            elif current_view == "Mono (Sum)":
                mono_sum_data = np.sum(audio_data_raw, axis=1) # Sum L+R for mono view
                mono_spectrum = self.analyzer.compute_spectrum(mono_sum_data)
                self.line_mono.set_ydata(self.analyzer.get_averaged_and_smoothed_spectrum(mono_spectrum, 'realtime_mono'))
                artists_to_redraw.append(self.line_mono)
        else: # Mono input device (or no audio data)
            mono_spectrum = self.analyzer.compute_spectrum(audio_data_raw)
            self.line_mono.set_ydata(self.analyzer.get_averaged_and_smoothed_spectrum(mono_spectrum, 'realtime_mono'))
            artists_to_redraw.append(self.line_mono)

        # Update visibility and collect artists for captured waveforms
        for waveform_info in self.captured_waveforms:
            if waveform_info['checkbox_var'].get():
                captured_view = waveform_info['channel_view_at_capture']
                captured_is_stereo = (waveform_info['raw_audio_data'].ndim == 2 and waveform_info['raw_audio_data'].shape[1] == 2)
                
                # Recompute spectrum for captured data based on its original capture mode
                # This ensures captured lines are consistent with how they were captured
                if captured_view == "Stereo (L & R)" and captured_is_stereo:
                    if len(waveform_info['lines']) >= 2:
                        waveform_info['lines'][0].set_ydata(self.analyzer.get_averaged_and_smoothed_spectrum(self.analyzer.compute_spectrum(waveform_info['raw_audio_data'][:, 0]), f"captured_{waveform_info['id']}_left"))
                        waveform_info['lines'][1].set_ydata(self.analyzer.get_averaged_and_smoothed_spectrum(self.analyzer.compute_spectrum(waveform_info['raw_audio_data'][:, 1]), f"captured_{waveform_info['id']}_right"))
                        artists_to_redraw.extend(waveform_info['lines'])
                elif captured_view == "Left Channel" and captured_is_stereo:
                    if len(waveform_info['lines']) >= 1:
                        waveform_info['lines'][0].set_ydata(self.analyzer.get_averaged_and_smoothed_spectrum(self.analyzer.compute_spectrum(waveform_info['raw_audio_data'][:, 0]), f"captured_{waveform_info['id']}_left"))
                        artists_to_redraw.append(waveform_info['lines'][0])
                elif captured_view == "Right Channel" and captured_is_stereo:
                    if len(waveform_info['lines']) >= 2:
                        waveform_info['lines'][1].set_ydata(self.analyzer.get_averaged_and_smoothed_spectrum(self.analyzer.compute_spectrum(waveform_info['raw_audio_data'][:, 1]), f"captured_{waveform_info['id']}_right"))
                        artists_to_redraw.append(waveform_info['lines'][1])
                else: # Mono (Sum) or mono captured data
                    if len(waveform_info['lines']) >= 1:
                        mono_data_cap = waveform_info['raw_audio_data']
                        if captured_is_stereo: # If stereo captured, sum it for mono view
                            mono_data_cap = np.sum(waveform_info['raw_audio_data'], axis=1)
                        waveform_info['lines'][0].set_ydata(self.analyzer.get_averaged_and_smoothed_spectrum(self.analyzer.compute_spectrum(mono_data_cap), f"captured_{waveform_info['id']}_mono"))
                        artists_to_redraw.append(waveform_info['lines'][0])


        # Draw all updated artists
        for artist in artists_to_redraw:
            self.ax.draw_artist(artist)

        # Blit the updated region
        self.canvas.blit(self.ax.bbox)

        return artists_to_redraw # Return the list of artists for blitting

    def on_input_device_change(self, selected_device_name):
        """
        Callback for input device selection.
        Recreates FuncAnimation, clears captured waveforms, changes device, then restarts animation.
        """
        selected_device_index = self.input_devices_map.get(selected_device_name)
        if selected_device_index is not None:
            self.audio_stream.open_stream(input_device_index=selected_device_index)
            self.analyzer.clear_fft_buffers() # Clear analyzer buffers on device change

            if self.audio_stream.current_input_device_info:
                if DEBUG_MODE:
                    print(f"Input device changed to: {self.audio_stream.current_input_device_info['name']}")
                # Adjust channel view options if the new device is mono
                if self.audio_stream.current_input_device_info.get('maxInputChannels', 1) == 1:
                    if self.channel_view_var.get() != "Mono (Sum)":
                        self.channel_view_var.set("Mono (Sum)") # Force mono view for mono device
                    self.channel_view_menu['menu'].entryconfig("Stereo (L & R)", state="disabled")
                    self.channel_view_menu['menu'].entryconfig("Left Channel", state="disabled")
                    self.channel_view_menu['menu'].entryconfig("Right Channel", state="disabled")
                else: # Stereo device, enable all options
                    self.channel_view_menu['menu'].entryconfig("Stereo (L & R)", state="normal")
                    self.channel_view_menu['menu'].entryconfig("Left Channel", state="normal")
                    self.channel_view_menu['menu'].entryconfig("Right Channel", state="normal")


                # Clear all captured waveforms when the device changes (with a slight delay)
                self.master.after(500, self.clear_all_captured)
            else:
                if DEBUG_MODE:
                    print(f"Failed to open input device: {selected_device_name}")
                # Disable channel options if no device opened
                self.channel_view_menu['menu'].entryconfig("Stereo (L & R)", state="disabled")
                self.channel_view_menu['menu'].entryconfig("Left Channel", state="disabled")
                self.channel_view_menu['menu'].entryconfig("Right Channel", state="disabled")
                self.channel_view_var.set("Mono (Sum)") # Default to mono if no device
        else:
            if DEBUG_MODE:
                print(f"Selected input device '{selected_device_name}' not found.")
            # Disable channel options if selected device not found
            self.channel_view_menu['menu'].entryconfig("Stereo (L & R)", state="disabled")
            self.channel_view_menu['menu'].entryconfig("Left Channel", state="disabled")
            self.channel_view_menu['menu'].entryconfig("Right Channel", state="disabled")
            self.channel_view_var.set("Mono (Sum)") # Default to mono if no device

        # Refresh plot lines based on new device and selected channel view
        self._refresh_plot_lines()

        # Recreate and start the animation
        self._start_animation()

    def on_output_device_change(self, selected_device_name):
        """
        Callback for output device selection.
        This primarily informs the user which output device they intend to monitor.
        It does NOT automatically route audio for loopback.
        """
        if DEBUG_MODE:
            print(f"Output device selected: {selected_device_name}")
            print("Note: For loopback analysis of this output, you may need to configure 'Stereo Mix' or use a virtual audio cable/physical connection.")

    def on_smoothing_change(self, *args):
        """Callback for smoothing factor slider."""
        new_smoothing = self.smoothing_var.get()
        self.analyzer.set_smoothing_factor(new_smoothing)
        # Clear analyzer buffers to apply smoothing effect immediately
        self.analyzer.clear_fft_buffers()
        if DEBUG_MODE:
            print(f"Smoothing factor changed to: {new_smoothing}")

    def on_channel_view_change(self, selected_view):
        """Callback for channel view selection."""
        if DEBUG_MODE:
            print(f"Channel view changed to: {selected_view}")
        self._refresh_plot_lines() # Update plot lines visibility


    def capture_spectrum(self):
        """Captures the current real-time averaged spectrum for comparison."""
        audio_data_raw = self.audio_stream.read_chunk()
        input_channels = self.audio_stream.current_input_device_info.get('maxInputChannels', 1) if self.audio_stream.current_input_device_info else 1

        if audio_data_raw is not None and len(audio_data_raw) > 0:
            captured_lines = []
            current_view = self.channel_view_var.get()
            label_base = f"Captured {self.next_waveform_id + 1}"
            selected_color = random.choice(self.capture_colors)

            if input_channels == 2 and current_view == "Stereo (L & R)":
                left_spectrum_data = self.analyzer.compute_spectrum(audio_data_raw[:, 0])
                right_spectrum_data = self.analyzer.compute_spectrum(audio_data_raw[:, 1])

                # Apply averaging and smoothing to captured data (using unique keys for buffers)
                smoothed_left_data = self.analyzer.get_averaged_and_smoothed_spectrum(left_spectrum_data, f"captured_{self.next_waveform_id}_left")
                smoothed_right_data = self.analyzer.get_averaged_and_smoothed_spectrum(right_spectrum_data, f"captured_{self.next_waveform_id}_right")

                line_left, = self.ax.plot(self.analyzer.frequencies, smoothed_left_data,
                                         label=f"{label_base} (L)", color=selected_color, linestyle='--', alpha=0.7)
                line_right, = self.ax.plot(self.analyzer.frequencies, smoothed_right_data,
                                          label=f"{label_base} (R)", color='dark' + selected_color, linestyle=':', alpha=0.7)
                captured_lines.extend([line_left, line_right])
            elif input_channels == 2 and current_view == "Left Channel":
                left_spectrum_data = self.analyzer.compute_spectrum(audio_data_raw[:, 0])
                smoothed_left_data = self.analyzer.get_averaged_and_smoothed_spectrum(left_spectrum_data, f"captured_{self.next_waveform_id}_left")
                line, = self.ax.plot(self.analyzer.frequencies, smoothed_left_data,
                                     label=f"{label_base} (L)", color=selected_color, linestyle='--', alpha=0.7)
                captured_lines.append(line)
            elif input_channels == 2 and current_view == "Right Channel":
                right_spectrum_data = self.analyzer.compute_spectrum(audio_data_raw[:, 1])
                smoothed_right_data = self.analyzer.get_averaged_and_smoothed_spectrum(right_spectrum_data, f"captured_{self.next_waveform_id}_right")
                line, = self.ax.plot(self.analyzer.frequencies, smoothed_right_data,
                                     label=f"{label_base} (R)", color=selected_color, linestyle='--', alpha=0.7)
                captured_lines.append(line)
            else: # Mono device or "Mono (Sum)" selected
                mono_data = audio_data_raw
                if input_channels == 2: # If stereo input, sum for mono capture
                    mono_data = np.sum(audio_data_raw, axis=1)
                mono_spectrum_data = self.analyzer.compute_spectrum(mono_data)
                smoothed_mono_data = self.analyzer.get_averaged_and_smoothed_spectrum(mono_spectrum_data, f"captured_{self.next_waveform_id}_mono")
                line, = self.ax.plot(self.analyzer.frequencies, smoothed_mono_data,
                                     label=f"{label_base} (Mono)", color=selected_color, linestyle='--', alpha=0.7)
                captured_lines.append(line)


            # Create a BooleanVar for the checkbox and set it to True (visible by default)
            checkbox_var = tk.BooleanVar(value=True)
            checkbox_var.trace_add("write", lambda name, index, mode, wf_id=self.next_waveform_id: self._toggle_captured_visibility(wf_id))

            # Store all relevant info
            waveform_info = {
                'id': self.next_waveform_id,
                'label': label_base,
                'raw_audio_data': audio_data_raw, # Store raw data for flexible re-plotting
                'channel_view_at_capture': current_view, # Store how it was captured
                'lines': captured_lines, # List of Line2D objects
                'checkbox_var': checkbox_var,
                'ui_frame': None, # Will be set by _add_captured_waveform_ui
                'color': selected_color # Store the base color
            }
            self.captured_waveforms.append(waveform_info)
            self._add_captured_waveform_ui(waveform_info) # Add UI elements

            self.next_waveform_id += 1
            self.ax.legend() # Update legend to include new captured line
            # Re-draw the entire canvas once after adding/removing lines to update the background for blitting
            self.canvas.draw()
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)
            if DEBUG_MODE:
                print(f"Averaged spectrum {waveform_info['label']} captured with color {selected_color}!")
        else:
            if DEBUG_MODE:
                print("No audio data to capture.")

    def _add_captured_waveform_ui(self, waveform_info):
        """Adds UI elements (checkbox and remove button) for a new captured waveform."""
        wf_frame = ttk.Frame(self.captured_inner_frame)
        wf_frame.pack(fill=tk.X, pady=1, padx=2) # Reduced pady

        checkbox = ttk.Checkbutton(wf_frame, text=waveform_info['label'], variable=waveform_info['checkbox_var'])
        checkbox.pack(side=tk.LEFT, padx=2) # Reduced padx

        remove_button = ttk.Button(wf_frame, text="X", width=2,
                                   command=lambda wf_id=waveform_info['id']: self._remove_captured_waveform(wf_id))
        remove_button.pack(side=tk.RIGHT, padx=2) # Reduced padx

        waveform_info['ui_frame'] = wf_frame # Store reference to the UI frame
        self.captured_canvas.update_idletasks() # Update canvas to recalculate scrollregion
        self.captured_canvas.config(scrollregion=self.captured_canvas.bbox("all"))


    def _toggle_captured_visibility(self, waveform_id):
        """Toggles the visibility of a specific captured waveform."""
        for waveform in self.captured_waveforms:
            if waveform['id'] == waveform_id:
                # Toggle visibility of all lines associated with this captured waveform
                for line in waveform['lines']:
                    line.set_visible(waveform['checkbox_var'].get())
                self.canvas.draw() # Force a redraw as blitting might not immediately pickable up visibility changes
                self.background = self.canvas.copy_from_bbox(self.ax.bbox) # Recapture background
                break

    def _remove_captured_waveform(self, waveform_id):
        """Removes a specific captured waveform from the list and plot."""
        waveform_to_remove = None
        for i, waveform in enumerate(self.captured_waveforms):
            if waveform['id'] == waveform_id:
                waveform_to_remove = waveform
                # Remove all lines associated with this captured waveform from the plot
                for line in waveform['lines']:
                    line.remove()
                # Remove its UI elements
                waveform['ui_frame'].destroy()
                self.captured_waveforms.pop(i)
                break

        if waveform_to_remove:
            self.ax.legend() # Update legend after removing a line
            # Re-draw the entire canvas once after adding/removing lines to update the background for blitting
            self.canvas.draw()
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)
            self.captured_canvas.update_idletasks() # Update canvas to recalculate scrollregion
            self.captured_canvas.config(scrollregion=self.captured_canvas.bbox("all"))
            if DEBUG_MODE:
                print(f"Removed captured spectrum: {waveform_to_remove['label']}")
        else:
            if DEBUG_MODE:
                print(f"Waveform with ID {waveform_id} not found.")


    def clear_all_captured(self):
        """Clears all captured spectra from the plot and UI."""
        for waveform in self.captured_waveforms:
            for line in waveform['lines']:
                line.remove() # Remove line from plot
            waveform['ui_frame'].destroy() # Destroy UI elements

        self.captured_waveforms.clear() # Clear the list
        self.next_waveform_id = 0 # Reset ID counter
        self.ax.legend() # Update legend
        # Re-draw the entire canvas once after adding/removing lines to update the background for blitting
        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.captured_canvas.update_idletasks() # Update canvas to recalculate scrollregion
        self.captured_canvas.config(scrollregion=self.captured_canvas.bbox("all"))
        if DEBUG_MODE:
            print("All captured spectra cleared.")

    def on_window_change(self, *args):
        """Callback for window type selection."""
        new_window = self.window_var.get()
        self.analyzer.set_window_type(new_window)
        if DEBUG_MODE:
            print(f"Window type changed to: {new_window}")

    def on_average_change(self, *args):
        """Callback for average samples slider."""
        new_average = self.average_var.get()
        self.analyzer.set_average_samples(new_average)
        # Clear buffer on average change to prevent old data influencing new average
        self.analyzer.clear_fft_buffers() # Clear all buffers
        if DEBUG_MODE:
            print(f"Average samples changed to: {new_average}")

    def on_channel_view_change(self, selected_view):
        """Callback for channel view selection."""
        if DEBUG_MODE:
            print(f"Channel view changed to: {selected_view}")
        self._refresh_plot_lines() # Update plot lines visibility

    def on_resize(self, event):
        """
        Handles window resize events to ensure the plot and blitting background are refreshed.
        Only re-initializes if the size actually changes to prevent unnecessary redraws.
        """
        # Get current figure size in inches
        fig_width_inches, fig_height_inches = self.fig.get_size_inches()

        # Get new canvas size in pixels
        new_canvas_width_pixels = self.canvas_widget.winfo_width()
        new_canvas_height_pixels = self.canvas_widget.winfo_height()

        # Convert canvas pixels to figure inches (assuming 100 dpi for simplicity or calculate actual dpi)
        # Matplotlib's default DPI is usually 100.
        dpi = self.fig.dpi
        new_fig_width_inches = new_canvas_width_pixels / dpi
        new_fig_height_inches = new_canvas_height_pixels / dpi

        # Check if the figure size has actually changed significantly
        if abs(new_fig_width_inches - fig_width_inches) > 0.1 or \
           abs(new_fig_height_inches - fig_height_inches) > 0.1:
            if DEBUG_MODE:
                print(f"Window resized. New dimensions: {new_canvas_width_pixels}x{new_canvas_height_pixels} pixels.")

            # Stop the current animation
            if self.ani:
                self.ani.event_source.stop()
                self.ani = None # Set to None to indicate it's stopped

            # Set the new figure size
            self.fig.set_size_inches(new_fig_width_inches, new_fig_height_inches, forward=True)

            # Redraw the canvas to apply new figure size and clear old content
            self.canvas.draw()

            # Recapture the background for blitting with the new size
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

            # Restart the animation
            self._start_animation()

            # Clear all captured waveforms a moment after resize
            self.master.after(500, self.clear_all_captured)


    def on_closing(self):
        """Handles proper shutdown when the window is closed."""
        if DEBUG_MODE:
            print("Closing application...")
        self.audio_stream.close_stream() # Close PyAudio stream
        self.audio_stream.p.terminate() # Terminate PyAudio instance on full app close
        self.master.destroy() # Destroy the Tkinter window
        plt.close(self.fig) # Close the Matplotlib figure (this will stop the animation)

# --- Main execution ---
if __name__ == "__main__":
    # Ensure PyAudio is installed: pip install PyAudio
    # Ensure Matplotlib is installed: pip install matplotlib
    # Ensure NumPy is installed: pip install numpy
    # Ensure SciPy is installed: pip install pip install scipy

    root = tk.Tk()
    app = SpectrumAnalyzerApp(root)
    root.mainloop()
