import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import time
from dataclasses import dataclass
from typing import Tuple, List
from collections import deque
import matplotlib.colors as colors

# Use TkAgg backend for matplotlib
matplotlib.use('TkAgg')

@dataclass
class WavePacket:
    position: np.ndarray    
    momentum: np.ndarray    
    amplitude: float        
    width: float           
    phase: float           
    coherence: float       
    lifetime: float = 5.0  

class QuantumField:
    def __init__(self, size=(100, 100)):
        self.size = size
        self.state = np.zeros(size, dtype=np.complex128)
        self.probability = np.zeros(size, dtype=np.float64)
        self.phase = np.zeros(size, dtype=np.float64)
        self.wave_packets = []
        self.coherence = 1.0
        
        # Neural coupling fields
        self.neural_field = np.zeros(size, dtype=np.float64)
        self.interference_patterns = np.zeros(size, dtype=np.complex128)
        self.entanglement_strength = np.ones(size, dtype=np.float64)
        
        # Field memory with Fourier analysis
        self.state_history = deque(maxlen=100)
        self.coherence_history = deque(maxlen=100)
        self.frequency_spectrum = np.zeros((size[0], size[1]//2 + 1), dtype=np.complex128)
        
    def add_wave_packet(self, position, momentum, amplitude=1.0, width=5.0):
        packet = WavePacket(
            position=np.array(position, dtype=np.float64),
            momentum=np.array(momentum, dtype=np.float64),
            amplitude=float(amplitude),
            width=float(width),
            phase=0.0,
            coherence=1.0
        )
        self.wave_packets.append(packet)
        self._update_field_state()
    
    def evolve(self, dt=0.1):
        new_packets = []
        for packet in self.wave_packets:
            packet.position += packet.momentum * dt
            packet.phase += float(np.linalg.norm(packet.momentum)) * dt
            packet.lifetime -= dt
            if packet.lifetime > 0:
                new_packets.append(packet)
        
        self.wave_packets = new_packets
        self._update_field_state()
        
        # Update coherence history
        self.coherence_history.append(self.coherence)
    
    def _update_field_state(self):
        self.state.fill(0)
        x = np.arange(self.size[0], dtype=np.float64)
        y = np.arange(self.size[1], dtype=np.float64)
        X, Y = np.meshgrid(x, y)
        
        # Calculate quantum state
        for packet in self.wave_packets:
            dx = X - packet.position[0]
            dy = Y - packet.position[1]
            r2 = dx**2 + dy**2
            gaussian = packet.amplitude * np.exp(-r2 / (2 * packet.width**2))
            phase = packet.phase + packet.momentum[0] * dx + packet.momentum[1] * dy
            psi = gaussian * np.exp(1j * phase)
            self.state += psi * packet.coherence
        
        # Neural field coupling
        self.neural_field = np.abs(self.state)**2
        
        # Calculate interference patterns
        self.interference_patterns = np.fft.fft2(self.state)
        
        # Update entanglement strength based on neural activity
        self.entanglement_strength = 1.0 / (1.0 + np.exp(-self.neural_field))
        
        # Apply entanglement effects
        self.state *= self.entanglement_strength
        
        # Calculate observables
        self.probability = np.abs(self.state)**2
        self.phase = np.angle(self.state)
        
        # Update frequency spectrum
        self.frequency_spectrum = np.fft.rfft2(self.probability)
        
        # Store state history
        self.state_history.append(self.state.copy())
        
        # Update coherence based on interference
        self.coherence = float(np.mean(self.entanglement_strength))

class FieldVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Quantum-Neural Field Visualizer")
        self.root.geometry("1600x900")  # Increased size for better visualization
        
        # Initialize simulation parameters
        self.resolutions = {
            "100 x 100": (100, 100),
            "200 x 200": (200, 200),
            "300 x 300": (300, 300)
        }
        self.selected_resolution = tk.StringVar(value="200 x 200")
        
        # Initialize QuantumField with default resolution
        self.quantum_field = QuantumField(size=self.resolutions[self.selected_resolution.get()])
        
        # Control flags
        self.paused = False
        self.running = True
        
        self.setup_ui()
        
        # Start the update loop in a separate thread
        self.update_thread = threading.Thread(target=self.update_loop, daemon=True)
        self.update_thread.start()
    
    def setup_ui(self):
        self.fig = Figure(figsize=(16, 9), facecolor='black')
        gs = self.fig.add_gridspec(2, 3, wspace=0.4, hspace=0.3)
        
        # Quantum probability
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.prob_plot = self.ax1.imshow(
            self.quantum_field.probability,
            cmap='magma',
            aspect='auto',
            interpolation='bilinear'
        )
        self.ax1.set_title('Quantum Probability', color='white')
        self.ax1.axis('off')
        self.cbar1 = self.fig.colorbar(self.prob_plot, ax=self.ax1, fraction=0.046, pad=0.04)
        self.cbar1.ax.yaxis.set_tick_params(color='white')
        self.cbar1.outline.set_edgecolor('white')
        self.cbar1.ax.yaxis.set_tick_params(labelcolor='white') 
        
        # Neural field
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self.neural_plot = self.ax2.imshow(
            self.quantum_field.neural_field,
            cmap='viridis',
            aspect='auto',
            interpolation='bilinear'
        )
        self.ax2.set_title('Neural Field', color='white')
        self.ax2.axis('off')
        self.cbar2 = self.fig.colorbar(self.neural_plot, ax=self.ax2, fraction=0.046, pad=0.04)
        self.cbar2.ax.yaxis.set_tick_params(color='white')
        self.cbar2.outline.set_edgecolor('white')
        self.cbar2.ax.yaxis.set_tick_params(labelcolor='white') 
        
        # Interference patterns
        self.ax3 = self.fig.add_subplot(gs[0, 2])
        # To avoid log(0), add a small constant
        interference_magnitude = np.abs(self.quantum_field.interference_patterns) + 1e-10
        self.interference_plot = self.ax3.imshow(
            interference_magnitude,
            cmap='plasma',
            norm=colors.LogNorm(vmin=interference_magnitude.min(), vmax=interference_magnitude.max()),
            aspect='auto',
            interpolation='bilinear'
        )
        self.ax3.set_title('Interference Patterns', color='white')
        self.ax3.axis('off')
        self.cbar3 = self.fig.colorbar(self.interference_plot, ax=self.ax3, fraction=0.046, pad=0.04)
        self.cbar3.ax.yaxis.set_tick_params(color='white')
        self.cbar3.outline.set_edgecolor('white')
        self.cbar3.ax.yaxis.set_tick_params(labelcolor='white') 
        
        # Phase with entanglement
        self.ax4 = self.fig.add_subplot(gs[1, 0])
        self.phase_plot = self.ax4.imshow(
            self.quantum_field.phase,
            cmap='hsv',
            aspect='auto',
            interpolation='bilinear'
        )
        self.ax4.set_title('Phase & Entanglement', color='white')
        self.ax4.axis('off')
        self.cbar4 = self.fig.colorbar(self.phase_plot, ax=self.ax4, fraction=0.046, pad=0.04)
        self.cbar4.ax.yaxis.set_tick_params(color='white')
        self.cbar4.outline.set_edgecolor('white')
        self.cbar4.ax.yaxis.set_tick_params(labelcolor='white') 
        
        # Frequency spectrum
        self.ax5 = self.fig.add_subplot(gs[1, 1])
        # To avoid log(0), add a small constant
        frequency_magnitude = np.abs(self.quantum_field.frequency_spectrum) + 1e-10
        self.spectrum_plot = self.ax5.imshow(
            frequency_magnitude,
            cmap='inferno',
            norm=colors.LogNorm(vmin=frequency_magnitude.min(), vmax=frequency_magnitude.max()),
            aspect='auto',
            interpolation='bilinear'
        )
        self.ax5.set_title('Frequency Spectrum', color='white')
        self.ax5.axis('off')
        self.cbar5 = self.fig.colorbar(self.spectrum_plot, ax=self.ax5, fraction=0.046, pad=0.04)
        self.cbar5.ax.yaxis.set_tick_params(color='white')
        self.cbar5.outline.set_edgecolor('white')
        self.cbar5.ax.yaxis.set_tick_params(labelcolor='white') 
        
        # Coherence plot
        self.ax6 = self.fig.add_subplot(gs[1, 2])
        self.coherence_history = deque(maxlen=200)
        self.coherence_plot, = self.ax6.plot([], [], 'w-', alpha=0.8)
        self.ax6.set_facecolor('black')
        self.ax6.set_title('System Coherence', color='white')
        self.ax6.set_ylim(0, 1)
        self.ax6.set_xlim(0, 200)
        self.ax6.grid(True, color='gray', alpha=0.5)
        self.ax6.tick_params(axis='x', colors='white')
        self.ax6.tick_params(axis='y', colors='white')
        self.ax6.spines['bottom'].set_color('white')
        self.ax6.spines['top'].set_color('white')
        self.ax6.spines['right'].set_color('white')
        self.ax6.spines['left'].set_color('white')
        
        # Remove axes for coherence plot
        self.ax6.set_xlabel('Time Steps', color='white')
        self.ax6.set_ylabel('Coherence', color='white')
        
        self.fig.tight_layout()
        
        # Control panel
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Resolution Choice
        ttk.Label(control_frame, text="Resolution:", foreground='white').pack(side=tk.LEFT, padx=(0,5))
        resolution_menu = ttk.OptionMenu(
            control_frame,
            self.selected_resolution,
            self.selected_resolution.get(),
            *self.resolutions.keys(),
            command=self.change_resolution  # Callback on change
        )
        resolution_menu.pack(side=tk.LEFT, padx=5)
        
        # Wave Amplitude Control
        ttk.Label(control_frame, text="Wave Amplitude:", foreground='white').pack(side=tk.LEFT, padx=(20,5))
        self.amplitude_var = tk.DoubleVar(value=1.0)
        amplitude_scale = ttk.Scale(control_frame, from_=0.1, to=3.0, variable=self.amplitude_var, orient=tk.HORIZONTAL, length=200)
        amplitude_scale.pack(side=tk.LEFT, padx=5)
        
        # Wave Width Control
        ttk.Label(control_frame, text="Wave Width:", foreground='white').pack(side=tk.LEFT, padx=(20,5))
        self.width_var = tk.DoubleVar(value=5.0)
        width_scale = ttk.Scale(control_frame, from_=1.0, to=15.0, variable=self.width_var, orient=tk.HORIZONTAL, length=200)
        width_scale.pack(side=tk.LEFT, padx=5)
        
        # Toggle Entanglement
        self.entangle_var = tk.BooleanVar(value=True)
        entangle_check = ttk.Checkbutton(control_frame, text="Enable Entanglement", variable=self.entangle_var)
        entangle_check.pack(side=tk.LEFT, padx=(20,0))
        
        # Pause/Resume Button
        self.pause_button = ttk.Button(control_frame, text="Pause", command=self.toggle_pause)
        self.pause_button.pack(side=tk.LEFT, padx=(20,5))
        
        # Quit Button
        quit_button = ttk.Button(control_frame, text="Quit", command=self.on_close)
        quit_button.pack(side=tk.LEFT, padx=5)
        
        # Apply dark theme to control frame
        self.root.configure(bg='black')
        control_frame.configure(style='TFrame')
        style = ttk.Style()
        style.theme_use('default')
        style.configure('TFrame', background='black')
        style.configure('TLabel', background='black', foreground='white')
        style.configure('TCheckbutton', background='black', foreground='white')
        style.configure('TButton', background='black', foreground='white')
        
        # Canvas for plots
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events for interactivity
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_drag)
    
    def change_resolution(self, selected):
        """Handle resolution change by reinitializing QuantumField and plots."""
        # Confirm change with the user
        if messagebox.askyesno("Change Resolution", "Changing resolution will reset the simulation. Continue?"):
            # Stop the current simulation
            self.paused = True
            time.sleep(0.1)  # Brief pause to ensure thread safety
            
            # Update QuantumField
            new_size = self.resolutions[selected]
            self.quantum_field = QuantumField(size=new_size)
            
            # Update plots with new resolution
            self.update_plots(reset=True)
            
            # Reset coherence history
            self.coherence_history.clear()
            self.coherence_plot.set_data([], [])
            self.ax6.set_xlim(0, 200)
            self.ax6.set_ylim(0, 1)
            
            # Resume simulation
            self.paused = False
            self.pause_button.config(text="Pause")
    
    def toggle_pause(self):
        """Toggle the simulation between paused and running states."""
        self.paused = not self.paused
        self.pause_button.config(text="Resume" if self.paused else "Pause")
    
    def update_loop(self):
        last_time = time.time()
        while self.running:
            if not self.paused:
                current_time = time.time()
                elapsed = current_time - last_time
                if elapsed >= 1.0:
                    for _ in range(5):  # Increased number of wave packets for richer visuals
                        x = np.random.randint(0, self.quantum_field.size[0])
                        y = np.random.randint(0, self.quantum_field.size[1])
                        momentum = (np.random.uniform(-1, 1), np.random.uniform(-1, 1))
                        self.quantum_field.add_wave_packet(
                            position=(x, y),
                            momentum=momentum,
                            amplitude=self.amplitude_var.get(),
                            width=self.width_var.get()
                        )
                    last_time = current_time
                
                # Toggle entanglement based on user control
                if not self.entangle_var.get():
                    self.quantum_field.entanglement_strength = np.ones(self.quantum_field.size, dtype=np.float64)
                
                self.quantum_field.evolve(dt=0.1)
                self.update_plots()
            time.sleep(0.05)
    
    def update_plots(self, reset=False):
        # Update Quantum Probability
        self.prob_plot.set_data(self.quantum_field.probability)
        self.prob_plot.set_clim(vmin=self.quantum_field.probability.min(), vmax=self.quantum_field.probability.max())
        self.cbar1.update_normal(self.prob_plot)
        
        # Update Neural Field
        self.neural_plot.set_data(self.quantum_field.neural_field)
        self.neural_plot.set_clim(vmin=self.quantum_field.neural_field.min(), vmax=self.quantum_field.neural_field.max())
        self.cbar2.update_normal(self.neural_plot)
        
        # Update Interference Patterns
        interference_magnitude = np.abs(self.quantum_field.interference_patterns) + 1e-10
        self.interference_plot.set_data(interference_magnitude)
        self.interference_plot.set_norm(colors.LogNorm(vmin=interference_magnitude.min(), vmax=interference_magnitude.max()))
        self.interference_plot.set_clim(vmin=interference_magnitude.min(), vmax=interference_magnitude.max())
        self.cbar3.update_normal(self.interference_plot)
        
        # Update Phase with Entanglement
        self.phase_plot.set_data(self.quantum_field.phase)
        self.phase_plot.set_clim(vmin=-np.pi, vmax=np.pi)
        self.cbar4.update_normal(self.phase_plot)
        
        # Update Frequency Spectrum
        frequency_magnitude = np.abs(self.quantum_field.frequency_spectrum) + 1e-10
        self.spectrum_plot.set_data(frequency_magnitude)
        self.spectrum_plot.set_norm(colors.LogNorm(vmin=frequency_magnitude.min(), vmax=frequency_magnitude.max()))
        self.spectrum_plot.set_clim(vmin=frequency_magnitude.min(), vmax=frequency_magnitude.max())
        self.cbar5.update_normal(self.spectrum_plot)
        
        # Update Coherence Plot
        if reset:
            self.coherence_history = deque(maxlen=200)
            self.coherence_plot.set_data([], [])
        else:
            self.coherence_history.append(self.quantum_field.coherence)
            self.coherence_plot.set_data(range(len(self.coherence_history)), list(self.coherence_history))
            self.ax6.set_xlim(max(0, len(self.coherence_history)-200), len(self.coherence_history))
            self.ax6.set_ylim(0, 1)
        
        # Refresh the canvas
        self.canvas.draw_idle()
    
    def on_click(self, event):
        if event.inaxes == self.ax1:
            x = int(event.xdata)
            y = int(event.ydata)
            self.quantum_field.add_wave_packet(
                position=(x, y),
                momentum=(0, 0),
                amplitude=self.amplitude_var.get(),
                width=self.width_var.get()
            )
    
    def on_drag(self, event):
        if event.inaxes == self.ax1 and event.button == 1:
            x = int(event.xdata)
            y = int(event.ydata)
            momentum = (np.random.uniform(-1, 1), np.random.uniform(-1, 1))
            self.quantum_field.add_wave_packet(
                position=(x, y),
                momentum=momentum,
                amplitude=self.amplitude_var.get() * 0.7,  # Adjusted amplitude for smoother visuals
                width=self.width_var.get() * 0.8
            )
    
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()
    
    def on_close(self):
        """Handle the closing of the application."""
        if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
            self.running = False
            self.update_thread.join()
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FieldVisualizerApp(root)
    app.run()
