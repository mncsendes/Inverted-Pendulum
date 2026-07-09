#!/usr/bin/env python3
"""
Inverted Pendulum Simulator
Interactive PID Control + Live Animation
EE Portfolio Project - Boston University
"""

import sys
import numpy as np
from scipy.integrate import odeint
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSlider, 
                             QDoubleSpinBox, QGroupBox, QGridLayout)
from PyQt6.QtCore import QTimer, Qt
import pyqtgraph as pg

# ===== Inverted Pendulum Dynamics =====
class InvertedPendulum:
    def __init__(self):
        self.M = 1.0      # Mass of cart (kg)
        self.m = 0.3      # Mass of pendulum (kg)
        self.L = 0.5      # Length of pendulum (m)
        self.b = 0.1      # Friction coefficient (N/m/s)
        self.g = 9.81     # Gravity (m/s²)
        self.I = self.m * self.L**2 / 3  # Moment of inertia (uniform rod)
        
        # State: [x, x_dot, theta, theta_dot]
        self.state = np.array([0.0, 0.0, 0.05, 0.0])  # Small initial angle
        
        # PID gains
        self.Kp = 100.0
        self.Ki = 0.0
        self.Kd = 20.0
        self.integral = 0.0
        self.last_error = 0.0
        
        # Control force
        self.force = 0.0
        
        # Time tracking
        self.time = 0.0
        self.history = {'time': [], 'x': [], 'theta': [], 'force': []}
        
    def reset(self):
        """Reset to initial state."""
        self.state = np.array([0.0, 0.0, 0.05, 0.0])
        self.integral = 0.0
        self.last_error = 0.0
        self.force = 0.0
        self.time = 0.0
        self.history = {'time': [], 'x': [], 'theta': [], 'force': []}
        
    def dynamics(self, state, t, force):
        """State-space dynamics: dx/dt = f(x, u)"""
        x, x_dot, theta, theta_dot = state
        M, m, L, b, g, I = self.M, self.m, self.L, self.b, self.g, self.I
        
        # Calculate sin/cos
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # Equations of motion (simplified from the full derivation)
        denom = (M + m) * I + M * m * L**2 * sin_theta**2
        
        # Acceleration of cart
        x_ddot = (I * (force - b * x_dot + m * L * theta_dot**2 * sin_theta) 
                  - m * L * cos_theta * (m * g * L * sin_theta)) / denom
        
        # Angular acceleration of pendulum
        theta_ddot = ((M + m) * (m * g * L * sin_theta) 
                      - m * L * cos_theta * (force - b * x_dot + m * L * theta_dot**2 * sin_theta)) / denom
        
        return [x_dot, x_ddot, theta_dot, theta_ddot]
    
    def step(self, dt=0.01):
        """Advance the simulation by one time step."""
        # Compute control force from PID
        error = -self.state[2]  # Target angle = 0 (upright)
        self.integral += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0
        self.force = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.last_error = error
        
        # Limit force
        self.force = np.clip(self.force, -50, 50)
        
        # Integrate dynamics
        t_span = [self.time, self.time + dt]
        result = odeint(self.dynamics, self.state, t_span, args=(self.force,))
        self.state = result[-1]
        self.time += dt
        
        # Store history
        self.history['time'].append(self.time)
        self.history['x'].append(self.state[0])
        self.history['theta'].append(self.state[2] * 180 / np.pi)  # Convert to degrees
        self.history['force'].append(self.force)
        
        return self.state
    
    def set_gains(self, Kp, Ki, Kd):
        """Update PID gains."""
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd


# ===== GUI Application =====
class PendulumSimulator(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Create the physics engine
        self.pendulum = InvertedPendulum()
        self.dt = 0.01
        self.is_running = False
        self.simulation_timer = QTimer()
        self.simulation_timer.timeout.connect(self.update_simulation)
        self.simulation_timer.start(20)  # 50 FPS
        
        # Setup the UI
        self.init_ui()
        
        # Set window title
        self.setWindowTitle("🎯 Inverted Pendulum Simulator - BU EE Portfolio")
        self.setGeometry(100, 100, 1400, 800)
        
    def init_ui(self):
        """Build the user interface."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout()
        central.setLayout(main_layout)
        
        # ===== LEFT PANEL: PLOTS =====
        plot_container = QWidget()
        plot_layout = QVBoxLayout()
        plot_container.setLayout(plot_layout)
        
        # 1. Animation view (cart + pendulum)
        self.anim_view = pg.PlotWidget(title="📐 Inverted Pendulum Animation")
        self.anim_view.setXRange(-2, 2)
        self.anim_view.setYRange(-0.5, 0.5)
        self.anim_view.setAspectLocked(True)
        self.anim_view.showGrid(x=True, y=True)
        
        # Draw cart as a rectangle and pendulum as a line
        self.cart_item = self.anim_view.plot([0], [0], pen=None, symbol='s', 
                                              symbolSize=30, symbolBrush=(100, 200, 255))
        self.pendulum_line = self.anim_view.plot([0, 0], [0, 0], pen=pg.mkPen(color='r', width=3))
        
        plot_layout.addWidget(self.anim_view)
        
        # 2. State plot (angle over time)
        self.state_plot = pg.PlotWidget(title="📈 Pendulum Angle")
        self.state_plot.setLabel('left', 'Angle (degrees)')
        self.state_plot.setLabel('bottom', 'Time (s)')
        self.state_plot.setXRange(0, 5)
        self.state_plot.setYRange(-30, 30)
        self.state_plot.showGrid(x=True, y=True)
        self.angle_curve = self.state_plot.plot(pen=pg.mkPen(color='r', width=2))
        
        plot_layout.addWidget(self.state_plot)
        
        main_layout.addWidget(plot_container, stretch=2)
        
        # ===== RIGHT PANEL: CONTROLS =====
        control_container = QWidget()
        control_layout = QVBoxLayout()
        control_container.setLayout(control_layout)
        
        # PID Group
        pid_group = QGroupBox("🎛️ PID Controller Gains")
        pid_layout = QGridLayout()
        
        # Kp
        pid_layout.addWidget(QLabel("Kp:"), 0, 0)
        self.kp_slider = QSlider(Qt.Orientation.Horizontal)
        self.kp_slider.setRange(0, 500)
        self.kp_slider.setValue(100)
        self.kp_slider.valueChanged.connect(self.update_gains)
        pid_layout.addWidget(self.kp_slider, 0, 1)
        self.kp_label = QLabel("100")
        pid_layout.addWidget(self.kp_label, 0, 2)
        
        # Ki
        pid_layout.addWidget(QLabel("Ki:"), 1, 0)
        self.ki_slider = QSlider(Qt.Orientation.Horizontal)
        self.ki_slider.setRange(0, 100)
        self.ki_slider.setValue(0)
        self.ki_slider.valueChanged.connect(self.update_gains)
        pid_layout.addWidget(self.ki_slider, 1, 1)
        self.ki_label = QLabel("0")
        pid_layout.addWidget(self.ki_label, 1, 2)
        
        # Kd
        pid_layout.addWidget(QLabel("Kd:"), 2, 0)
        self.kd_slider = QSlider(Qt.Orientation.Horizontal)
        self.kd_slider.setRange(0, 100)
        self.kd_slider.setValue(20)
        self.kd_slider.valueChanged.connect(self.update_gains)
        pid_layout.addWidget(self.kd_slider, 2, 1)
        self.kd_label = QLabel("20")
        pid_layout.addWidget(self.kd_label, 2, 2)
        
        pid_group.setLayout(pid_layout)
        control_layout.addWidget(pid_group)
        
        # Control Buttons
        btn_group = QGroupBox("🎮 Simulation Controls")
        btn_layout = QVBoxLayout()
        
        self.start_btn = QPushButton("▶ Start Simulation")
        self.start_btn.clicked.connect(self.toggle_simulation)
        self.start_btn.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold; padding: 8px;")
        btn_layout.addWidget(self.start_btn)
        
        self.reset_btn = QPushButton("🔄 Reset")
        self.reset_btn.clicked.connect(self.reset_simulation)
        btn_layout.addWidget(self.reset_btn)
        
        self.disturb_btn = QPushButton("👊 Disturb (Push Pendulum)")
        self.disturb_btn.clicked.connect(self.apply_disturb)
        btn_layout.addWidget(self.disturb_btn)
        
        btn_group.setLayout(btn_layout)
        control_layout.addWidget(btn_group)
        
        # Status display
        status_group = QGroupBox("📊 System Status")
        status_layout = QGridLayout()
        
        status_layout.addWidget(QLabel("Angle:"), 0, 0)
        self.angle_display = QLabel("0.0°")
        status_layout.addWidget(self.angle_display, 0, 1)
        
        status_layout.addWidget(QLabel("Force:"), 1, 0)
        self.force_display = QLabel("0.0 N")
        status_layout.addWidget(self.force_display, 1, 1)
        
        status_layout.addWidget(QLabel("Status:"), 2, 0)
        self.status_display = QLabel("Ready")
        status_layout.addWidget(self.status_display, 2, 1)
        
        status_group.setLayout(status_layout)
        control_layout.addWidget(status_group)
        
        # Tips
        tips_label = QLabel("💡 Tip: Start with Kp≈100, Ki≈0, Kd≈20\nIncrease Kd to damp oscillations")
        tips_label.setStyleSheet("font-size: 10px; color: #888;")
        control_layout.addWidget(tips_label)
        
        control_layout.addStretch()
        main_layout.addWidget(control_container, stretch=1)
        
    def update_gains(self):
        """Update PID gains from sliders."""
        Kp = self.kp_slider.value()
        Ki = self.ki_slider.value()
        Kd = self.kd_slider.value()
        
        self.kp_label.setText(str(Kp))
        self.ki_label.setText(str(Ki))
        self.kd_label.setText(str(Kd))
        
        self.pendulum.set_gains(Kp, Ki, Kd)
        
    def update_simulation(self):
        """Called by timer to update the simulation."""
        if self.is_running:
            # Step the simulation
            self.pendulum.step(self.dt)
            
            # Update animation
            x = self.pendulum.state[0]
            theta = self.pendulum.state[2]
            
            # Draw cart
            self.cart_item.setData([x], [0])
            
            # Draw pendulum (line from cart to tip)
            L = self.pendulum.L
            tip_x = x + L * np.sin(theta)
            tip_y = -L * np.cos(theta)  # Negative because y-axis is inverted in plot
            self.pendulum_line.setData([x, tip_x], [0, tip_y])
            
            # Update state plot
            if len(self.pendulum.history['time']) > 0:
                times = self.pendulum.history['time'][-200:]  # Show last 2 seconds
                angles = self.pendulum.history['theta'][-200:]
                self.angle_curve.setData(times, angles)
                self.state_plot.setXRange(times[0], times[-1] if len(times) > 1 else 2)
            
            # Update status displays
            angle_deg = self.pendulum.state[2] * 180 / np.pi
            self.angle_display.setText(f"{angle_deg:.1f}°")
            self.force_display.setText(f"{self.pendulum.force:.1f} N")
            
            # Check if pendulum is stable (within 5 degrees for 1 second)
            if len(self.pendulum.history['theta']) > 100:
                recent = self.pendulum.history['theta'][-100:]
                if all(abs(a) < 5 for a in recent):
                    self.status_display.setText("✅ Stabilized!")
                    self.status_display.setStyleSheet("color: #2ecc71;")
                elif abs(angle_deg) > 30:
                    self.status_display.setText("❌ Unstable!")
                    self.status_display.setStyleSheet("color: #e74c3c;")
                else:
                    self.status_display.setText("🔄 Balancing...")
                    self.status_display.setStyleSheet("color: #f39c12;")
        
    def toggle_simulation(self):
        """Start or stop the simulation."""
        if not self.is_running:
            self.is_running = True
            self.start_btn.setText("⏹ Stop Simulation")
            self.start_btn.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold; padding: 8px;")
            self.status_display.setText("▶ Running")
        else:
            self.is_running = False
            self.start_btn.setText("▶ Start Simulation")
            self.start_btn.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold; padding: 8px;")
            self.status_display.setText("⏸ Paused")
    
    def reset_simulation(self):
        """Reset the pendulum to initial state."""
        self.pendulum.reset()
        self.is_running = False
        self.start_btn.setText("▶ Start Simulation")
        self.start_btn.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold; padding: 8px;")
        self.status_display.setText("🔄 Reset")
        self.angle_display.setText("0.0°")
        self.force_display.setText("0.0 N")
        
        # Clear plots
        self.cart_item.setData([0], [0])
        self.pendulum_line.setData([0, 0], [0, 0])
        self.angle_curve.setData([], [])
        
    def apply_disturb(self):
        """Apply a disturbance to the pendulum (push it)."""
        if not self.is_running:
            self.toggle_simulation()
        # Perturb the angle
        self.pendulum.state[2] += 0.2  # Add ~11 degrees disturbance
        self.pendulum.state[3] += 0.5  # Add some angular velocity
        
    def closeEvent(self, event):
        """Clean up when closing."""
        self.simulation_timer.stop()
        event.accept()


# ===== MAIN ENTRY POINT =====
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PendulumSimulator()
    window.show()
    sys.exit(app.exec())
