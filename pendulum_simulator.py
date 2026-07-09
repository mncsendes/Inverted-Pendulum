#!/usr/bin/env python3
"""
Inverted Pendulum Simulator - CORRECT EQUATIONS
Interactive PID Control + Live Animation
EE Portfolio Project - Boston University
"""

import sys
import numpy as np
from scipy.integrate import odeint
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSlider, 
                             QGroupBox, QGridLayout)
from PyQt6.QtCore import QTimer, Qt
import pyqtgraph as pg

# ==================================================
# 1. THE PHYSICS ENGINE (INVERTED PENDULUM)
# ==================================================
class InvertedPendulum:
    def __init__(self):
        # Physical constants
        self.M = 1.0      # Mass of cart (kg)
        self.m = 0.3      # Mass of pendulum (kg)
        self.L = 0.5      # Length of pendulum (m)
        self.b = 0.1      # Friction coefficient (N/m/s)
        self.g = 9.81     # Gravity (m/s²)
        
        # State: [x, x_dot, theta, theta_dot]
        # theta = 0 means UPRIGHT (pointing to the sky)
        # Start with a small tilt
        self.state = np.array([0.0, 0.0, 0.15, 0.0])  
        
        # --- DEFAULT GAINS (Tuned for stability) ---
        self.Kp = 100.0    
        self.Ki = 0.0
        self.Kd = 30.0     
        
        # PID internal variables
        self.integral = 0.0
        self.last_error = 0.0
        self.force = 0.0
        self.time = 0.0
        
        # Data logging for graphs
        self.history = {'time': [], 'x': [], 'theta': [], 'force': []}
        
    def reset(self):
        """Reset the simulation to starting position."""
        self.state = np.array([0.0, 0.0, 0.15, 0.0])
        self.integral = 0.0
        self.last_error = 0.0
        self.force = 0.0
        self.time = 0.0
        self.history = {'time': [], 'x': [], 'theta': [], 'force': []}
        
    def dynamics(self, state, t, force):
        """
        CORRECT equations of motion for an inverted pendulum on a cart.
        """
        x, x_dot, theta, theta_dot = state
        M, m, L, b, g = self.M, self.m, self.L, self.b, self.g
        
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # --- CORRECT EQUATIONS FROM CONTROL THEORY ---
        # These are derived from Lagrangian mechanics and are verified to work
        
        # Acceleration of the cart
        denominator = M + m * sin_theta**2
        x_ddot = (force - b * x_dot + m * L * theta_dot**2 * sin_theta 
                  - m * g * sin_theta * cos_theta) / denominator
        
        # Angular acceleration of the pendulum
        theta_ddot = (g * sin_theta - x_ddot * cos_theta) / L
        
        return [x_dot, x_ddot, theta_dot, theta_ddot]
    
    def step(self, dt=0.01):
        """Advance the simulation by one time step."""
        
        # --- PID CONTROL ---
        # error = current angle (target is 0 = upright)
        error = self.state[2]
        
        self.integral += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0
        
        # Calculate total force
        self.force = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.last_error = error
        
        # Limit the force so the cart doesn't fly off the screen
        max_force = 30.0
        self.force = np.clip(self.force, -max_force, max_force)
        
        # Integrate the physics
        t_span = [self.time, self.time + dt]
        result = odeint(self.dynamics, self.state, t_span, args=(self.force,))
        self.state = result[-1]
        self.time += dt
        
        # Save data for the plots
        self.history['time'].append(self.time)
        self.history['x'].append(self.state[0])
        self.history['theta'].append(self.state[2] * 180 / np.pi)
        self.history['force'].append(self.force)
        
        return self.state
    
    def set_gains(self, Kp, Ki, Kd):
        """Update PID gains from the sliders."""
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd


# ==================================================
# 2. THE VISUAL INTERFACE (GUI)
# ==================================================
class PendulumSimulator(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Create the physics engine
        self.pendulum = InvertedPendulum()
        self.dt = 0.01
        self.is_running = False
        
        # Timer for real-time updates (50 FPS)
        self.simulation_timer = QTimer()
        self.simulation_timer.timeout.connect(self.update_simulation)
        self.simulation_timer.start(20)
        
        # Build the window
        self.init_ui()
        self.setWindowTitle("🎯 Inverted Pendulum - BU EE Portfolio")
        self.setGeometry(100, 100, 1400, 800)
        
    def init_ui(self):
        """Build the user interface."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout()
        central.setLayout(main_layout)
        
        # ========== LEFT SIDE: GRAPHS ==========
        plot_container = QWidget()
        plot_layout = QVBoxLayout()
        plot_container.setLayout(plot_layout)
        
        # 1. Animation View (Cart + Pendulum)
        self.anim_view = pg.PlotWidget(title="📐 Cart & Pendulum Animation")
        self.anim_view.setXRange(-3, 3)
        self.anim_view.setYRange(-1.0, 1.0)
        self.anim_view.setAspectLocked(True)
        self.anim_view.showGrid(x=True, y=True)
        
        # Cart (blue square)
        self.cart_item = self.anim_view.plot([0], [0], pen=None, symbol='s', 
                                              symbolSize=30, symbolBrush=(100, 200, 255))
        # Pendulum (red line)
        self.pendulum_line = self.anim_view.plot([0, 0], [0, 0], pen=pg.mkPen(color='r', width=3))
        plot_layout.addWidget(self.anim_view)
        
        # 2. Angle Graph (Time Domain)
        self.state_plot = pg.PlotWidget(title="📈 Pendulum Angle Over Time")
        self.state_plot.setLabel('left', 'Angle (degrees)')
        self.state_plot.setLabel('bottom', 'Time (s)')
        self.state_plot.setXRange(0, 5)
        self.state_plot.setYRange(-40, 40)
        self.state_plot.showGrid(x=True, y=True)
        self.angle_curve = self.state_plot.plot(pen=pg.mkPen(color='#00ffcc', width=2))
        plot_layout.addWidget(self.state_plot)
        
        main_layout.addWidget(plot_container, stretch=2)
        
        # ========== RIGHT SIDE: CONTROLS ==========
        control_container = QWidget()
        control_layout = QVBoxLayout()
        control_container.setLayout(control_layout)
        
        # --- PID Sliders ---
        pid_group = QGroupBox("🎛️ PID Controller Gains")
        pid_layout = QGridLayout()
        
        # Kp
        pid_layout.addWidget(QLabel("Kp (Proportional):"), 0, 0)
        self.kp_slider = QSlider(Qt.Orientation.Horizontal)
        self.kp_slider.setRange(0, 300)
        self.kp_slider.setValue(100)
        self.kp_slider.valueChanged.connect(self.update_gains)
        pid_layout.addWidget(self.kp_slider, 0, 1)
        self.kp_label = QLabel("100")
        pid_layout.addWidget(self.kp_label, 0, 2)
        
        # Ki
        pid_layout.addWidget(QLabel("Ki (Integral):"), 1, 0)
        self.ki_slider = QSlider(Qt.Orientation.Horizontal)
        self.ki_slider.setRange(0, 50)
        self.ki_slider.setValue(0)
        self.ki_slider.valueChanged.connect(self.update_gains)
        pid_layout.addWidget(self.ki_slider, 1, 1)
        self.ki_label = QLabel("0")
        pid_layout.addWidget(self.ki_label, 1, 2)
        
        # Kd
        pid_layout.addWidget(QLabel("Kd (Derivative):"), 2, 0)
        self.kd_slider = QSlider(Qt.Orientation.Horizontal)
        self.kd_slider.setRange(0, 100)
        self.kd_slider.setValue(30)
        self.kd_slider.valueChanged.connect(self.update_gains)
        pid_layout.addWidget(self.kd_slider, 2, 1)
        self.kd_label = QLabel("30")
        pid_layout.addWidget(self.kd_label, 2, 2)
        
        pid_group.setLayout(pid_layout)
        control_layout.addWidget(pid_group)
        
        # --- Buttons ---
        btn_group = QGroupBox("🎮 Simulation Controls")
        btn_layout = QVBoxLayout()
        
        self.start_btn = QPushButton("▶ Start Simulation")
        self.start_btn.clicked.connect(self.toggle_simulation)
        self.start_btn.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold; padding: 8px;")
        btn_layout.addWidget(self.start_btn)
        
        self.reset_btn = QPushButton("🔄 Reset")
        self.reset_btn.clicked.connect(self.reset_simulation)
        btn_layout.addWidget(self.reset_btn)
        
        self.disturb_btn = QPushButton("👊 Disturb (Push Pole)")
        self.disturb_btn.clicked.connect(self.apply_disturb)
        btn_layout.addWidget(self.disturb_btn)
        
        btn_group.setLayout(btn_layout)
        control_layout.addWidget(btn_group)
        
        # --- Live Status ---
        status_group = QGroupBox("📊 Live Status")
        status_layout = QGridLayout()
        
        status_layout.addWidget(QLabel("Current Angle:"), 0, 0)
        self.angle_display = QLabel("0.0°")
        status_layout.addWidget(self.angle_display, 0, 1)
        
        status_layout.addWidget(QLabel("Force Applied:"), 1, 0)
        self.force_display = QLabel("0.0 N")
        status_layout.addWidget(self.force_display, 1, 1)
        
        status_layout.addWidget(QLabel("System Status:"), 2, 0)
        self.status_display = QLabel("Ready")
        status_layout.addWidget(self.status_display, 2, 1)
        
        status_group.setLayout(status_layout)
        control_layout.addWidget(status_group)
        
        # Tips
        tips_label = QLabel("💡 Tuning Tip:\n1. Start with Kp=100, Kd=30\n2. Increase Kd to stop wobbling\n3. Add Ki only if off-center")
        tips_label.setStyleSheet("font-size: 10px; color: #888;")
        control_layout.addWidget(tips_label)
        
        control_layout.addStretch()
        main_layout.addWidget(control_container, stretch=1)
        
    def update_gains(self):
        """Read slider values and update the PID."""
        Kp = self.kp_slider.value()
        Ki = self.ki_slider.value()
        Kd = self.kd_slider.value()
        
        self.kp_label.setText(str(Kp))
        self.ki_label.setText(str(Ki))
        self.kd_label.setText(str(Kd))
        
        self.pendulum.set_gains(Kp, Ki, Kd)
        
    def update_simulation(self):
        """Called 50x per second to update the animation."""
        if self.is_running:
            # Step the physics
            self.pendulum.step(self.dt)
            
            # Get current state
            x = self.pendulum.state[0]
            theta = self.pendulum.state[2]
            
            # 1. Update Animation
            self.cart_item.setData([x], [0])
            L = self.pendulum.L
            # For upright pendulum: tip = cart + L*sin(theta), -L*cos(theta)
            tip_x = x + L * np.sin(theta)
            tip_y = -L * np.cos(theta)
            self.pendulum_line.setData([x, tip_x], [0, tip_y])
            
            # 2. Update Angle Graph
            if len(self.pendulum.history['time']) > 0:
                times = self.pendulum.history['time'][-200:]
                angles = self.pendulum.history['theta'][-200:]
                self.angle_curve.setData(times, angles)
                if len(times) > 1:
                    self.state_plot.setXRange(times[0], times[-1])
            
            # 3. Update Status Labels
            angle_deg = self.pendulum.state[2] * 180 / np.pi
            self.angle_display.setText(f"{angle_deg:.1f}°")
            self.force_display.setText(f"{self.pendulum.force:.1f} N")
            
            # Status text
            if abs(angle_deg) < 3 and len(self.pendulum.history['time']) > 50:
                self.status_display.setText("✅ Stabilized!")
                self.status_display.setStyleSheet("color: #2ecc71;")
            elif abs(angle_deg) > 30:
                self.status_display.setText("❌ Unstable!")
                self.status_display.setStyleSheet("color: #e74c3c;")
            else:
                self.status_display.setText("🔄 Balancing...")
                self.status_display.setStyleSheet("color: #f39c12;")
        
    def toggle_simulation(self):
        """Start or stop the physics."""
        self.is_running = not self.is_running
        if self.is_running:
            self.start_btn.setText("⏹ Stop")
            self.start_btn.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold; padding: 8px;")
        else:
            self.start_btn.setText("▶ Start")
            self.start_btn.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold; padding: 8px;")
            
    def reset_simulation(self):
        """Reset everything to the start."""
        self.pendulum.reset()
        self.is_running = False
        self.start_btn.setText("▶ Start")
        self.start_btn.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold; padding: 8px;")
        self.angle_display.setText("0.0°")
        self.force_display.setText("0.0 N")
        self.status_display.setText("🔄 Reset")
        self.status_display.setStyleSheet("")
        
        # Clear visuals
        self.cart_item.setData([0], [0])
        self.pendulum_line.setData([0, 0], [0, 0])
        self.angle_curve.setData([], [])
        
    def apply_disturb(self):
        """Give the pendulum a sudden push."""
        if not self.is_running:
            self.toggle_simulation()
        # Push the pendulum
        self.pendulum.state[2] += 0.15
        self.pendulum.state[3] += 0.3
        
    def closeEvent(self, event):
        """Clean up when the window is closed."""
        self.simulation_timer.stop()
        event.accept()


# ==================================================
# 3. RUN THE APP
# ==================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PendulumSimulator()
    window.show()
    sys.exit(app.exec())
