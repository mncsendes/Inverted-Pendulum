#!/usr/bin/env python3
"""
Inverted Pendulum Simulator 
Pole points UP. Cart has friction to prevent drift.
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
# 1. THE PHYSICS ENGINE
# ==================================================
class InvertedPendulum:
    def __init__(self):
        # Physical constants
        self.M = 1.0      
        self.m = 0.3      
        self.L = 0.5      
        self.b = 0.1      # Friction coefficient
        self.g = 9.81     
        
        # State: [x, x_dot, theta, theta_dot]
        # theta = 0 means UPRIGHT (Pointing to the sky)
        # Start with a small tilt so we can see it correct itself
        self.state = np.array([0.0, 0.0, 0.15, 0.0])  
        
        # --- PERFECT GAINS (Tuned to stop drift) ---
        self.Kp = 120.0    
        self.Ki = 0.5      # Small integral to eliminate steady-state error
        self.Kd = 35.0     
        
        # PID variables
        self.integral = 0.0
        self.last_error = 0.0
        self.force = 0.0
        self.time = 0.0
        
        # History for plotting
        self.history = {'time': [], 'x': [], 'theta': [], 'force': []}
        
    def reset(self):
        self.state = np.array([0.0, 0.0, 0.15, 0.0])
        self.integral = 0.0
        self.last_error = 0.0
        self.force = 0.0
        self.time = 0.0
        self.history = {'time': [], 'x': [], 'theta': [], 'force': []}
        
    def dynamics(self, state, t, force):
        """Correct dynamics for cart-pole system."""
        x, x_dot, theta, theta_dot = state
        M, m, L, b, g = self.M, self.m, self.L, self.b, self.g
        
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # Equations of motion
        denominator = M + m * sin_theta**2
        x_ddot = (force - b * x_dot + m * L * theta_dot**2 * sin_theta 
                  - m * g * sin_theta * cos_theta) / denominator
        theta_ddot = (g * sin_theta - x_ddot * cos_theta) / L
        
        return [x_dot, x_ddot, theta_dot, theta_ddot]
    
    def step(self, dt=0.01):
        """Advance the simulation."""
        
        # --- PID CONTROL ---
        error = self.state[2]  # Target = 0 (upright)
        
        # Anti-windup: Limit integral term
        self.integral += error * dt
        self.integral = np.clip(self.integral, -10.0, 10.0)
        
        derivative = (error - self.last_error) / dt if dt > 0 else 0
        
        # Calculate force
        self.force = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.last_error = error
        
        # --- FIX 1: DEADBAND (Stops the cart from chasing micro-errors) ---
        # If the pendulum is almost perfectly upright, set force to 0
        if abs(error) < 0.005:  # About 0.3 degrees
            self.force = 0.0
            # Also gently slow down the cart if it's moving
            if abs(self.state[1]) < 0.01:
                self.state[1] = self.state[1] * 0.95  # Friction to stop drift
        
        # Limit the force
        max_force = 25.0
        self.force = np.clip(self.force, -max_force, max_force)
        
        # Integrate the physics
        t_span = [self.time, self.time + dt]
        result = odeint(self.dynamics, self.state, t_span, args=(self.force,))
        self.state = result[-1]
        self.time += dt
        
        # Save history
        self.history['time'].append(self.time)
        self.history['x'].append(self.state[0])
        self.history['theta'].append(self.state[2] * 180 / np.pi)
        self.history['force'].append(self.force)
        
        return self.state
    
    def set_gains(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd


# ==================================================
# 2. THE GUI
# ==================================================
class PendulumSimulator(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.pendulum = InvertedPendulum()
        self.dt = 0.01
        self.is_running = False
        
        self.simulation_timer = QTimer()
        self.simulation_timer.timeout.connect(self.update_simulation)
        self.simulation_timer.start(20)
        
        self.init_ui()
        self.setWindowTitle("🎯 Inverted Pendulum - FINAL VERSION")
        self.setGeometry(100, 100, 1400, 800)
        
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout()
        central.setLayout(main_layout)
        
        # LEFT: GRAPHS
        plot_container = QWidget()
        plot_layout = QVBoxLayout()
        plot_container.setLayout(plot_layout)
        
        # Animation View
        self.anim_view = pg.PlotWidget(title="📐 Cart & Pendulum Animation")
        self.anim_view.setXRange(-3, 3)
        self.anim_view.setYRange(-1.0, 1.0)
        self.anim_view.setAspectLocked(True)
        self.anim_view.showGrid(x=True, y=True)
        self.anim_view.setLabel('left', 'Height', units='m')
        self.anim_view.setLabel('bottom', 'Position', units='m')
        
        # Draw cart
        self.cart_item = self.anim_view.plot([0], [0], pen=None, symbol='s', 
                                              symbolSize=35, symbolBrush=(100, 200, 255))
        # Draw pendulum (RED line) - FIXED TO POINT UP
        self.pendulum_line = self.anim_view.plot([0, 0], [0, 0], pen=pg.mkPen(color='r', width=4))
        
        # Draw a "ground line"
        self.ground_line = self.anim_view.plot([-3, 3], [0, 0], pen=pg.mkPen(color='gray', width=1, style=Qt.PenStyle.DashLine))
        
        plot_layout.addWidget(self.anim_view)
        
        # Angle Graph
        self.state_plot = pg.PlotWidget(title="📈 Pendulum Angle Over Time")
        self.state_plot.setLabel('left', 'Angle (degrees)')
        self.state_plot.setLabel('bottom', 'Time (s)')
        self.state_plot.setXRange(0, 5)
        self.state_plot.setYRange(-40, 40)
        self.state_plot.showGrid(x=True, y=True)
        self.angle_curve = self.state_plot.plot(pen=pg.mkPen(color='#00ffcc', width=2))
        plot_layout.addWidget(self.state_plot)
        
        main_layout.addWidget(plot_container, stretch=2)
        
        # RIGHT: CONTROLS
        control_container = QWidget()
        control_layout = QVBoxLayout()
        control_container.setLayout(control_layout)
        
        # PID Sliders
        pid_group = QGroupBox("🎛️ PID Controller Gains")
        pid_layout = QGridLayout()
        
        pid_layout.addWidget(QLabel("Kp (Proportional):"), 0, 0)
        self.kp_slider = QSlider(Qt.Orientation.Horizontal)
        self.kp_slider.setRange(0, 300)
        self.kp_slider.setValue(120)
        self.kp_slider.valueChanged.connect(self.update_gains)
        pid_layout.addWidget(self.kp_slider, 0, 1)
        self.kp_label = QLabel("120")
        pid_layout.addWidget(self.kp_label, 0, 2)
        
        pid_layout.addWidget(QLabel("Ki (Integral):"), 1, 0)
        self.ki_slider = QSlider(Qt.Orientation.Horizontal)
        self.ki_slider.setRange(0, 20)
        self.ki_slider.setValue(1)
        self.ki_slider.valueChanged.connect(self.update_gains)
        pid_layout.addWidget(self.ki_slider, 1, 1)
        self.ki_label = QLabel("1")
        pid_layout.addWidget(self.ki_label, 1, 2)
        
        pid_layout.addWidget(QLabel("Kd (Derivative):"), 2, 0)
        self.kd_slider = QSlider(Qt.Orientation.Horizontal)
        self.kd_slider.setRange(0, 100)
        self.kd_slider.setValue(35)
        self.kd_slider.valueChanged.connect(self.update_gains)
        pid_layout.addWidget(self.kd_slider, 2, 1)
        self.kd_label = QLabel("35")
        pid_layout.addWidget(self.kd_label, 2, 2)
        
        pid_group.setLayout(pid_layout)
        control_layout.addWidget(pid_group)
        
        # Buttons
        btn_group = QGroupBox("🎮 Controls")
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
        
        # Status
        status_group = QGroupBox("📊 Live Status")
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
        
        tips_label = QLabel("💡 This version has a deadband.\nThe cart will STOP moving when balanced.")
        tips_label.setStyleSheet("font-size: 10px; color: #888;")
        control_layout.addWidget(tips_label)
        
        control_layout.addStretch()
        main_layout.addWidget(control_container, stretch=1)
        
    def update_gains(self):
        Kp = self.kp_slider.value()
        Ki = self.ki_slider.value()
        Kd = self.kd_slider.value()
        self.kp_label.setText(str(Kp))
        self.ki_label.setText(str(Ki))
        self.kd_label.setText(str(Kd))
        self.pendulum.set_gains(Kp, Ki, Kd)
        
    def update_simulation(self):
        if self.is_running:
            self.pendulum.step(self.dt)
            
            x = self.pendulum.state[0]
            theta = self.pendulum.state[2]
            
            # --- FIX 2: DRAW THE PENDULUM POINTING UP ---
            L = self.pendulum.L
            # When theta=0, the tip of the pendulum should be directly ABOVE the cart
            # So: tip_x = cart_x + L * sin(theta), tip_y = L * cos(theta)
            tip_x = x + L * np.sin(theta)
            tip_y = L * np.cos(theta)  # POSITIVE Y (up) because theta=0 is upright!
            
            self.cart_item.setData([x], [0])
            self.pendulum_line.setData([x, tip_x], [0, tip_y])
            
            # Update graph
            if len(self.pendulum.history['time']) > 0:
                times = self.pendulum.history['time'][-200:]
                angles = self.pendulum.history['theta'][-200:]
                self.angle_curve.setData(times, angles)
                if len(times) > 1:
                    self.state_plot.setXRange(times[0], times[-1])
            
            # Status
            angle_deg = self.pendulum.state[2] * 180 / np.pi
            self.angle_display.setText(f"{angle_deg:.1f}°")
            self.force_display.setText(f"{self.pendulum.force:.1f} N")
            
            if abs(angle_deg) < 2 and len(self.pendulum.history['time']) > 30:
                self.status_display.setText("✅ Stabilized! (Cart Stopped)")
                self.status_display.setStyleSheet("color: #2ecc71;")
            elif abs(angle_deg) > 30:
                self.status_display.setText("❌ Unstable!")
                self.status_display.setStyleSheet("color: #e74c3c;")
            else:
                self.status_display.setText("🔄 Balancing...")
                self.status_display.setStyleSheet("color: #f39c12;")
        
    def toggle_simulation(self):
        self.is_running = not self.is_running
        if self.is_running:
            self.start_btn.setText("⏹ Stop")
            self.start_btn.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold; padding: 8px;")
        else:
            self.start_btn.setText("▶ Start")
            self.start_btn.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold; padding: 8px;")
            
    def reset_simulation(self):
        self.pendulum.reset()
        self.is_running = False
        self.start_btn.setText("▶ Start")
        self.start_btn.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold; padding: 8px;")
        self.angle_display.setText("0.0°")
        self.force_display.setText("0.0 N")
        self.status_display.setText("🔄 Reset")
        self.cart_item.setData([0], [0])
        self.pendulum_line.setData([0, 0], [0, 0])
        self.angle_curve.setData([], [])
        
    def apply_disturb(self):
        if not self.is_running:
            self.toggle_simulation()
        self.pendulum.state[2] += 0.15
        self.pendulum.state[3] += 0.3
        
    def closeEvent(self, event):
        self.simulation_timer.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PendulumSimulator()
    window.show()
    sys.exit(app.exec())
