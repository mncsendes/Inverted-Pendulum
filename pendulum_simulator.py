import sys
import numpy as np
from scipy.integrate import odeint
from PyQt6.QtWidgets import *
from PyQt6.QtCore import QTimer, Qt
import pyqtgraph as pg

class InvertedPendulum:
    def __init__(self):
        self.M = 1.0
        self.m = 0.3
        self.L = 0.5
        self.b = 0.1
        self.g = 9.81
        self.state = np.array([0.0, 0.0, 0.2, 0.0])
        
        self.Kp = 100.0
        self.Ki = 0.0
        self.Kd = 20.0
        self.integral = 0.0
        self.last_error = 0.0
        self.force = 0.0
        self.time = 0.0
        self.history = {'time': [], 'x': [], 'theta': [], 'force': []}
        
    def reset(self):
        self.state = np.array([0.0, 0.0, 0.2, 0.0])
        self.integral = 0.0
        self.last_error = 0.0
        self.force = 0.0
        self.time = 0.0
        self.history = {'time': [], 'x': [], 'theta': [], 'force': []}
        
    def dynamics(self, state, t, force):
        x, x_dot, theta, theta_dot = state
        M, m, L, b, g = self.M, self.m, self.L, self.b, self.g
        
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        denominator = M + m * sin_theta**2
        x_ddot = (force - b * x_dot + m * L * theta_dot**2 * sin_theta 
                  - m * g * sin_theta * cos_theta) / denominator
        theta_ddot = (g * sin_theta - x_ddot * cos_theta) / L
        
        return [x_dot, x_ddot, theta_dot, theta_ddot]
    
    def step(self, dt=0.01):
        error = self.state[2]
        self.integral += error * dt
        self.integral = np.clip(self.integral, -5.0, 5.0)
        derivative = (error - self.last_error) / dt if dt > 0 else 0
        self.last_error = error
        
        self.force = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.force = np.clip(self.force, -30.0, 30.0)
        
        if abs(error) < 0.005 and abs(self.state[1]) < 0.01:
            self.force = 0.0
        
        t_span = [self.time, self.time + dt]
        result = odeint(self.dynamics, self.state, t_span, args=(self.force,))
        self.state = result[-1]
        self.time += dt
        
        self.history['time'].append(self.time)
        self.history['x'].append(self.state[0])
        self.history['theta'].append(self.state[2] * 180 / np.pi)
        self.history['force'].append(self.force)
        
        return self.state
    
    def set_gains(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd


class PendulumSimulator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.pendulum = InvertedPendulum()
        self.dt = 0.01
        self.is_running = False
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(20)
        
        self.init_ui()
        self.setWindowTitle("Inverted Pendulum Simulator")
        self.setGeometry(100, 100, 1200, 700)
        
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout()
        central.setLayout(main_layout)
        
        plot_container = QWidget()
        plot_layout = QVBoxLayout()
        plot_container.setLayout(plot_layout)
        
        self.anim_view = pg.PlotWidget()
        self.anim_view.setXRange(-3, 3)
        self.anim_view.setYRange(-0.5, 1.5)
        self.anim_view.setAspectLocked(True)
        self.anim_view.showGrid(x=True, y=True)
        
        self.cart = self.anim_view.plot([0], [0], pen=None, symbol='s', 
                                        symbolSize=35, symbolBrush=(100, 200, 255))
        self.pendulum_line = self.anim_view.plot([0, 0], [0, 0], pen=pg.mkPen(color='r', width=3))
        self.anim_view.plot([-3, 3], [0, 0], pen=pg.mkPen(color='gray', width=1))
        
        plot_layout.addWidget(self.anim_view)
        
        self.state_plot = pg.PlotWidget()
        self.state_plot.setLabel('left', 'Angle (degrees)')
        self.state_plot.setLabel('bottom', 'Time (s)')
        self.state_plot.setXRange(0, 5)
        self.state_plot.setYRange(-45, 45)
        self.state_plot.showGrid(x=True, y=True)
        self.angle_curve = self.state_plot.plot(pen=pg.mkPen(color='cyan', width=2))
        plot_layout.addWidget(self.state_plot)
        
        self.pos_plot = pg.PlotWidget()
        self.pos_plot.setLabel('left', 'Position (m)')
        self.pos_plot.setLabel('bottom', 'Time (s)')
        self.pos_plot.setXRange(0, 5)
        self.pos_plot.setYRange(-1.5, 1.5)
        self.pos_plot.showGrid(x=True, y=True)
        self.pos_curve = self.pos_plot.plot(pen=pg.mkPen(color='orange', width=2))
        plot_layout.addWidget(self.pos_plot)
        
        main_layout.addWidget(plot_container, stretch=2)
        
        control_container = QWidget()
        control_layout = QVBoxLayout()
        control_container.setLayout(control_layout)
        
        pid_group = QGroupBox("PID Controller")
        pid_layout = QGridLayout()
        
        pid_layout.addWidget(QLabel("Kp"), 0, 0)
        self.kp_slider = QSlider(Qt.Orientation.Horizontal)
        self.kp_slider.setRange(0, 300)
        self.kp_slider.setValue(100)
        self.kp_slider.valueChanged.connect(self.update_gains)
        pid_layout.addWidget(self.kp_slider, 0, 1)
        self.kp_label = QLabel("100")
        pid_layout.addWidget(self.kp_label, 0, 2)
        
        pid_layout.addWidget(QLabel("Ki"), 1, 0)
        self.ki_slider = QSlider(Qt.Orientation.Horizontal)
        self.ki_slider.setRange(0, 20)
        self.ki_slider.setValue(0)
        self.ki_slider.valueChanged.connect(self.update_gains)
        pid_layout.addWidget(self.ki_slider, 1, 1)
        self.ki_label = QLabel("0")
        pid_layout.addWidget(self.ki_label, 1, 2)
        
        pid_layout.addWidget(QLabel("Kd"), 2, 0)
        self.kd_slider = QSlider(Qt.Orientation.Horizontal)
        self.kd_slider.setRange(0, 80)
        self.kd_slider.setValue(20)
        self.kd_slider.valueChanged.connect(self.update_gains)
        pid_layout.addWidget(self.kd_slider, 2, 1)
        self.kd_label = QLabel("20")
        pid_layout.addWidget(self.kd_label, 2, 2)
        
        pid_group.setLayout(pid_layout)
        control_layout.addWidget(pid_group)
        
        btn_group = QGroupBox("Controls")
        btn_layout = QVBoxLayout()
        
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.toggle_simulation)
        btn_layout.addWidget(self.start_btn)
        
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_simulation)
        btn_layout.addWidget(self.reset_btn)
        
        disturb_layout = QHBoxLayout()
        self.disturb_left_btn = QPushButton("← Push Left")
        self.disturb_left_btn.clicked.connect(lambda: self.apply_disturb(-0.2))
        self.disturb_left_btn.setStyleSheet("background-color: #ff6b6b; color: white; font-weight: bold; padding: 6px;")
        disturb_layout.addWidget(self.disturb_left_btn)
        
        self.disturb_right_btn = QPushButton("Push Right →")
        self.disturb_right_btn.clicked.connect(lambda: self.apply_disturb(0.2))
        self.disturb_right_btn.setStyleSheet("background-color: #4ecdc4; color: white; font-weight: bold; padding: 6px;")
        disturb_layout.addWidget(self.disturb_right_btn)
        
        btn_layout.addLayout(disturb_layout)
        
        btn_group.setLayout(btn_layout)
        control_layout.addWidget(btn_group)
        
        status_group = QGroupBox("Status")
        status_layout = QGridLayout()
        
        status_layout.addWidget(QLabel("Angle"), 0, 0)
        self.angle_display = QLabel("0.0°")
        status_layout.addWidget(self.angle_display, 0, 1)
        
        status_layout.addWidget(QLabel("Position"), 1, 0)
        self.pos_display = QLabel("0.00 m")
        status_layout.addWidget(self.pos_display, 1, 1)
        
        status_layout.addWidget(QLabel("Force"), 2, 0)
        self.force_display = QLabel("0.0 N")
        status_layout.addWidget(self.force_display, 2, 1)
        
        status_group.setLayout(status_layout)
        control_layout.addWidget(status_group)
        
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
            
            L = self.pendulum.L
            tip_x = x + L * np.sin(theta)
            tip_y = L * np.cos(theta)
            
            self.cart.setData([x], [0])
            self.pendulum_line.setData([x, tip_x], [0, tip_y])
            
            if len(self.pendulum.history['time']) > 0:
                times = self.pendulum.history['time'][-200:]
                angles = self.pendulum.history['theta'][-200:]
                positions = self.pendulum.history['x'][-200:]
                self.angle_curve.setData(times, angles)
                self.pos_curve.setData(times, positions)
                if len(times) > 1:
                    self.state_plot.setXRange(times[0], times[-1])
                    self.pos_plot.setXRange(times[0], times[-1])
            
            angle_deg = self.pendulum.state[2] * 180 / np.pi
            self.angle_display.setText(f"{angle_deg:.1f}°")
            self.pos_display.setText(f"{x:.2f} m")
            self.force_display.setText(f"{self.pendulum.force:.1f} N")
        
    def toggle_simulation(self):
        self.is_running = not self.is_running
        self.start_btn.setText("Stop" if self.is_running else "Start")
        
    def reset_simulation(self):
        self.pendulum.reset()
        self.is_running = False
        self.start_btn.setText("Start")
        self.angle_display.setText("0.0°")
        self.pos_display.setText("0.00 m")
        self.force_display.setText("0.0 N")
        self.cart.setData([0], [0])
        self.pendulum_line.setData([0, 0], [0, 0])
        self.angle_curve.setData([], [])
        self.pos_curve.setData([], [])
        
    def apply_disturb(self, magnitude):
        if not self.is_running:
            self.toggle_simulation()
        self.pendulum.state[2] += magnitude
        self.pendulum.state[3] += magnitude * 1.5
        
    def closeEvent(self, event):
        self.timer.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PendulumSimulator()
    window.show()
    sys.exit(app.exec())
