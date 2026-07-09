# Inverted Pendulum Simulator

Real-time PID control simulation of an inverted pendulum on a cart. Built in Python with PyQt6 and PyQtGraph.

# Features

- Real-time physics simulation using scipy.odeint
- Interactive PID tuning with live sliders
- Live oscilloscope view of pendulum angle
- Cart position tracking
- Disturbance rejection testing (left/right push buttons)
- Physical wall limits to prevent cart from leaving the screen

# Tech Stack

- Python 3.14+
- PyQt6 (GUI framework)
- PyQtGraph (real-time plotting)
- NumPy (numerical computation)
- SciPy (ODE integration)

# How It Works

The pendulum is modeled using the standard cart-pole dynamics equations. The PID controller continuously computes the force needed to keep the pendulum upright, balancing:

- **Kp (Proportional)**: Responds to the current angle error
- **Ki (Integral)**: Eliminates steady-state drift
- **Kd (Derivative)**: Dampens oscillations and prevents overshoot

# Setup & Run

```bash
# Clone the repository
git clone https://github.com/your-username/inverted-pendulum.git
cd inverted-pendulum

# Install dependencies
pip install numpy scipy PyQt6 pyqtgraph

# Run the simulator
python pendulum_simulator.py
