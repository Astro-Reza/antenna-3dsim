import os
import math
import random
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from scipy.spatial.transform import Rotation
from collections import deque

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                             QVBoxLayout, QSlider, QLabel, QPushButton, 
                             QGroupBox, QGridLayout, QSplitter)
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg

# --- Configuration & Kinematics ---
STL_FILES = {
    'azimuth': 'CADs-Azimuth-Body.stl',
    'elevation': 'CADs-Elevation-Body.stl',
    'polarization': 'CADs-Polarization-Body.stl'
}

LINKS_CONFIG = {
    'azimuth': {'pivot': np.array([0.0, 0.0, 0.0]), 'axis': np.array([0.0, 0.0, 1.0]), 'color': 'gray'},
    'elevation': {'pivot': np.array([0.0, 0.0, 95.0]), 'axis': np.array([1.0, 0.0, 0.0]), 'color': 'skyblue'},
    'polarization': {'pivot': np.array([0.0, 135.0, 170.0]), 'axis': np.array([0.0, 0.0, 1.0]), 'color': 'orange'}
}

AXES = ['azimuth', 'elevation', 'polarization']
DT = 0.05
MAX_PTS = 200

def get_transform_matrix(pivot, axis, angle_degrees):
    rad = np.radians(angle_degrees)
    rot = Rotation.from_rotvec(rad * axis)
    R = rot.as_matrix()
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = pivot - R @ pivot
    return M

def apply_transform_to_points(points, matrix):
    ones = np.ones((points.shape[0], 1))
    pts_homo = np.hstack([points, ones])
    transformed = (matrix @ pts_homo.T).T
    return transformed[:, :3].astype(np.float32)

def get_load_factor(axis_name, elevation_angle):
    rad = math.radians(elevation_angle)
    if axis_name == 'elevation': return max(0.30, math.cos(rad))
    elif axis_name == 'azimuth': return max(0.75, 1 - 0.15 * math.sin(rad))
    return max(0.65, 1 - 0.22 * math.sin(rad))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("COTP Antenna Simulation")
        self.resize(1200, 800)

        # State Variables
        self.sim_time = 0.0
        self.running = False
        self.params = {'noise': 1.5, 'speed': 2.0, 'alpha': 0.15}
        self.targets = {'azimuth': 88.0, 'elevation': 44.0, 'polarization': 90.0}
        self.state = {ax: {'true': 0.0, 'fused': 0.0, 'pwm': 0.0, 'phase': 'IDLE'} for ax in AXES}
        
        # Telemetry Data
        self.t_data = deque(maxlen=MAX_PTS)
        self.pwm_data = {ax: deque(maxlen=MAX_PTS) for ax in AXES}
        self.ang_data = {ax: {'true': deque(maxlen=MAX_PTS), 'sensor': deque(maxlen=MAX_PTS), 'fused': deque(maxlen=MAX_PTS), 'target': deque(maxlen=MAX_PTS)} for ax in AXES}

        # Setup UI
        self._setup_ui()
        self._setup_3d()
        
        # Setup Timer loop
        self.timer = QTimer()
        self.timer.timeout.connect(self.step_simulation)
        self.timer.start(int(DT * 1000))

    def _setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Use QSplitter for reactive resizing
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left side: 3D Render
        self.plotter = QtInteractor(self)
        splitter.addWidget(self.plotter)

        # Right side: GUI Panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        splitter.addWidget(right_panel)
        
        # Set splitter ratio (60% 3D, 40% GUI)
        splitter.setSizes([700, 500])

        # --- Controls ---
        ctrl_group = QGroupBox("Control Panel")
        ctrl_layout = QGridLayout(ctrl_group)
        
        self.labels = {}
        row = 0
        for name, max_val, default in [('Azimuth', 360, 88), ('Elevation', 90, 44), ('Polarization', 180, 90)]:
            ctrl_layout.addWidget(QLabel(name), row, 0)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, max_val)
            slider.setValue(default)
            lbl = QLabel(f"{default}°")
            
            slider.valueChanged.connect(lambda v, n=name.lower(), l=lbl: self.update_target(n, v, l))
            
            ctrl_layout.addWidget(slider, row, 1)
            ctrl_layout.addWidget(lbl, row, 2)
            row += 1

        for name, max_val, default, step_mult, is_float in [('Noise', 50, 15, 10, True), ('Speed', 50, 20, 10, True)]:
            ctrl_layout.addWidget(QLabel(name), row, 0)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(1 if name=='Speed' else 0, max_val)
            slider.setValue(default)
            lbl = QLabel(f"{default/step_mult}")
            
            slider.valueChanged.connect(lambda v, n=name.lower(), l=lbl, sm=step_mult: self.update_param(n, v, l, sm))
            
            ctrl_layout.addWidget(slider, row, 1)
            ctrl_layout.addWidget(lbl, row, 2)
            row += 1

        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("START")
        self.btn_stop = QPushButton("STOP")
        self.btn_reset = QPushButton("RESET")
        
        self.btn_start.clicked.connect(self.start_sim)
        self.btn_stop.clicked.connect(self.stop_sim)
        self.btn_reset.clicked.connect(self.reset_sim)
        
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addWidget(self.btn_reset)
        ctrl_layout.addLayout(btn_layout, row, 0, 1, 3)
        
        right_layout.addWidget(ctrl_group)

        # --- Plots (PyQtGraph) ---
        pg.setConfigOptions(antialias=True)
        
        # PWM Plot
        self.pwm_plot = pg.PlotWidget(title="PWM Duty Cycle (%)")
        self.pwm_plot.addLegend()
        self.pwm_plot.setYRange(0, 105)
        self.pwm_plot.showGrid(x=True, y=True, alpha=0.3)
        self.lines_pwm = {
            'azimuth': self.pwm_plot.plot(pen=pg.mkPen('#378add', width=2), name="Azimuth"),
            'elevation': self.pwm_plot.plot(pen=pg.mkPen('#639922', width=2), name="Elevation"),
            'polarization': self.pwm_plot.plot(pen=pg.mkPen('#ba7517', width=2), name="Polarization")
        }
        right_layout.addWidget(self.pwm_plot)

        # Angle Plot (Showing Azimuth by Default)
        self.ang_plot = pg.PlotWidget(title="Angle Response - Azimuth (°)")
        self.ang_plot.addLegend()
        self.ang_plot.setYRange(0, 370)
        self.ang_plot.showGrid(x=True, y=True, alpha=0.3)
        self.lines_ang = {
            'target': self.ang_plot.plot(pen=pg.mkPen('gray', style=Qt.DashLine), name="Target"),
            'sensor': self.ang_plot.plot(pen=pg.mkPen(color=(55, 138, 221, 100)), name="Sensor (Noisy)"),
            'fused': self.ang_plot.plot(pen=pg.mkPen('#7f77dd', width=2), name="Fused"),
            'true': self.ang_plot.plot(pen=pg.mkPen('#378add', width=3), name="True")
        }
        right_layout.addWidget(self.ang_plot)

    def _setup_3d(self):
        self.original_points = {}
        self.display_meshes = {}
        
        self.plotter.set_background('white')

        for name, cfg in LINKS_CONFIG.items():
            path = STL_FILES[name]
            if os.path.exists(path):
                mesh = pv.read(path).decimate(0.5)
            else:
                center = cfg['pivot']
                if name == 'azimuth': mesh = pv.Cylinder(center=center, direction=[0,0,1], radius=40, height=20)
                elif name == 'elevation': mesh = pv.Box(bounds=(center[0]-30, center[0]+30, center[1]-10, center[1]+10, center[2]-40, center[2]+40))
                else: mesh = pv.Cylinder(center=center, direction=[0,1,0], radius=20, height=60)
            
            mesh.points = mesh.points.astype(np.float32)
            self.original_points[name] = mesh.points.copy()
            self.display_meshes[name] = mesh
            self.plotter.add_mesh(mesh, color=cfg['color'], smooth_shading=True)

        self.plotter.add_axes()
        self.plotter.camera_position = 'iso'

    # --- Callbacks ---
    def update_target(self, name, val, lbl):
        self.targets[name] = float(val)
        lbl.setText(f"{val}°")

    def update_param(self, name, val, lbl, mult):
        v = val / mult
        self.params[name] = v
        lbl.setText(f"{v}")

    def start_sim(self): self.running = True
    def stop_sim(self): self.running = False
    def reset_sim(self):
        self.running = False
        self.sim_time = 0.0
        self.t_data.clear()
        for ax in AXES:
            self.state[ax] = {'true': 0.0, 'fused': 0.0, 'pwm': 0.0, 'phase': 'IDLE'}
            self.pwm_data[ax].clear()
            for k in self.ang_data[ax]: self.ang_data[ax][k].clear()
        self.update_plots()
        self.update_3d()

    # --- Main Loop Logic ---
    def step_simulation(self):
        if not self.running:
            return

        self.sim_time += DT
        self.t_data.append(self.sim_time)
        
        BASE_STALL = 28.0
        RAMP_RATE = 25.0
        current_elevation = self.state['elevation']['true']
        
        for ax in AXES:
            s = self.state[ax]
            target = self.targets[ax]
            err = target - s['fused']
            
            load_factor = get_load_factor(ax, current_elevation)
            stall_threshold = BASE_STALL * load_factor
            run_pwm = stall_threshold + 18.0 * load_factor
            
            if abs(err) < 0.2:
                s['phase'] = 'DONE'
                s['pwm'] = 0.0
            elif s['phase'] in ['IDLE', 'DONE']:
                s['phase'] = 'RAMPING'
                
            if s['phase'] == 'RAMPING':
                s['pwm'] = min(s['pwm'] + RAMP_RATE * DT * self.params['speed'], 100)
                if s['pwm'] >= stall_threshold:
                    s['phase'] = 'RUNNING'
            elif s['phase'] == 'RUNNING':
                s['pwm'] += (run_pwm - s['pwm']) * 0.25
                
            velocity = 0.0
            if s['phase'] == 'RUNNING' and s['pwm'] >= stall_threshold:
                velocity = math.copysign(max(0, s['pwm'] - stall_threshold) * 0.55 * load_factor * self.params['speed'], err)
                
            s['true'] += velocity * DT
            
            # Noise & Low-Pass Filter
            sensor_val = s['true'] + random.uniform(-self.params['noise'], self.params['noise'])
            s['fused'] = self.params['alpha'] * sensor_val + (1 - self.params['alpha']) * s['fused']
            
            # Log Data
            self.pwm_data[ax].append(s['pwm'])
            self.ang_data[ax]['true'].append(s['true'])
            self.ang_data[ax]['sensor'].append(sensor_val)
            self.ang_data[ax]['fused'].append(s['fused'])
            self.ang_data[ax]['target'].append(target)

        if all(self.state[ax]['phase'] == 'DONE' for ax in AXES):
            self.running = False

        self.update_plots()
        self.update_3d()

    def update_plots(self):
        if not self.t_data: return
        t = list(self.t_data)
        
        for ax in AXES:
            self.lines_pwm[ax].setData(t, list(self.pwm_data[ax]))
            
        # Hardcoding to view Azimuth response for the angle plot.
        # You can tie this to a dropdown later!
        view_ax = 'azimuth' 
        self.lines_ang['target'].setData(t, list(self.ang_data[view_ax]['target']))
        self.lines_ang['sensor'].setData(t, list(self.ang_data[view_ax]['sensor']))
        self.lines_ang['fused'].setData(t, list(self.ang_data[view_ax]['fused']))
        self.lines_ang['true'].setData(t, list(self.ang_data[view_ax]['true']))

    def update_3d(self):
        az_cfg = LINKS_CONFIG['azimuth']
        el_cfg = LINKS_CONFIG['elevation']
        pol_cfg = LINKS_CONFIG['polarization']
        
        M_az = get_transform_matrix(az_cfg['pivot'], az_cfg['axis'], self.state['azimuth']['true'])
        M_el = get_transform_matrix(el_cfg['pivot'], el_cfg['axis'], self.state['elevation']['true'])
        M_pol = get_transform_matrix(pol_cfg['pivot'], pol_cfg['axis'], self.state['polarization']['true'])
        
        transforms = {
            'azimuth': M_az,
            'elevation': M_az @ M_el,
            'polarization': M_az @ M_el @ M_pol
        }
        
        for name in AXES:
            new_points = apply_transform_to_points(self.original_points[name], transforms[name])
            self.display_meshes[name].points[:] = new_points
            
        self.plotter.update()


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())