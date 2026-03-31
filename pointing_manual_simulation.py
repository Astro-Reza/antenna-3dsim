import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation
import threading
import time

# --- Configuration ---
STL_FILES = {
    'azimuth': 'CADs-Azimuth-Body.stl',
    'elevation': 'CADs-Elevation-Body.stl',
    'polarization': 'CADs-Polarization-Body.stl'
}

DECIMATE_TARGET = {
    'azimuth': 0.5,       # Keep 50% of faces
    'elevation': 0.05,    # Keep 5% (77k -> ~4k faces)
    'polarization': 0.3   # Keep 30%
}

LINKS_CONFIG = {
    'azimuth': {
        'pivot': np.array([0.0, 0.0, 0.0]),
        'axis': np.array([0.0, 0.0, 1.0]),
        'color': 'gray'
    },
    'elevation': {
        'pivot': np.array([0.0, 0.0, 95.0]),
        'axis': np.array([1.0, 0.0, 0.0]),
        'color': 'skyblue'
    },
    'polarization': {
        'pivot': np.array([0.0, 135.0, 170.0]),
        'axis': np.array([0.0, 0.0, 1.0]),
        'color': 'orange'
    }
}

def get_transform_matrix(pivot, axis, angle_degrees):
    """Returns 4x4 matrix for rotation around pivot."""
    rad = np.radians(angle_degrees)
    rot = Rotation.from_rotvec(rad * axis)
    R = rot.as_matrix()
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = pivot - R @ pivot
    return M

def apply_transform_to_points(points, matrix):
    """Apply 4x4 transform to Nx3 points array."""
    ones = np.ones((points.shape[0], 1))
    pts_homo = np.hstack([points, ones])
    transformed = (matrix @ pts_homo.T).T
    return transformed[:, :3].astype(np.float32)

# Load and decimate meshes
print("Loading and decimating meshes...")
original_points = {}
display_meshes = {}

for name, path in STL_FILES.items():
    print(f"  Loading {name}...")
    mesh = pv.read(path)
    original_cells = mesh.n_cells
    
    # Decimate for performance
    target = DECIMATE_TARGET.get(name, 1.0)
    if target < 1.0:
        mesh = mesh.decimate(1.0 - target)
    
    # Convert to float32 for faster math
    mesh.points = mesh.points.astype(np.float32)
    
    original_points[name] = mesh.points.copy()
    display_meshes[name] = mesh
    print(f"    - {original_cells} -> {mesh.n_cells} cells (kept {target*100:.0f}%)")

# --- PyVista Plotter ---
plotter = pv.Plotter(window_size=(800, 600))
plotter.set_background('white')

for name, mesh in display_meshes.items():
    cfg = LINKS_CONFIG[name]
    plotter.add_mesh(
        mesh,
        color=cfg['color'],
        show_edges=False,
        opacity=0.9,
        smooth_shading=True,
        name=name
    )

# Add pivot point markers
pivot_colors = {'azimuth': 'red', 'elevation': 'green', 'polarization': 'blue'}
original_pivots = {}
pivot_meshes = {}

for name, cfg in LINKS_CONFIG.items():
    pivot = cfg['pivot']
    original_pivots[name] = pivot.copy()
    sphere = pv.Sphere(radius=8, center=pivot)
    plotter.add_mesh(sphere, color=pivot_colors[name], name=f'pivot_{name}')
    pivot_meshes[name] = sphere
    
    # Add label
    label = f"{name.upper()}\n({pivot[0]:.0f}, {pivot[1]:.0f}, {pivot[2]:.0f})"
    plotter.add_point_labels(
        [pivot + np.array([0, 0, 15])], 
        [label],
        font_size=12,
        point_color=pivot_colors[name],
        text_color='black',
        name=f'label_{name}',
        shape_opacity=0.7,
        always_visible=True
    )


plotter.add_axes()

# State
current_angles = {'azimuth': 0.0, 'elevation': 0.0, 'polarization': 0.0}
target_angles = {'azimuth': 88.0, 'elevation': 44.0, 'polarization': 90.0}
animation_active = False
animation_speed = 1.0  # degrees per frame

# Store slider widgets for updating
slider_widgets = {}

# Add text display for coordinates
coord_text = plotter.add_text(
    "Az: 0.0°  El: 0.0°  Pol: 0.0°",
    position='upper_right',
    font_size=12,
    color='black',
    name='coord_display'
)

def update_scene(_=None):
    """Recompute transforms and update mesh points in-place."""
    az_cfg = LINKS_CONFIG['azimuth']
    el_cfg = LINKS_CONFIG['elevation']
    pol_cfg = LINKS_CONFIG['polarization']
    
    M_az = get_transform_matrix(az_cfg['pivot'], az_cfg['axis'], current_angles['azimuth'])
    M_el = get_transform_matrix(el_cfg['pivot'], el_cfg['axis'], current_angles['elevation'])
    M_pol = get_transform_matrix(pol_cfg['pivot'], pol_cfg['axis'], current_angles['polarization'])
    
    transforms = {
        'azimuth': M_az,
        'elevation': M_az @ M_el,
        'polarization': M_az @ M_el @ M_pol
    }
    
    for name in display_meshes:
        new_points = apply_transform_to_points(original_points[name], transforms[name])
        display_meshes[name].points[:] = new_points
    
    # Update coordinate display text
    plotter.remove_actor('coord_display')
    plotter.add_text(
        f"Az: {current_angles['azimuth']:.1f}°  El: {current_angles['elevation']:.1f}°  Pol: {current_angles['polarization']:.1f}°",
        position='upper_right',
        font_size=12,
        color='black',
        name='coord_display'
    )

def make_callback(key):
    def callback(value):
        current_angles[key] = value
        update_scene()
    return callback

# Animation using VTK timer events (thread-safe)
timer_id = None

def animation_step(obj, event):
    """Animation step called by VTK timer."""
    global animation_active, timer_id
    
    done = True
    for key in ['azimuth', 'elevation', 'polarization']:
        diff = target_angles[key] - current_angles[key]
        if abs(diff) > 0.1:
            done = False
            step = min(animation_speed, abs(diff)) * np.sign(diff)
            current_angles[key] += step
    
    # Update slider widget values to match current angles
    for key in ['azimuth', 'elevation', 'polarization']:
        if key in slider_widgets and slider_widgets[key] is not None:
            slider_widgets[key].GetRepresentation().SetValue(current_angles[key])
    
    update_scene()
    plotter.render()
    
    if done:
        animation_active = False
        if timer_id is not None:
            plotter.iren.interactor.DestroyTimer(timer_id)
            timer_id = None
        print("Animation complete!")

def start_animation(state):
    """Start button callback."""
    global animation_active, timer_id
    if state and not animation_active:
        animation_active = True
        print(f"Moving to target: Az={target_angles['azimuth']}, El={target_angles['elevation']}, Pol={target_angles['polarization']}")
        # Add observer for timer events
        plotter.iren.interactor.AddObserver('TimerEvent', animation_step)
        timer_id = plotter.iren.interactor.CreateRepeatingTimer(30)  # 30ms interval

# Add Start button (top left)
plotter.add_checkbox_button_widget(
    start_animation,
    value=False,
    position=(10, 550),
    size=30,
    border_size=2,
    color_on='green',
    color_off='lightgray'
)

# Add label for Start button
plotter.add_text(
    "START",
    position=(50, 550),
    font_size=10,
    color='black',
    name='start_label'
)

# Add sliders (positioned higher) and store references
slider_az = plotter.add_slider_widget(
    make_callback('azimuth'),
    [0, 360], value=0, title='Azimuth',
    pointa=(0.1, 0.2), pointb=(0.4, 0.2),
    style='modern'
)

slider_el = plotter.add_slider_widget(
    make_callback('elevation'),
    [0, 90], value=0, title='Elevation',
    pointa=(0.1, 0.15), pointb=(0.4, 0.15),
    style='modern'
)

slider_pol = plotter.add_slider_widget(
    make_callback('polarization'),
    [0, 180], value=0, title='Polarization',
    pointa=(0.55, 0.2), pointb=(0.9, 0.2),
    style='modern'
)

# Store slider references for animation sync
slider_widgets['azimuth'] = slider_az
slider_widgets['elevation'] = slider_el
slider_widgets['polarization'] = slider_pol

plotter.camera_position = 'iso'
plotter.reset_camera()

print("Starting Real-time Simulation...")
print(f"\nTarget Position: Az={target_angles['azimuth']}°, El={target_angles['elevation']}°, Pol={target_angles['polarization']}°")
print("\nPivot Points:")
for name, cfg in LINKS_CONFIG.items():
    pivot = cfg['pivot']
    axis = cfg['axis']
    print(f"  {name.upper()}: Pivot=({pivot[0]}, {pivot[1]}, {pivot[2]}), Axis=({axis[0]}, {axis[1]}, {axis[2]})")

plotter.show()
