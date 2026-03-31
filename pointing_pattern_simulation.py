"""
Integrated Antenna Scan Simulation
Combines PyVista 3D antenna visualization with Pygame scan pattern plotting.
"""

import pygame
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# LAYOUT PYGAME WINDOW 
WIN_WIDTH = 1400
WIN_HEIGHT = 700
PYVISTA_WIDTH = 700
PYVISTA_HEIGHT = 700
PATTERN_WIDTH = 450
PATTERN_HEIGHT = 700
SIDEBAR_WIDTH = WIN_WIDTH - PYVISTA_WIDTH - PATTERN_WIDTH
PADDING = 20

# TARGET POSITION (Center of scan)
TARGET_AZIMUTH = 88.0
TARGET_ELEVATION = 44.0
SCAN_FOV = 20.0  # ±10 degrees from target

# STL CONFIGG
STL_FILES = {
    'azimuth': 'CADs-Azimuth-Body.stl',
    'elevation': 'CADs-Elevation-Body.stl',
    'polarization': 'CADs-Polarization-Body.stl'
}

DECIMATE_TARGET = {
    'azimuth': 0.5,
    'elevation': 0.05,
    'polarization': 0.3
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

# --- Transform Functions ---
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

def clip_polygon_to_rect(polygon, rect_left, rect_top, rect_right, rect_bottom):
    """Clip polygon to rectangle using Sutherland-Hodgman algorithm."""
    def inside(p, edge):
        x, y = p
        if edge == 'left': return x >= rect_left
        if edge == 'right': return x <= rect_right
        if edge == 'top': return y >= rect_top
        if edge == 'bottom': return y <= rect_bottom
    
    def intersection(p1, p2, edge):
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        
        if edge == 'left':
            t = (rect_left - x1) / dx if dx != 0 else 0
            return (rect_left, y1 + t * dy)
        elif edge == 'right':
            t = (rect_right - x1) / dx if dx != 0 else 0
            return (rect_right, y1 + t * dy)
        elif edge == 'top':
            t = (rect_top - y1) / dy if dy != 0 else 0
            return (x1 + t * dx, rect_top)
        elif edge == 'bottom':
            t = (rect_bottom - y1) / dy if dy != 0 else 0
            return (x1 + t * dx, rect_bottom)
    
    output = list(polygon)
    for edge in ['left', 'right', 'top', 'bottom']:
        if len(output) == 0:
            break
        input_list = output
        output = []
        for i in range(len(input_list)):
            current = input_list[i]
            next_pt = input_list[(i + 1) % len(input_list)]
            if inside(current, edge):
                output.append(current)
                if not inside(next_pt, edge):
                    output.append(intersection(current, next_pt, edge))
            elif inside(next_pt, edge):
                output.append(intersection(current, next_pt, edge))
    return output

# --- Pattern Generation (CPU for simplicity) ---
def generate_lissajous_pattern(num_pts, res_factor, fov_w, fov_h):
    """Generate Lissajous pattern points in FOV coordinates."""
    min_lobes, max_lobes = 5.0, 15.0
    k = min_lobes + (max_lobes - min_lobes) * res_factor
    fx, fy = k, k + 1.0
    
    points = []
    for i in range(num_pts):
        t = i / float(num_pts)
        x = (fov_w / 2.0) * np.cos(2 * np.pi * fx * t)
        y = (fov_h / 2.0) * np.cos(2 * np.pi * fy * t + np.pi/2)
        points.append([x, y])
    return np.array(points)

def generate_spiral_pattern(num_pts, num_turns, fov_w, fov_h):
    """Generate Archimedean spiral pattern."""
    max_radius = min(fov_w, fov_h) / 2.0
    total_theta = num_turns * 2.0 * np.pi
    
    points = []
    for i in range(num_pts):
        t = i / float(num_pts)
        theta = t * total_theta
        r = max_radius * t
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y])
    return np.array(points)

def generate_raster_pattern(num_pts, num_lines, fov_w, fov_h):
    """Generate raster (boustrophedon) pattern."""
    half_w = fov_w / 2.0
    half_h = fov_h / 2.0
    
    points = []
    for i in range(num_pts):
        t = i / float(num_pts)
        line_idx = int(t * num_lines)
        if line_idx >= num_lines:
            line_idx = num_lines - 1
        
        line_progress = (t * num_lines) - line_idx
        
        if num_lines > 1:
            raw_y = half_h - (line_idx / (num_lines - 1)) * fov_h
        else:
            raw_y = 0.0
        
        if line_idx % 2 == 0:
            raw_x = -half_w + line_progress * fov_w
        else:
            raw_x = half_w - line_progress * fov_w
        
        points.append([raw_x, raw_y])
    return np.array(points)

# --- UI Classes ---
class Slider:
    def __init__(self, x, y, w, h, min_val, max_val, start_val, label):
        self.rect = pygame.Rect(x, y, w, h)
        self.min_val, self.max_val, self.val = min_val, max_val, start_val
        self.label, self.dragging = label, False
        rel = (self.val - self.min_val) / (self.max_val - self.min_val)
        self.handle_rect = pygame.Rect(x + rel * w - 10, y - 5, 20, h + 10)
        self.visible = True

    def draw(self, screen, font):
        if not self.visible:
            return
        label_surf = font.render(f"{self.label}", True, (200, 200, 200))
        screen.blit(label_surf, (self.rect.x, self.rect.y - 25))
        val_surf = font.render(f"{self.val:.2f}", True, (50, 200, 255))
        screen.blit(val_surf, (self.rect.x + 110, self.rect.y - 25))
        pygame.draw.rect(screen, (80, 80, 80), self.rect)
        pygame.draw.rect(screen, (50, 150, 255), self.handle_rect)

    def update(self, event):
        if not self.visible:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.handle_rect.collidepoint(event.pos) or self.rect.collidepoint(event.pos):
                self.dragging = True
                self.move(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.move(event.pos[0])
            return True
        return False

    def move(self, mx):
        x = max(self.rect.left, min(mx, self.rect.right))
        self.handle_rect.centerx = x
        self.val = self.min_val + ((x - self.rect.left) / self.rect.width) * (self.max_val - self.min_val)

class Checkbox:
    def __init__(self, x, y, label, checked=True):
        self.rect = pygame.Rect(x, y, 20, 20)
        self.label = label
        self.checked = checked
        self.visible = True
        
    def draw(self, screen, font):
        if not self.visible:
            return
        color = (50, 150, 255) if self.checked else (80, 80, 80)
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, (200, 200, 200), self.rect, 2)
        text_surf = font.render(self.label, True, (220, 220, 220))
        screen.blit(text_surf, (self.rect.right + 10, self.rect.y))
        if self.checked:
            pygame.draw.line(screen, (255, 255, 255), (self.rect.left+4, self.rect.top+4), (self.rect.right-4, self.rect.bottom-4), 2)
            pygame.draw.line(screen, (255, 255, 255), (self.rect.left+4, self.rect.bottom-4), (self.rect.right-4, self.rect.top+4), 2)

    def update(self, event):
        if not self.visible:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.checked = not self.checked
                return True
        return False

class Dropdown:
    def __init__(self, x, y, w, h, options, label):
        self.rect = pygame.Rect(x, y, w, h)
        self.options = options
        self.label = label
        self.selected_idx = 0
        self.expanded = False
        self.option_rects = [pygame.Rect(x, y + h * (i + 1), w, h) for i in range(len(options))]
    
    @property
    def selected(self):
        return self.options[self.selected_idx]
    
    def draw(self, screen, font):
        label_surf = font.render(self.label, True, (200, 200, 200))
        screen.blit(label_surf, (self.rect.x, self.rect.y - 22))
        pygame.draw.rect(screen, (60, 60, 70), self.rect)
        pygame.draw.rect(screen, (100, 100, 120), self.rect, 2)
        text_surf = font.render(self.selected, True, (255, 255, 255))
        screen.blit(text_surf, (self.rect.x + 10, self.rect.y + 5))
        
        arrow_x = self.rect.right - 20
        arrow_y = self.rect.centery
        if self.expanded:
            pygame.draw.polygon(screen, (200, 200, 200), [(arrow_x, arrow_y + 4), (arrow_x + 10, arrow_y + 4), (arrow_x + 5, arrow_y - 4)])
        else:
            pygame.draw.polygon(screen, (200, 200, 200), [(arrow_x, arrow_y - 4), (arrow_x + 10, arrow_y - 4), (arrow_x + 5, arrow_y + 4)])
        
        if self.expanded:
            for i, opt in enumerate(self.options):
                rect = self.option_rects[i]
                bg_color = (80, 80, 100) if i == self.selected_idx else (50, 50, 60)
                pygame.draw.rect(screen, bg_color, rect)
                pygame.draw.rect(screen, (100, 100, 120), rect, 1)
                opt_surf = font.render(opt, True, (255, 255, 255))
                screen.blit(opt_surf, (rect.x + 10, rect.y + 5))
    
    def update(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.expanded = not self.expanded
                return False
            if self.expanded:
                for i, rect in enumerate(self.option_rects):
                    if rect.collidepoint(event.pos):
                        self.selected_idx = i
                        self.expanded = False
                        return True
                self.expanded = False
        return False

class Button:
    def __init__(self, x, y, w, h, label, color_off, color_on):
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.active = False
        self.color_off = color_off
        self.color_on = color_on
    
    def draw(self, screen, font):
        color = self.color_on if self.active else self.color_off
        pygame.draw.rect(screen, color, self.rect, border_radius=5)
        pygame.draw.rect(screen, (255, 255, 255), self.rect, 2, border_radius=5)
        text_surf = font.render(self.label, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)
    
    def update(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = not self.active
                return True
        return False

# --- Main Application ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption("Integrated Antenna Scan Simulation")
    font = pygame.font.SysFont("Arial", 14)
    title_font = pygame.font.SysFont("Arial", 18, bold=True)
    clock = pygame.time.Clock()
    
    # --- Load STL Meshes ---
    print("Loading meshes...")
    original_points = {}
    display_meshes = {}
    
    for name, path in STL_FILES.items():
        try:
            mesh = pv.read(path)
            target = DECIMATE_TARGET.get(name, 1.0)
            if target < 1.0:
                mesh = mesh.decimate(1.0 - target)
            mesh.points = mesh.points.astype(np.float32)
            original_points[name] = mesh.points.copy()
            display_meshes[name] = mesh
            print(f"  Loaded {name}: {mesh.n_cells} cells")
        except Exception as e:
            print(f"  Error loading {name}: {e}")
    
    # --- PyVista Offscreen Plotter ---
    plotter = pv.Plotter(off_screen=True, window_size=(PYVISTA_WIDTH, PYVISTA_HEIGHT))
    plotter.set_background('white')
    
    for name, mesh in display_meshes.items():
        cfg = LINKS_CONFIG[name]
        plotter.add_mesh(mesh, color=cfg['color'], show_edges=False, opacity=0.9, smooth_shading=True, name=name)
    
    plotter.add_axes()
    
    # Set camera position for better alignment (center on antenna)
    # Camera position: (camera_location, focal_point, view_up)
    plotter.camera_position = [(900, -900, 675), (0, 50, 100), (0, 0, 1)]
    
    # --- Animation State ---
    current_angles = {'azimuth': TARGET_AZIMUTH, 'elevation': TARGET_ELEVATION, 'polarization': 90.0}
    current_idx = 0.0  # Float for smooth animation
    is_playing = False
    animation_speed = 1.0
    
    # --- Camera control state ---
    camera_distance = 900.0
    camera_focal = np.array([0.0, 50.0, 100.0])
    is_panning = False
    last_mouse_pos = (0, 0)
    
    # --- Generate Pattern ---
    num_pts = 400
    fov_w = SCAN_FOV
    fov_h = SCAN_FOV * (PATTERN_HEIGHT / PATTERN_WIDTH)
    pattern_points = generate_lissajous_pattern(num_pts, 0.5, fov_w, fov_h)
    
    # --- UI Setup ---
    ui_x = PYVISTA_WIDTH + PATTERN_WIDTH + 15
    
    dropdown_pattern = Dropdown(ui_x, 60, 210, 28, ["Lissajous", "Spiral", "Raster"], "Pattern Type")
    
    slider_res = Slider(ui_x, 150, 160, 8, 0.0, 1.0, 0.5, "Res Factor")
    slider_turns = Slider(ui_x, 150, 160, 8, 1.0, 10.0, 5.0, "Num Turns")
    slider_lines = Slider(ui_x, 150, 160, 8, 5, 50, 15, "Num Lines")
    slider_samples = Slider(ui_x, 220, 160, 8, 100, 1000, 400, "Points")
    slider_speed = Slider(ui_x, 290, 160, 8, 0.1, 3.0, 1.0, "Speed")
    
    cb_path = Checkbox(ui_x, 340, "Show Path", True)
    cb_heatmap = Checkbox(ui_x, 370, "Show Heatmap", False)
    
    btn_play = Button(ui_x, 420, 100, 40, "▶ PLAY", (60, 60, 80), (50, 180, 50))
    btn_reset = Button(ui_x + 110, 420, 100, 40, "↺ RESET", (60, 60, 80), (180, 50, 50))
    
    pattern_sliders = {"Lissajous": slider_res, "Spiral": slider_turns, "Raster": slider_lines}
    
    def update_antenna_pose():
        """Update antenna mesh based on current angles."""
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
    
    def regenerate_pattern():
        nonlocal pattern_points, num_pts
        num_pts = int(slider_samples.val)
        pattern = dropdown_pattern.selected
        
        if pattern == "Lissajous":
            pattern_points = generate_lissajous_pattern(num_pts, slider_res.val, fov_w, fov_h)
        elif pattern == "Spiral":
            pattern_points = generate_spiral_pattern(num_pts, slider_turns.val, fov_w, fov_h)
        elif pattern == "Raster":
            pattern_points = generate_raster_pattern(num_pts, int(slider_lines.val), fov_w, fov_h)
    
    def pattern_to_screen(pt):
        """Convert pattern coordinates to screen coordinates for pattern display area."""
        scale_x = (PATTERN_WIDTH - 2 * PADDING) / fov_w
        scale_y = (PATTERN_HEIGHT - 2 * PADDING) / fov_h
        sx = PYVISTA_WIDTH + PATTERN_WIDTH // 2 + pt[0] * scale_x
        sy = PATTERN_HEIGHT // 2 - pt[1] * scale_y
        return int(sx), int(sy)
    
    # Initial pattern generation
    regenerate_pattern()
    update_antenna_pose()
    
    running = True
    last_pattern = dropdown_pattern.selected
    
    print("Starting simulation...")
    print(f"Target: Az={TARGET_AZIMUTH}°, El={TARGET_ELEVATION}°")
    
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            dropdown_pattern.update(event)
            
            for name, slider in pattern_sliders.items():
                if name == dropdown_pattern.selected:
                    if slider.update(event):
                        regenerate_pattern()
            
            if slider_samples.update(event):
                regenerate_pattern()
            
            slider_speed.update(event)
            cb_path.update(event)
            cb_heatmap.update(event)
            
            if btn_play.update(event):
                is_playing = btn_play.active
            
            if btn_reset.update(event):
                current_idx = 0
                btn_play.active = False
                is_playing = False
                btn_reset.active = False
            
            # --- 3D Viewport mouse controls ---
            # Check if mouse is in the 3D viewport area
            if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION, pygame.MOUSEWHEEL):
                mouse_x, mouse_y = pygame.mouse.get_pos()
                in_3d_viewport = mouse_x < PYVISTA_WIDTH
                
                if in_3d_viewport:
                    # Mouse wheel zoom
                    if event.type == pygame.MOUSEWHEEL:
                        zoom_factor = 50.0
                        camera_distance -= event.y * zoom_factor
                        camera_distance = max(200.0, min(2000.0, camera_distance))
                        # Update camera position
                        cam_pos = (camera_distance, -camera_distance, camera_distance * 0.75)
                        plotter.camera_position = [cam_pos, tuple(camera_focal), (0, 0, 1)]
                    
                    # Middle mouse button pan
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:  # Middle button
                        is_panning = True
                        last_mouse_pos = (mouse_x, mouse_y)
                    
                    if event.type == pygame.MOUSEBUTTONUP and event.button == 2:
                        is_panning = False
                    
                    if event.type == pygame.MOUSEMOTION and is_panning:
                        dx = mouse_x - last_mouse_pos[0]
                        dy = mouse_y - last_mouse_pos[1]
                        pan_speed = 0.5
                        camera_focal[0] -= dx * pan_speed
                        camera_focal[1] += dy * pan_speed
                        last_mouse_pos = (mouse_x, mouse_y)
                        # Update camera position
                        cam_pos = (camera_distance, -camera_distance, camera_distance * 0.75)
                        plotter.camera_position = [cam_pos, tuple(camera_focal), (0, 0, 1)]
        
        # --- Pattern regeneration on type change ---
        if dropdown_pattern.selected != last_pattern:
            regenerate_pattern()
            current_idx = 0
            last_pattern = dropdown_pattern.selected
        
        # Update slider visibility
        for name, slider in pattern_sliders.items():
            slider.visible = (name == dropdown_pattern.selected)
        
        # --- Animation Update ---
        if is_playing and num_pts > 0:
            animation_speed = slider_speed.val
            current_idx = (current_idx + animation_speed * 0.5) % num_pts
            
            # Update antenna angles based on current pattern position
            pt = pattern_points[int(current_idx)]
            current_angles['azimuth'] = TARGET_AZIMUTH - pt[0]
            current_angles['elevation'] = TARGET_ELEVATION - pt[1]
            update_antenna_pose()
            plotter.render()
        
        screen.fill((40, 40, 45))
        
        # RENDER PYVISTANYAA
        pv_offset_x = 20  # Shift right
        pv_offset_y = 30  # Shift down
        try:
            pv_img = plotter.screenshot(return_img=True)
            pv_surf = pygame.surfarray.make_surface(pv_img.swapaxes(0, 1))
            screen.blit(pv_surf, (pv_offset_x, pv_offset_y))
        except Exception as e:
            pygame.draw.rect(screen, (30, 30, 35), (0, 0, PYVISTA_WIDTH, PYVISTA_HEIGHT))
            err_text = font.render(f"3D Render Error: {e}", True, (255, 100, 100))
            screen.blit(err_text, (10, PYVISTA_HEIGHT // 2))
        
        # Area Buat Pattern
        pattern_rect = pygame.Rect(PYVISTA_WIDTH, 0, PATTERN_WIDTH, PATTERN_HEIGHT)
        pygame.draw.rect(screen, (255, 255, 255), pattern_rect)
        pygame.draw.rect(screen, (0, 0, 0), pattern_rect, 2)
        
        # Draw pattern boundary
        fov_rect = pygame.Rect(
            PYVISTA_WIDTH + PADDING,
            PADDING,
            PATTERN_WIDTH - 2 * PADDING,
            PATTERN_HEIGHT - 2 * PADDING
        )
        pygame.draw.rect(screen, (200, 200, 200), fov_rect, 1)
        
        # Define clipping bounds for pattern area
        clip_left = PYVISTA_WIDTH + PADDING
        clip_top = PADDING
        clip_right = PYVISTA_WIDTH + PATTERN_WIDTH - PADDING
        clip_bottom = PATTERN_HEIGHT - PADDING
        
        # Draw crosshairs at center (target)
        center_x = PYVISTA_WIDTH + PATTERN_WIDTH // 2
        center_y = PATTERN_HEIGHT // 2
        pygame.draw.line(screen, (200, 200, 200), (center_x - 20, center_y), (center_x + 20, center_y), 1)
        pygame.draw.line(screen, (200, 200, 200), (center_x, center_y - 20), (center_x, center_y + 20), 1)
        
        # Draw axis rulers with degree labels
        ruler_font = pygame.font.SysFont("Arial", 10)
        half_fov = SCAN_FOV / 2
        
        # X-axis ruler (Azimuth)
        for i in range(-int(half_fov), int(half_fov) + 1, 5):
            screen_x = center_x + i * (clip_right - clip_left) / SCAN_FOV
            tick_len = 8 if i % 10 == 0 else 4
            pygame.draw.line(screen, (100, 100, 100), (screen_x, clip_bottom - tick_len), (screen_x, clip_bottom), 1)
            if i % 10 == 0:
                label = ruler_font.render(f"{TARGET_AZIMUTH + i:.0f}°", True, (80, 80, 80))
                screen.blit(label, (screen_x - label.get_width()//2, clip_bottom + 2))
        
        # Y-axis ruler (Elevation)
        for i in range(-int(half_fov), int(half_fov) + 1, 5):
            screen_y = center_y - i * (clip_bottom - clip_top) / (SCAN_FOV * (PATTERN_HEIGHT / PATTERN_WIDTH))
            if screen_y < clip_top or screen_y > clip_bottom:
                continue
            tick_len = 8 if i % 10 == 0 else 4
            pygame.draw.line(screen, (100, 100, 100), (clip_left, screen_y), (clip_left + tick_len, screen_y), 1)
            if i % 10 == 0:
                label = ruler_font.render(f"{TARGET_ELEVATION + i:.0f}°", True, (80, 80, 80))
                screen.blit(label, (clip_left - label.get_width() - 3, screen_y - label.get_height()//2))
        
        # Axis labels
        az_label = font.render("Azimuth (°)", True, (60, 60, 60))
        screen.blit(az_label, (center_x - az_label.get_width()//2, clip_bottom + 18))
        
        el_label = font.render("El (°)", True, (60, 60, 60))
        screen.blit(el_label, (clip_left - 40, center_y - el_label.get_height()//2))
        
        # Draw Voronoi heatmap if enabled
        if cb_heatmap.checked and len(pattern_points) > 3:
            try:
                # Convert pattern points to screen coordinates
                screen_pts = np.array([pattern_to_screen(pt) for pt in pattern_points])
                num_orig = len(screen_pts)
                
                # Create boundary reflections (like resolution_simulations.py)
                # This ensures Voronoi cells are properly bounded by the rectangle
                box_width = clip_right - clip_left
                box_height = clip_bottom - clip_top
                box_center_x = (clip_left + clip_right) / 2
                box_center_y = (clip_top + clip_bottom) / 2
                
                # Reflect points across all 4 boundaries
                reflected_left = screen_pts.copy()
                reflected_left[:, 0] = 2 * clip_left - screen_pts[:, 0]
                
                reflected_right = screen_pts.copy()
                reflected_right[:, 0] = 2 * clip_right - screen_pts[:, 0]
                
                reflected_top = screen_pts.copy()
                reflected_top[:, 1] = 2 * clip_top - screen_pts[:, 1]
                
                reflected_bottom = screen_pts.copy()
                reflected_bottom[:, 1] = 2 * clip_bottom - screen_pts[:, 1]
                
                # Combine all points
                all_pts = np.vstack([screen_pts, reflected_left, reflected_right, reflected_top, reflected_bottom])
                
                vor = Voronoi(all_pts)
                
                # Color cells based on area (smaller = redder = higher probability)
                cmap = cm.get_cmap('RdYlGn_r')
                
                # Only process original points (not reflections)
                for i in range(num_orig):
                    region_idx = vor.point_region[i]
                    region = vor.regions[region_idx]
                    if -1 not in region and len(region) > 2:
                        polygon = [vor.vertices[v] for v in region]
                        
                        # Clip to bounds (simple clipping for any stray vertices)
                        clipped = []
                        for px, py in polygon:
                            cx = max(clip_left, min(px, clip_right))
                            cy = max(clip_top, min(py, clip_bottom))
                            clipped.append((cx, cy))
                        
                        if len(clipped) < 3:
                            continue
                        
                        # Calculate area using shoelace formula
                        poly_arr = np.array(clipped)
                        n = len(poly_arr)
                        area = 0.5 * abs(sum(poly_arr[j,0]*poly_arr[(j+1)%n,1] - poly_arr[(j+1)%n,0]*poly_arr[j,1] for j in range(n)))
                        
                        # Normalize area (smaller = redder)
                        norm_area = min(area / 3000.0, 1.0)
                        color = cmap(norm_area)
                        rgb = tuple(int(c * 255) for c in color[:3])
                        
                        pygame.draw.polygon(screen, rgb, clipped)
                        pygame.draw.polygon(screen, (100, 100, 100), clipped, 1)
            except Exception as e:
                pass  # Voronoi may fail for edge cases
        
        # Draw pattern path
        if cb_path.checked and len(pattern_points) > 1:
            screen_pts = [pattern_to_screen(pt) for pt in pattern_points]
            pygame.draw.lines(screen, (0, 0, 0), False, screen_pts, 2)
        
        # Draw current position indicator
        if num_pts > 0:
            current_pt = pattern_points[int(current_idx) % len(pattern_points)]
            sx, sy = pattern_to_screen(current_pt)
            pygame.draw.circle(screen, (255, 255, 255), (sx, sy), 10)
            pygame.draw.circle(screen, (255, 50, 50), (sx, sy), 8)
            pygame.draw.circle(screen, (0, 0, 0), (sx, sy), 10, 2)
        
        # Draw target marker (center)
        pygame.draw.circle(screen, (50, 150, 255), (center_x, center_y), 6)
        pygame.draw.circle(screen, (0, 0, 0), (center_x, center_y), 6, 2)
        
        # Draw info text on pattern area
        info_text = f"Target: Az {TARGET_AZIMUTH:.0f}°, El {TARGET_ELEVATION:.0f}°  |  FOV: ±{SCAN_FOV/2:.0f}°"
        info_surf = font.render(info_text, True, (80, 80, 80))
        screen.blit(info_surf, (PYVISTA_WIDTH + 10, PATTERN_HEIGHT - 30))
        
        # Current angle display
        angle_text = f"Current: Az {current_angles['azimuth']:.1f}°, El {current_angles['elevation']:.1f}°"
        angle_surf = font.render(angle_text, True, (80, 80, 80))
        screen.blit(angle_surf, (PYVISTA_WIDTH + 10, PATTERN_HEIGHT - 50))
        
        # --- Draw Sidebar ---
        sidebar_rect = pygame.Rect(PYVISTA_WIDTH + PATTERN_WIDTH, 0, SIDEBAR_WIDTH, WIN_HEIGHT)
        pygame.draw.rect(screen, (50, 50, 55), sidebar_rect)
        pygame.draw.line(screen, (0, 0, 0), (sidebar_rect.left, 0), (sidebar_rect.left, WIN_HEIGHT), 2)
        
        # Title
        title = title_font.render("Controls", True, (255, 255, 255))
        screen.blit(title, (ui_x, 20))
        
        # Draw UI elements
        for name, slider in pattern_sliders.items():
            if slider.visible:
                slider.draw(screen, font)
        
        slider_samples.draw(screen, font)
        slider_speed.draw(screen, font)
        cb_path.draw(screen, font)
        cb_heatmap.draw(screen, font)
        btn_play.draw(screen, font)
        btn_reset.draw(screen, font)
        
        # Progress indicator
        progress = (int(current_idx) / max(1, num_pts - 1)) * 100
        prog_text = font.render(f"Progress: {progress:.0f}%", True, (150, 150, 150))
        screen.blit(prog_text, (ui_x, 480))
        
        # FPS
        fps_surf = font.render(f"FPS: {clock.get_fps():.0f}", True, (100, 255, 100))
        screen.blit(fps_surf, (ui_x, WIN_HEIGHT - 30))
        
        # Draw dropdown last (on top)
        dropdown_pattern.draw(screen, font)
        
        pygame.display.flip()
        clock.tick(60)
    
    plotter.close()
    pygame.quit()

if __name__ == "__main__":
    main()
