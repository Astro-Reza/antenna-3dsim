import pygame
import numpy as np
import matplotlib.pyplot as plt
import taichi as ti

# Initialize Taichi
try:
    ti.init(arch=ti.gpu) 
except:
    ti.init(arch=ti.vulkan)

# --- LAYOUT CONFIGURATION ---
WIN_WIDTH = 1000
WIN_HEIGHT = 700
SIDEBAR_WIDTH = 250
PLOT_WIDTH = WIN_WIDTH - SIDEBAR_WIDTH
PLOT_HEIGHT = WIN_HEIGHT

MAX_POINTS = 2000 
PADDING = 20

# --- GPU FIELDS ---
points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_POINTS * 5)
closest_site = ti.field(dtype=ti.int32, shape=(PLOT_WIDTH, PLOT_HEIGHT))
site_area = ti.field(dtype=ti.int32, shape=MAX_POINTS * 5)
screen_pixels = ti.Vector.field(3, dtype=ti.f32, shape=(PLOT_WIDTH, PLOT_HEIGHT))
colormap_lut = ti.Vector.field(3, dtype=ti.f32, shape=256)

# --- COLORMAP SETUP ---
def load_colormap():
    cmap = plt.get_cmap('RdYlGn_r')
    colors = cmap(np.linspace(0, 1, 256))[:, :3]
    colormap_lut.from_numpy(colors.astype(np.float32))

load_colormap()

# --- TAICHI KERNELS ---

@ti.kernel
def generate_lissajous(res_factor: float, num: int, width_fov: float, height_fov: float):
    min_lobes, max_lobes = 5.0, 15.0
    k = min_lobes + (max_lobes - min_lobes) * res_factor
    fx, fy = k, k + 1.0
    
    # Add padding to edges
    padding = 20.0
    effective_width = PLOT_WIDTH - 2.0 * padding
    effective_height = PLOT_HEIGHT - 2.0 * padding
    
    scale_x = effective_width / width_fov
    scale_y = effective_height / height_fov
    center_x = PLOT_WIDTH / 2.0
    center_y = PLOT_HEIGHT / 2.0

    for i in range(num):
        t = i / float(num)
        # Lissajous Math
        raw_x = (width_fov / 2.0) * ti.cos(2 * 3.14159 * fx * t)
        raw_y = (height_fov / 2.0) * ti.cos(2 * 3.14159 * fy * t + 1.5708)
        
        # Center Point
        points[i] = ti.Vector([raw_x * scale_x + center_x, raw_y * scale_y + center_y])
        
        # Reflections (Left, Right, Down, Up)
        points[num + i]   = ti.Vector([(-width_fov - raw_x) * scale_x + center_x, raw_y * scale_y + center_y])
        points[2*num + i] = ti.Vector([(width_fov - raw_x) * scale_x + center_x, raw_y * scale_y + center_y])
        points[3*num + i] = ti.Vector([raw_x * scale_x + center_x, (-height_fov - raw_y) * scale_y + center_y])
        points[4*num + i] = ti.Vector([raw_x * scale_x + center_x, (height_fov - raw_y) * scale_y + center_y])

@ti.kernel
def generate_spiral(num_turns: float, num: int, width_fov: float, height_fov: float):
    """Archimedean spiral: r = a * theta"""
    padding = 20.0
    effective_width = PLOT_WIDTH - 2.0 * padding
    effective_height = PLOT_HEIGHT - 2.0 * padding
    
    scale_x = effective_width / width_fov
    scale_y = effective_height / height_fov
    center_x = PLOT_WIDTH / 2.0
    center_y = PLOT_HEIGHT / 2.0
    
    # Maximum radius should fill the FOV
    max_radius = ti.min(width_fov, height_fov) / 2.0
    total_theta = num_turns * 2.0 * 3.14159
    
    for i in range(num):
        t = i / float(num)
        theta = t * total_theta
        # Archimedean spiral: r grows linearly with theta
        r = max_radius * t
        
        raw_x = r * ti.cos(theta)
        raw_y = r * ti.sin(theta)
        
        # Center Point
        points[i] = ti.Vector([raw_x * scale_x + center_x, raw_y * scale_y + center_y])
        
        # Reflections (Left, Right, Down, Up)
        points[num + i]   = ti.Vector([(-width_fov - raw_x) * scale_x + center_x, raw_y * scale_y + center_y])
        points[2*num + i] = ti.Vector([(width_fov - raw_x) * scale_x + center_x, raw_y * scale_y + center_y])
        points[3*num + i] = ti.Vector([raw_x * scale_x + center_x, (-height_fov - raw_y) * scale_y + center_y])
        points[4*num + i] = ti.Vector([raw_x * scale_x + center_x, (height_fov - raw_y) * scale_y + center_y])

@ti.kernel
def generate_raster(num_lines: int, num: int, width_fov: float, height_fov: float):
    """Raster scan: horizontal sweeps with alternating direction (boustrophedon)"""
    padding = 20.0
    effective_width = PLOT_WIDTH - 2.0 * padding
    effective_height = PLOT_HEIGHT - 2.0 * padding
    
    scale_x = effective_width / width_fov
    scale_y = effective_height / height_fov
    center_x = PLOT_WIDTH / 2.0
    center_y = PLOT_HEIGHT / 2.0
    
    half_w = width_fov / 2.0
    half_h = height_fov / 2.0
    
    for i in range(num):
        t = i / float(num)
        
        # Determine which line we're on
        line_idx = int(t * float(num_lines))
        if line_idx >= num_lines:
            line_idx = num_lines - 1
        
        # Calculate position within the line (0 to 1)
        line_progress = (t * float(num_lines)) - float(line_idx)
        
        # Y position: from top to bottom
        raw_y = half_h - (float(line_idx) / float(num_lines - 1)) * height_fov if num_lines > 1 else 0.0
        
        # Initialize raw_x (required by Taichi before conditional assignment)
        raw_x = 0.0
        
        # X position: alternating direction (boustrophedon)
        if line_idx % 2 == 0:
            raw_x = -half_w + line_progress * width_fov  # Left to right
        else:
            raw_x = half_w - line_progress * width_fov   # Right to left
        
        # Center Point
        points[i] = ti.Vector([raw_x * scale_x + center_x, raw_y * scale_y + center_y])
        
        # Reflections (Left, Right, Down, Up)
        points[num + i]   = ti.Vector([(-width_fov - raw_x) * scale_x + center_x, raw_y * scale_y + center_y])
        points[2*num + i] = ti.Vector([(width_fov - raw_x) * scale_x + center_x, raw_y * scale_y + center_y])
        points[3*num + i] = ti.Vector([raw_x * scale_x + center_x, (-height_fov - raw_y) * scale_y + center_y])
        points[4*num + i] = ti.Vector([raw_x * scale_x + center_x, (height_fov - raw_y) * scale_y + center_y])

@ti.kernel
def compute_voronoi(total_pts: int):
    for x, y in closest_site:
        min_dist = 1e9
        closest_idx = -1
        pixel_pos = ti.Vector([float(x), float(y)])
        
        for k in range(total_pts):
            dist = (pixel_pos - points[k]).norm_sqr()
            if dist < min_dist:
                min_dist = dist
                closest_idx = k
        closest_site[x, y] = closest_idx

@ti.kernel
def compute_area():
    for i in site_area:
        site_area[i] = 0
    for x, y in closest_site:
        ti.atomic_add(site_area[closest_site[x, y]], 1)

@ti.kernel
def render_frame(min_area: float, max_area: float, num_pts: int, show_color: int, show_walls: int, show_dots: int):
    denom = max_area - min_area
    if denom == 0: denom = 1.0
    
    # Padding for empty border
    padding = 20

    for x, y in screen_pixels:
        # Check if pixel is in padding area - render as white and skip
        if x < padding or x >= PLOT_WIDTH - padding or y < padding or y >= PLOT_HEIGHT - padding:
            screen_pixels[x, y] = ti.Vector([1.0, 1.0, 1.0])
            continue
            
        idx = closest_site[x, y]
        
        # --- 1. Base Color ---
        if show_color == 1:
            area = float(site_area[idx])
            norm = (area - min_area) / denom
            norm = ti.max(0.0, ti.min(norm, 1.0))
            lut_idx = int(norm * 255)
            screen_pixels[x, y] = colormap_lut[lut_idx]
        else:
            # White background if color is off
            screen_pixels[x, y] = ti.Vector([1.0, 1.0, 1.0])

        # --- 2. Walls ---
        if show_walls == 1:
            is_edge = False
            if x < PLOT_WIDTH - 1 and closest_site[x+1, y] != idx: is_edge = True
            if y < PLOT_HEIGHT - 1 and closest_site[x, y+1] != idx: is_edge = True
            
            if is_edge:
                if show_color == 1:
                    # Black edges on colored map
                    screen_pixels[x, y] = ti.Vector([0.0, 0.0, 0.0]) 
                else:
                    # Black edges on white map (Wireframe look)
                    screen_pixels[x, y] = ti.Vector([0.0, 0.0, 0.0])

        # --- 3. Dots (FIXED) ---
        if show_dots == 1:
            p = points[idx]
            # Simple distance check for dot
            if (ti.Vector([float(x), float(y)]) - p).norm_sqr() < 9.0: # Radius squared
                if show_color == 1:
                    screen_pixels[x, y] = ti.Vector([0.0, 0.0, 0.0])
                else:
                    screen_pixels[x, y] = ti.Vector([0.0, 0.0, 0.0])

# --- STATE MANAGER ---
class SmoothingState:
    def __init__(self):
        self.smooth_min = 0.0
        self.smooth_max = 1000.0

# --- UI CLASSES ---
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
        screen.blit(val_surf, (self.rect.x + 150, self.rect.y - 25))
        pygame.draw.rect(screen, (80, 80, 80), self.rect)
        pygame.draw.rect(screen, (50, 150, 255), self.handle_rect)

    def update(self, event):
        if not self.visible:
            return False
        updated = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.handle_rect.collidepoint(event.pos) or self.rect.collidepoint(event.pos):
                self.dragging, updated = True, True
                self.move(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP: self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.move(event.pos[0])
            updated = True
        return updated

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
        # Draw box
        color = (50, 150, 255) if self.checked else (80, 80, 80)
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, (200, 200, 200), self.rect, 2) # Border
        
        # Draw Label
        text_surf = font.render(self.label, True, (220, 220, 220))
        screen.blit(text_surf, (self.rect.right + 10, self.rect.y))
        
        # Checkmark (X)
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
        self.option_rects = []
        for i in range(len(options)):
            self.option_rects.append(pygame.Rect(x, y + h * (i + 1), w, h))
    
    @property
    def selected(self):
        return self.options[self.selected_idx]
    
    def draw(self, screen, font):
        # Draw label
        label_surf = font.render(self.label, True, (200, 200, 200))
        screen.blit(label_surf, (self.rect.x, self.rect.y - 22))
        
        # Draw main box
        pygame.draw.rect(screen, (60, 60, 70), self.rect)
        pygame.draw.rect(screen, (100, 100, 120), self.rect, 2)
        
        # Draw selected text
        text_surf = font.render(self.selected, True, (255, 255, 255))
        screen.blit(text_surf, (self.rect.x + 10, self.rect.y + 5))
        
        # Draw arrow
        arrow_x = self.rect.right - 20
        arrow_y = self.rect.centery
        if self.expanded:
            pygame.draw.polygon(screen, (200, 200, 200), [(arrow_x, arrow_y + 4), (arrow_x + 10, arrow_y + 4), (arrow_x + 5, arrow_y - 4)])
        else:
            pygame.draw.polygon(screen, (200, 200, 200), [(arrow_x, arrow_y - 4), (arrow_x + 10, arrow_y - 4), (arrow_x + 5, arrow_y + 4)])
        
        # Draw expanded options
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
            # Check if clicked on main box
            if self.rect.collidepoint(event.pos):
                self.expanded = not self.expanded
                return False
            
            # Check if clicked on an option
            if self.expanded:
                for i, rect in enumerate(self.option_rects):
                    if rect.collidepoint(event.pos):
                        self.selected_idx = i
                        self.expanded = False
                        return True  # Selection changed
                # Clicked elsewhere, close dropdown
                self.expanded = False
        return False
    
    def get_expanded_height(self):
        if self.expanded:
            return self.rect.height * (len(self.options) + 1)
        return self.rect.height

# --- MAIN ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption("Scan Pattern Visualization Studio")
    font = pygame.font.SysFont("Arial", 16)
    title_font = pygame.font.SysFont("Arial", 22, bold=True)
    clock = pygame.time.Clock()
    
    # --- UI LAYOUT ---
    ui_x = PLOT_WIDTH + 25
    
    # Pattern Selector Dropdown
    dropdown_pattern = Dropdown(ui_x, 50, 200, 28, ["Lissajous", "Spiral", "Raster"], "Pattern Type")
    
    # Pattern-specific sliders
    slider_res = Slider(ui_x, 130, 200, 8, 0.0, 1.0, 0.5, "Res Factor")       # Lissajous
    slider_turns = Slider(ui_x, 130, 200, 8, 1.0, 10.0, 5.0, "Num Turns")     # Spiral
    slider_lines = Slider(ui_x, 130, 200, 8, 5, 50, 15, "Num Lines")          # Raster
    
    # Common slider
    slider_samples = Slider(ui_x, 200, 200, 8, 100, 1000, 400, "Point Count")
    
    # Checkboxes
    cb_color = Checkbox(ui_x, 270, "Show Color (Heatmap)", True)
    cb_walls = Checkbox(ui_x, 310, "Show Walls", True)
    cb_dots  = Checkbox(ui_x, 350, "Show Dots", False)
    cb_path  = Checkbox(ui_x, 390, "Show Path", False)
    
    # Store pattern-specific UI groups
    pattern_sliders = {
        "Lissajous": slider_res,
        "Spiral": slider_turns,
        "Raster": slider_lines
    }
    
    state = SmoothingState()
    running = True
    
    while running:
        # Background (White)
        screen.fill((255, 255, 255))
        
        # Draw Sidebar Background
        pygame.draw.rect(screen, (40, 40, 45), (PLOT_WIDTH, 0, SIDEBAR_WIDTH, WIN_HEIGHT))
        pygame.draw.line(screen, (0, 0, 0), (PLOT_WIDTH, 0), (PLOT_WIDTH, WIN_HEIGHT), 2)

        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                running = False
            
            # Update dropdown first (it may overlap other elements)
            dropdown_pattern.update(event)
            
            # Update visible sliders
            for name, slider in pattern_sliders.items():
                if name == dropdown_pattern.selected:
                    slider.update(event)
            
            slider_samples.update(event)
            cb_color.update(event)
            cb_walls.update(event)
            cb_dots.update(event)
            cb_path.update(event)

        # Update slider visibility based on pattern
        current_pattern = dropdown_pattern.selected
        for name, slider in pattern_sliders.items():
            slider.visible = (name == current_pattern)

        # 1. GPU Calculations
        num_pts = int(slider_samples.val)
        fov_w = 20.0
        fov_h = 20.0 * (PLOT_HEIGHT / PLOT_WIDTH) 
        
        # Generate pattern based on selection
        if current_pattern == "Lissajous":
            generate_lissajous(slider_res.val, num_pts, fov_w, fov_h)
        elif current_pattern == "Spiral":
            generate_spiral(slider_turns.val, num_pts, fov_w, fov_h)
        elif current_pattern == "Raster":
            generate_raster(int(slider_lines.val), num_pts, fov_w, fov_h)
        
        compute_voronoi(num_pts * 5)
        compute_area()
        
        # 2. Smoothing
        raw_areas = site_area.to_numpy()[:num_pts]
        if len(raw_areas) > 0:
            t_min = np.percentile(raw_areas, 5)
            t_max = np.percentile(raw_areas, 95)
        else:
            t_min, t_max = 0, 100

        state.smooth_min += (t_min - state.smooth_min) * 0.1
        state.smooth_max += (t_max - state.smooth_max) * 0.1
        
        # 3. Render GPU Image
        render_frame(state.smooth_min, state.smooth_max, num_pts, 
                     int(cb_color.checked), int(cb_walls.checked), int(cb_dots.checked))
        
        img = screen_pixels.to_numpy().swapaxes(0, 1)
        img = (img * 255).astype(np.uint8)
        surf = pygame.transform.flip(pygame.transform.rotate(pygame.surfarray.make_surface(img), -90), True, False)
        
        # 4. Draw Image
        screen.blit(surf, (0, 0))
        
        # 5. Draw Path (CPU Overlay)
        if cb_path.checked:
            try:
                # Get raw float data from GPU
                raw_pts = points.to_numpy()[:num_pts]
                
                # Safety Check: Ensure we have enough points
                if len(raw_pts) > 1:
                    path_int = raw_pts.astype(np.int32)
                    pygame.draw.lines(screen, (0, 0, 0), False, path_int, 2)
            except Exception as e:
                print(f"Error drawing path: {e}")
        
        # 6. Draw UI
        title = title_font.render("Settings", True, (255, 255, 255))
        screen.blit(title, (ui_x, 15))
        
        # Draw pattern-specific slider
        for name, slider in pattern_sliders.items():
            if slider.visible:
                slider.draw(screen, font)
        
        # Draw common elements
        slider_samples.draw(screen, font)
        cb_color.draw(screen, font)
        cb_walls.draw(screen, font)
        cb_dots.draw(screen, font)
        cb_path.draw(screen, font)
        
        fps_surf = font.render(f"FPS: {clock.get_fps():.1f}", True, (100, 255, 100))
        screen.blit(fps_surf, (ui_x, WIN_HEIGHT - 40))
        
        # Draw dropdown LAST so expanded options appear on top
        dropdown_pattern.draw(screen, font)
        
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()