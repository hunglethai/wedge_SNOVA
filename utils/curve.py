from sage.all import *
import math
import os
import imageio
import random as pyrandom  
from tqdm import tqdm

# Generate the combined plot
t = var('t')
x = t
y = t**2
z = t**3

plot1 = parametric_plot3d((x, y, z), (t, -2, 2), color='blue', thickness=4)

# Use Python random instead of Sage's conflicting "random"
points = [(a, a**2, a**3) for a in [pyrandom.uniform(-2, 2) for _ in range(100)]]
plot2 = point3d(points, color='red', size=10)
combined_plot = plot1 + plot2

# Create frames directory
frame_dir = "frames"
os.makedirs(frame_dir, exist_ok=True)

# Save rotated frames
filenames = []
for i, angle_deg in tqdm(enumerate(range(0, 360, 5)),ncols = 100,desc = "Generating frames"):
    angle_rad = math.radians(angle_deg)
    rotated = combined_plot.rotate((0, 0, 1), angle_rad)
    filename = os.path.join(frame_dir, f"frame_{i:03d}.png")
    rotated.save(filename, dpi=150)
    filenames.append(filename)

# Create GIF using imageio
gif_filename = "twisted_curve_rotation.gif"
with imageio.get_writer(gif_filename, mode='I', duration=0.05) as writer:
    for fname in filenames:
        image = imageio.imread(fname)
        writer.append_data(image)

print(f"✅ GIF saved as: {gif_filename}")
