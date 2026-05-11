import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def create_3d_grid(shape=(100, 100, 100)):
    """
    Step 1: Creates a 3D coordinate grid.
    Returns:
        xv, yv, zv: 3D coordinate grids.
    """
    print(f"Generating 3D grid of shape {shape}...")
    # Using a normalized coordinate space from -1 to 1
    x = np.linspace(-1, 1, shape[0])
    y = np.linspace(-1, 1, shape[1])
    z = np.linspace(-1, 1, shape[2])
    
    # Create a 3D grid of coordinates (ij indexing ensures consistency with matrix axes)
    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
    return xv, yv, zv

def plant_seed(xv, yv, zv, center=(0, 0, 0), radius=0.3):
    """
    Step 2: Plants a simple spherical seed.
    Returns:
        seed_mask: A boolean mask representing the initial tumor.
        distance: The distance field from the center (useful for later deformations).
    """
    print(f"Planting seed at {center} with radius {radius}...")
    # Calculate Euclidean distance from the center for each point in the grid
    distance = np.sqrt((xv - center[0])**2 + (yv - center[1])**2 + (zv - center[2])**2)
    
    # Create a binary mask (1 inside the sphere, 0 outside)
    seed_mask = (distance <= radius).astype(float)
    return seed_mask, distance

def visualize_slice(volume, title="Slice", axis=2, slice_idx=None):
    """
    Visualizes a 2D slice of a 3D volume.
    """
    if slice_idx is None:
        slice_idx = volume.shape[axis] // 2
        
    if axis == 0:
        slice_data = volume[slice_idx, :, :]
    elif axis == 1:
        slice_data = volume[:, slice_idx, :]
    else:
        slice_data = volume[:, :, slice_idx]
        
    plt.figure(figsize=(6, 6))
    plt.imshow(slice_data, cmap='gray', origin='lower')
    plt.title(f"{title} (Axis {axis}, Slice {slice_idx})")
    plt.colorbar(label='Intensity')
    plt.axis('off')
    
    # Save the figure so we can verify it if run remotely, but also show it.
    plt.savefig(f"{title.replace(' ', '_').replace(':', '')}.png", bbox_inches='tight')
    plt.close()

def apply_perlin_wiggle(distance_field, radius=0.4, noise_scale=0.15, noise_smoothness=5.0):
    """
    Step 3: Applies a Perlin-like wiggle to deform the spherical seed into an organic shape.
    Returns:
        deformed_mask: Boolean mask of the deformed tumor.
        deformed_distance: The deformed distance field.
    """
    print(f"Applying Perlin wiggle (scale={noise_scale}, smoothness={noise_smoothness})...")
    
    # Generate random uniform noise
    random_noise = np.random.rand(*distance_field.shape)
    
    # Smooth the noise with a Gaussian filter to make it correlated (like Perlin noise)
    smoothed_noise = gaussian_filter(random_noise, sigma=noise_smoothness)
    
    # Normalize the noise to be centered around 0, between -1 and 1
    smoothed_noise = (smoothed_noise - np.min(smoothed_noise)) / (np.max(smoothed_noise) - np.min(smoothed_noise))
    smoothed_noise = smoothed_noise * 2.0 - 1.0
    
    # Deform the distance field by adding the noise
    deformed_distance = distance_field + smoothed_noise * noise_scale
    
    # Create the new mask based on the deformed distance
    deformed_mask = (deformed_distance <= radius).astype(float)
    return deformed_mask, deformed_distance

def simulate_infiltration(tumor_mask, blur_sigma=2.0):
    """
    Step 4: Simulates tumor infiltration into surrounding tissue by softening the edges.
    Returns:
        infiltrated_tumor: The continuous (non-binary) tumor representation.
    """
    print(f"Simulating infiltration with Gaussian blur (sigma={blur_sigma})...")
    infiltrated_tumor = gaussian_filter(tumor_mask, sigma=blur_sigma)
    return infiltrated_tumor

if __name__ == "__main__":
    # Define grid resolution
    grid_shape = (100, 100, 100)
    
    # Step 1: Generate 3D Grid
    xv, yv, zv = create_3d_grid(grid_shape)
    
    # Step 2: Plant a Seed
    seed_mask, distance_field = plant_seed(xv, yv, zv, radius=0.4)
    
    # Visualize the center slice
    print("Visualizing the initial seed...")
    visualize_slice(seed_mask, title="Step 2 - Initial Seed")
    
    # Step 3: Apply Perlin Wiggle
    deformed_mask, deformed_distance = apply_perlin_wiggle(distance_field, radius=0.4, noise_scale=0.15, noise_smoothness=5.0)
    
    # Visualize the deformed seed
    print("Visualizing the deformed seed...")
    visualize_slice(deformed_mask, title="Step 3 - Deformed Seed")
    
    # Step 4: Simulate Infiltration
    infiltrated_tumor = simulate_infiltration(deformed_mask, blur_sigma=2.0)
    
    # Visualize the infiltrated tumor
    print("Visualizing the infiltrated tumor...")
    visualize_slice(infiltrated_tumor, title="Step 4 - Infiltrated Tumor")
