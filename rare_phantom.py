import numpy as np
import matplotlib.pyplot as plt

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
    plt.show()

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
