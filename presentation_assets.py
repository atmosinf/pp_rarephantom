import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from rare_phantom import (
    create_3d_grid, 
    plant_seed, 
    apply_perlin_wiggle, 
    simulate_infiltration, 
    visualize_slice, 
    visualize_3d
)

def create_presentation_slides():
    print("🚀 Generating high-quality assets for your presentation...")
    grid_shape = (100, 100, 100)
    mid = grid_shape[0] // 2

    # --- STEP 1: The Coordinate Grid ---
    xv, yv, zv = create_3d_grid(grid_shape)
    plt.figure(figsize=(12, 4))
    plt.subplot(131); plt.imshow(xv[mid,:,:], cmap='RdBu'); plt.title("X-Map (-1 to 1)")
    plt.subplot(132); plt.imshow(yv[:,mid,:], cmap='RdBu'); plt.title("Y-Map (-1 to 1)")
    plt.subplot(133); plt.imshow(zv[:,:,mid], cmap='RdBu'); plt.title("Z-Map (-1 to 1)")
    plt.suptitle("Step 1: Establishing the 3D Coordinate Space", fontsize=16)
    plt.tight_layout()
    plt.savefig("Slide_1_Coordinate_Grid.png", dpi=150)
    plt.close()

    # --- STEP 2: The Initial Seed ---
    seed_mask, distance_field = plant_seed(xv, yv, zv, radius=0.4)
    plt.figure(figsize=(10, 5))
    plt.subplot(121); plt.imshow(distance_field[mid,:,:], cmap='viridis'); plt.title("Distance Field")
    plt.subplot(122); plt.imshow(seed_mask[mid,:,:], cmap='gray'); plt.title("Binary Mask")
    plt.suptitle("Step 2: Planting the Spherical Seed", fontsize=16)
    plt.savefig("Slide_2_Initial_Seed.png", dpi=150)
    plt.close()

    # --- STEP 3: Noise Logic (Crucial for explaining the "wiggle") ---
    raw_noise = np.random.rand(*grid_shape)
    smooth_noise = gaussian_filter(raw_noise, sigma=5.0)
    # Normalize for visualization
    smooth_noise = (smooth_noise - np.min(smooth_noise)) / (np.max(smooth_noise) - np.min(smooth_noise))
    
    plt.figure(figsize=(10, 5))
    plt.subplot(121); plt.imshow(raw_noise[mid,:,:], cmap='gray'); plt.title("Raw Random Static")
    plt.subplot(122); plt.imshow(smooth_noise[mid,:,:], cmap='magma'); plt.title("Smoothed Organic Noise")
    plt.suptitle("Step 3A: Converting Randomness to Organic Patterns", fontsize=16)
    plt.savefig("Slide_3A_Noise_Logic.png", dpi=150)
    plt.close()

    # --- STEP 3B: The Resulting Deformed Shape ---
    deformed_mask, _ = apply_perlin_wiggle(distance_field, radius=0.4)
    visualize_slice(deformed_mask, title="Step 3B - Deformed Organic Shape")
    # This saves as 'Step_3B_-_Deformed_Organic_Shape.png' via the original function

    # --- STEP 4: Infiltration Logic ---
    infiltrated = simulate_infiltration(deformed_mask, blur_sigma=2.0)
    plt.figure(figsize=(10, 5))
    plt.subplot(121); plt.imshow(deformed_mask[mid,:,:], cmap='gray'); plt.title("Harsh Edge (Artificial)")
    plt.subplot(122); plt.imshow(infiltrated[mid,:,:], cmap='hot'); plt.title("Soft Edge (Infiltrative)")
    plt.suptitle("Step 4: Simulating Biological Infiltration", fontsize=16)
    plt.savefig("Slide_4_Infiltration_Effect.png", dpi=150)
    plt.close()

    # --- STEP 5: Final 3D View ---
    visualize_3d(infiltrated, title="Step 5 - Final 3D Phantom")

    print("\n✅ All presentation assets created!")
    print("Files to use in your slides:")
    print("- Slide_1_Coordinate_Grid.png")
    print("- Slide_2_Initial_Seed.png")
    print("- Slide_3A_Noise_Logic.png")
    print("- Step_3B_-_Deformed_Organic_Shape.png")
    print("- Slide_4_Infiltration_Effect.png")
    print("- Slide_5_-_Final_3D_Phantom.png")

if __name__ == "__main__":
    create_presentation_slides()
