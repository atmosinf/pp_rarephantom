import numpy as np
import matplotlib.pyplot as plt
from rare_phantom import create_3d_grid, plant_seed, apply_perlin_wiggle, simulate_infiltration, visualize_3d

def generate_gallery():
    print("🎨 Generating a gallery of different tumor morphologies...")
    grid_shape = (100, 100, 100)
    xv, yv, zv = create_3d_grid(grid_shape)

    # Define 4 different "Clinical Cases"
    cases = [
        {
            "name": "Case_A_Small_Smooth",
            "radius": 0.25,
            "scale": 0.05,
            "smoothness": 10.0,
            "blur": 1.0
        },
        {
            "name": "Case_B_Large_Lobulated",
            "radius": 0.5,
            "scale": 0.2,
            "smoothness": 8.0,
            "blur": 2.0
        },
        {
            "name": "Case_C_Irregular_Aggressive",
            "radius": 0.35,
            "scale": 0.25,
            "smoothness": 4.0,
            "blur": 1.5
        },
        {
            "name": "Case_D_Diffuse_Infiltrative",
            "radius": 0.4,
            "scale": 0.1,
            "smoothness": 6.0,
            "blur": 4.5
        }
    ]

    for case in cases:
        print(f"\nProcessing {case['name']}...")
        # 1. Seed
        mask, dist = plant_seed(xv, yv, zv, radius=case['radius'])
        
        # 2. Wiggle
        deformed_mask, _ = apply_perlin_wiggle(
            dist, 
            radius=case['radius'], 
            noise_scale=case['scale'], 
            noise_smoothness=case['smoothness']
        )
        
        # 3. Infiltrate
        final_volume = simulate_infiltration(deformed_mask, blur_sigma=case['blur'])
        
        # 4. Render 3D
        visualize_3d(final_volume, title=f"Gallery - {case['name']}")

    print("\n✅ Gallery complete! Check your folder for the 'Gallery - Case_*.png' files.")

if __name__ == "__main__":
    generate_gallery()
