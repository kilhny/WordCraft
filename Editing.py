import os
import torch
from PIL import Image
from diffusers.pipelines import FluxPipeline

from src.flux.condition import Condition
from src.flux.generate import generate, seed_everything

def main():
    # ================= 1. Paths & Basic Configurations =================
    model_id = "black-forest-labs/FLUX.1-dev"
    lora_dir = "./weights" 
    lora_name = "pytorch_lora_weights.safetensors"
    
    # Folder/File path configurations
    image_path = "./assets/input/sample.jpg"         # image_folder: Input original image
    mask_path = "./assets/masks/sample_mask.jpg"     # mask_folder: Mask dedicated for continuous editing
    output_path = "./output/editing_result.jpg"      # output_folder: Output path for the result
    
    # Noise paths (Crucial: Load noise from the previous step, and optionally save)
    load_noise_path = "./output/noise/sample_noise.pt" # Must load the Noise generated from the Generation phase
    save_noise_path = None # Set to a path if you want to continue editing based on this result
    
    seed = 42
    use_attention = False # Disable multi-regional generation mechanism during the editing phase
    
    # Prompts for single image testing (Applies only to the region selected by the mask)
    base_prompt = "Change the texture to glowing neon blue..."

    # ================= 2. Model Loading =================
    print("Loading pipeline...")
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")
    pipe.load_lora_weights(lora_dir, weight_name=lora_name, adapter_name="depth")

    # ================= 3. Data Preparation =================
    print(f"Processing image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # mask_image must be loaded during the Editing phase
    if os.path.exists(mask_path):
        mask_image = Image.open(mask_path).convert("L")
    else:
        raise FileNotFoundError(f"Editing requires a mask image, but {mask_path} was not found.")

    condition = Condition("depth", image)

    # ================= 4. Run Generation (Editing) =================
    print(f"Editing image with seed {seed}...")
    seed_everything(seed)
    
    if save_noise_path:
        os.makedirs(os.path.dirname(save_noise_path), exist_ok=True)

    result = generate(
        pipe,
        height=height, width=width,
        prompt=base_prompt,
        mask_image=mask_image,         # <--- Pass the mask for local editing
        use_attention=use_attention,   # False
        conditions=[condition],
        mask_inject_steps=10,          # Adjust according to the required editing strength
        layers_list=list(range(57)),
        joint_attention_kwargs=None,   # Pass None when attention is disabled
        load_noise_path=load_noise_path, # <--- Inject original Noise to maintain structural stability
        save_noise_path=save_noise_path  # <--- Optional: Save the new Noise
    )

    # ================= 5. Save Results =================
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.images[0].save(output_path)
    print(f"Editing complete! Image saved to: {output_path}")

if __name__ == "__main__":
    main()
