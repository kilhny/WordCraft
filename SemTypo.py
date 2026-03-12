import os
import torch
from PIL import Image
from diffusers.pipelines import FluxPipeline

from src.flux.condition import Condition
from src.flux.generate import generate, seed_everything
from src.flux.regional import prepare_regional_control

def main():
    # ================= 1. Paths & Basic Configurations =================
    model_id = "black-forest-labs/FLUX.1-dev"
    lora_dir = "./weights" 
    lora_name = "pytorch_lora_weights.safetensors"
    
    # Folder/File path configurations
    image_path = "./assets/input/sample.jpg"          # image_folder: Input original image
    output_path = "./output/generation_result.jpg"    # output_folder: Output path for the result
    
    # Noise save path (Crucial: For subsequent Continuous Editing)
    save_noise_path = "./output/noise/sample_noise.pt" # Set to None if you do not want to save
    
    seed = 42
    use_attention = True # Enable multi-regional generation
    
    # Prompts for single image testing (Replaces JSON used in batch processing)
    base_prompt = "A beautiful artistic typography design..."
    background_prompt = {"description": "A clean, dark background"} 
    regional_prompt = {"region_1": "Fiery metallic texture, glowing edges"}
    region_mask_path = "./assets/region_mask/sample"  # region_mask: Folder for auxiliary regional control

    # ================= 2. Model Loading =================
    print("Loading pipeline...")
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")
    pipe.load_lora_weights(lora_dir, weight_name=lora_name, adapter_name="depth")

    # ================= 3. Data Preparation =================
    print(f"Processing image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Mask image is not used in the Generation phase
    mask_image = None 

    # Prepare regional attention control
    joint_attention_kwargs = prepare_regional_control(
        regional_prompt, width, height, background_prompt, region_mask_path
    )
    condition = Condition("depth", image)

    # ================= 4. Run Generation =================
    print(f"Generating image with seed {seed}...")
    seed_everything(seed)
    
    # Ensure the directory for saving noise exists
    if save_noise_path:
        os.makedirs(os.path.dirname(save_noise_path), exist_ok=True)

    result = generate(
        pipe,
        height=height, width=width,
        prompt=base_prompt,
        mask_image=mask_image,
        use_attention=use_attention,
        conditions=[condition],
        mask_inject_steps=10,
        layers_list=list(range(57)),
        joint_attention_kwargs=joint_attention_kwargs,
        save_noise_path=save_noise_path  # <--- Save Noise
    )

    # ================= 5. Save Results =================
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.images[0].save(output_path)
    print(f"Generation complete! Image saved to: {output_path}")
    if save_noise_path:
        print(f"Noise saved to: {save_noise_path}")

if __name__ == "__main__":
    main()
