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
    lora_dir = "./loras" 
    lora_name = "pytorch_lora_weights.safetensors"
    
    # Folder/File path configurations based on your directory
    image_path = "./assets/test/cat.jpg"
    output_path = "./output/cat_generation_result.jpg"
    
    seed = 42
    use_attention = True # Enable multi-regional generation
    
    # Noise save path (Saving to your cat_noise folder)
    save_noise_path = f"./assets/test/cat_noise/{seed}" 
    
    # Prompts based on your "cat" JSON configuration
    base_prompt = "Playful cat motif combining fluffy tail, and round face"
    background_prompt = {"description": "Warm white background"} 
    
    # Notice how the mask paths/boxes are embedded directly in the dict
    regional_prompt = {
        "0": {
            "description": "Fluffy playful tail",
            "mask": "./assets/test/cat_region_mask.jpg",
        },
        "1": {
            "description": "Round kitten face",
            "mask": [201, 260, 442, 451],
        }
    }

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

    # Prepare regional attention control (Passing None for global region_mask as it is in the dict now)
    joint_attention_kwargs = prepare_regional_control(
        regional_prompt, width, height, background_prompt, None
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
        save_noise_path=save_noise_path  # <--- Save Noise for Editing
    )

    # ================= 5. Save Results =================
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.images[0].save(output_path)
    print(f"Generation complete! Image saved to: {output_path}")
    print(f"Noise successfully saved to: {save_noise_path}")

if __name__ == "__main__":
    main()
