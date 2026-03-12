import os
import torch
from PIL import Image
from diffusers.pipelines import FluxPipeline

from models.condition import Condition
from models.generate import generate, seed_everything

def main():
    # ================= 1. Paths & Basic Configurations =================
    model_id = "black-forest-labs/FLUX.1-dev"
    lora_dir = "./loras"
    lora_name = "pytorch_lora_weights.safetensors"

    # Folder/File path configurations based on your directory
    image_path = "./assets/test/cat.jpg"
    mask_path = "./assets/test/cat_mask.jpg"     # Mask dedicated for continuous editing
    output_path = "./output/cat_editing_result.jpg"

    seed = 42
    use_attention = False # Disable multi-regional generation mechanism

    # Noise paths (Loading the noise generated from Scenario 1)
    load_noise_path = f"./assets/test/cat_noise/{seed}"
    save_noise_path = None # Set to a path if you want to save it again

    # Prompt for single image editing based on your config
    prompt = "A Snapback cap"

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

    result = generate(
        pipe,
        height=height, width=width,
        prompt=prompt,
        mask_image=mask_image,           # <--- Pass the cat_mask for local editing
        use_attention=use_attention,     # False
        conditions=[condition],
        mask_inject_steps=10,
        layers_list=list(range(57)),
        joint_attention_kwargs=None,
        load_noise_path=load_noise_path, # <--- Inject cat_noise/noise.pt to maintain structure
        save_noise_path=save_noise_path
    )

    # ================= 5. Save Results =================
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.images[0].save(output_path)
    print(f"Editing complete! Image saved to: {output_path}")

if __name__ == "__main__":
    main()
