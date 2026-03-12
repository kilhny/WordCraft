import os
import torch
from PIL import Image
from diffusers.pipelines import FluxPipeline

from src.flux.condition import Condition
from src.flux.generate import generate, seed_everything

def main():
    # ================= 1. 路径与基础配置 =================
    model_id = "black-forest-labs/FLUX.1-dev"
    lora_dir = "./weights" 
    lora_name = "pytorch_lora_weights.safetensors"
    
    # 文件夹/文件路径配置
    image_path = "./assets/input/sample.jpg"         # image_folder: 输入的原始图像
    mask_path = "./assets/masks/sample_mask.jpg"     # mask_folder: 连续编辑专用的 mask
    output_path = "./output/editing_result.jpg"      # output_folder: 结果输出路径
    
    # Noise 路径 (关键：加载上一步的 Noise，并可选择是否覆盖保存)
    load_noise_path = "./output/noise/sample_noise.pt" # 必须加载 Generation 生成的 Noise
    save_noise_path = None # 如果还要继续基于这张图编辑，可以设为 "./output/noise/sample_noise_edited.pt"
    
    seed = 42
    use_attention = False # 编辑阶段关闭多区域生成机制
    
    # 单图测试的提示词 (仅针对被 mask 选中的区域进行编辑)
    base_prompt = "Change the texture to glowing neon blue..."

    # ================= 2. 模型加载 =================
    print("Loading pipeline...")
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")
    pipe.load_lora_weights(lora_dir, weight_name=lora_name, adapter_name="depth")

    # ================= 3. 数据准备 =================
    print(f"Processing image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Editing 阶段必须加载 mask_image
    if os.path.exists(mask_path):
        mask_image = Image.open(mask_path).convert("L")
    else:
        raise FileNotFoundError(f"Editing requires a mask image, but {mask_path} was not found.")

    condition = Condition("depth", image)

    # ================= 4. 运行生成 (编辑) =================
    print(f"Editing image with seed {seed}...")
    seed_everything(seed)
    
    if save_noise_path:
        os.makedirs(os.path.dirname(save_noise_path), exist_ok=True)

    result = generate(
        pipe,
        height=height, width=width,
        prompt=base_prompt,
        mask_image=mask_image,         # <--- 传入局部编辑的 Mask
        use_attention=use_attention,   # False
        conditions=[condition],
        mask_inject_steps=10,          # 可根据编辑强度的需求调整
        layers_list=list(range(57)),
        joint_attention_kwargs=None,   # 关闭 attention 时传入 None 即可
        load_noise_path=load_noise_path, # <--- 注入原始 Noise 以保持结构稳定
        save_noise_path=save_noise_path  # <--- 可选保存新的 Noise
    )

    # ================= 5. 保存结果 =================
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.images[0].save(output_path)
    print(f"编辑完成！图像已保存至: {output_path}")

if __name__ == "__main__":
    main()
