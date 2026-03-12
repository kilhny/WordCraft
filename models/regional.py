import torch
from PIL import Image
import os
import numpy as np
from typing import List, Union, Optional, Dict, Any, Callable
from diffusers.models.attention_processor import Attention, F
from typing import Dict, List, Optional, Union
from .lora_controller import enable_lora

def attn_forward_condition(
    attn: Attention,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor = None,
    condition_latents: torch.FloatTensor = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
    cond_rotary_emb: Optional[torch.Tensor] = None,
    model_config: Optional[Dict[str, Any]] = {},
) -> torch.FloatTensor:
    batch_size, _, _ = (
        hidden_states.shape
        if encoder_hidden_states is None
        else encoder_hidden_states.shape
    )

    with enable_lora(
        (attn.to_q, attn.to_k, attn.to_v), model_config.get("latent_lora", False)
    ):
        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
    if encoder_hidden_states is not None:
        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(
                encoder_hidden_states_query_proj
            )
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(
                encoder_hidden_states_key_proj
            )

        # attention
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

    if image_rotary_emb is not None:
        from diffusers.models.embeddings import apply_rotary_emb

        query = apply_rotary_emb(query, image_rotary_emb)
        key = apply_rotary_emb(key, image_rotary_emb)


    if condition_latents is not None:
        cond_query = attn.to_q(condition_latents)
        cond_key = attn.to_k(condition_latents)
        cond_value = attn.to_v(condition_latents)

        cond_query = cond_query.view(batch_size, -1, attn.heads, head_dim).transpose(
            1, 2
        )
        cond_key = cond_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        cond_value = cond_value.view(batch_size, -1, attn.heads, head_dim).transpose(
            1, 2
        )
        if attn.norm_q is not None:
            cond_query = attn.norm_q(cond_query)
        if attn.norm_k is not None:
            cond_key = attn.norm_k(cond_key)
    if cond_rotary_emb is not None:
        cond_query = apply_rotary_emb(cond_query, cond_rotary_emb)
        cond_key = apply_rotary_emb(cond_key, cond_rotary_emb)

    if condition_latents is not None:
        query = torch.cat([query, cond_query], dim=2)
        key = torch.cat([key, cond_key], dim=2)
        value = torch.cat([value, cond_value], dim=2)

    if not model_config.get("union_cond_attn", True):
        # If we don't want to use the union condition attention, we need to mask the attention
        # between the hidden states and the condition latents
        attention_mask = torch.ones(
            query.shape[2], key.shape[2], device=query.device, dtype=torch.bool
        )
        condition_n = cond_query.shape[2]
        attention_mask[-condition_n:, :-condition_n] = False
        attention_mask[:-condition_n, -condition_n:] = False

    if attention_mask is not None:
        attn_mask = torch.ones(
            query.shape[2], key.shape[2], device=query.device, dtype=query.dtype
        )
        condition_n = cond_query.shape[2]
        attn_mask[:-condition_n , :-condition_n] = attention_mask
        q_t = query.shape[2] - (2 * condition_n) 
        attn_mask[-condition_n:,:q_t] = False
        attn_mask[:q_t, -condition_n:] = False
        attention_mask = attn_mask

    if hasattr(attn, "c_factor"):
        bias = torch.log(attn.c_factor[0])
        condition_n = cond_query.shape[2]
        q_t = query.shape[2] - (2 * condition_n)
        attention_mask[-condition_n:, q_t:-condition_n] = bias
        attention_mask[q_t:-condition_n, -condition_n:] = bias

    hidden_states = F.scaled_dot_product_attention(
        query, key, value, dropout_p=0.0, is_causal=False, attn_mask=attention_mask
    )
    hidden_states = hidden_states.transpose(1, 2).reshape(
        batch_size, -1, attn.heads * head_dim
    )
    hidden_states = hidden_states.to(query.dtype)

    if encoder_hidden_states is not None:
        if condition_latents is not None:
            encoder_hidden_states, hidden_states, condition_latents = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[
                    :, encoder_hidden_states.shape[1] : -condition_latents.shape[1]
                ],
                hidden_states[:, -condition_latents.shape[1] :],
            )
        else:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

        with enable_lora((attn.to_out[0],), model_config.get("latent_lora", False)):
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if condition_latents is not None:
            condition_latents = attn.to_out[0](condition_latents)
            condition_latents = attn.to_out[1](condition_latents)

        return (
            (hidden_states, encoder_hidden_states, condition_latents)
            if condition_latents is not None
            else (hidden_states, encoder_hidden_states)
        )
    elif condition_latents is not None:
        # if there are condition_latents, we need to separate the hidden_states and the condition_latents
        hidden_states, condition_latents = (
            hidden_states[:, : -condition_latents.shape[1]],
            hidden_states[:, -condition_latents.shape[1] :],
        )
        return hidden_states, condition_latents
    else:
        return hidden_states

def prepare_regional_control(
    regional_prompt_mask_pairs: dict,
    image_width: int,
    image_height: int,
    background_prompt: str = None,
    base_dir: str = None,
    double_inject_blocks_interval: int = 1,
    single_inject_blocks_interval: int = 1,
) -> dict:
    """
    准备区域性提示 (regional_prompts) 和掩码 (regional_masks) 以供 Flux Pipeline 使用。

    Args:
        regional_prompt_mask_pairs (dict): 区域提示与掩码的配对字典
        image_width (int): 图像宽度
        image_height (int): 图像高度
        background_prompt (str): 背景提示词
        base_ratio (float): 基础图像占比
        double_inject_blocks_interval (int): 双注入块的间隔
        single_inject_blocks_interval (int): 单注入块的间隔

    Returns:
        dict: joint_attention_kwargs，传递给 pipeline 的参数
    """
    regional_prompts = []
    regional_masks = []
    background_mask = torch.ones((image_height, image_width))

    for region in regional_prompt_mask_pairs.values():
        description = region['description']
        mask_data = region['mask']
        # 支持两种 mask：矩形坐标或图像路径
        if isinstance(mask_data, (list, tuple)):
            # 原来的方式：矩形 mask
            x1, y1, x2, y2 = mask_data
            mask = torch.zeros((image_height, image_width))
            mask[y1:y2, x1:x2] = 1.0
        elif isinstance(mask_data, str):
            # 新方式：从图片路径读取 mask
            mask_data = os.path.join(base_dir, mask_data)
            #mask_data = os.path.splitext(mask_data)[0] + ".jpg"
            mask_img = Image.open(mask_data).convert("L").resize((image_width, image_height))
            mask = torch.tensor(np.array(mask_img), dtype=torch.float32) / 255.0
            mask = mask.clamp(0, 1)
        else:
            raise ValueError("mask 必须是坐标列表或图像路径字符串")
        
        background_mask -= mask

        regional_prompts.append(description)
        regional_masks.append(mask)

    if background_mask.sum() > 0:
        regional_prompts.append(background_prompt)
        regional_masks.append(background_mask)

    joint_attention_kwargs = {
        'regional_prompts': regional_prompts,
        'regional_masks': regional_masks,
        'double_inject_blocks_interval': double_inject_blocks_interval,
        'single_inject_blocks_interval': single_inject_blocks_interval,
    }

    return joint_attention_kwargs

class RegionalProcessor:
    def __init__(self, height, width, vae_scale_factor, device):
        self.height = height
        self.width = width
        self.vae_scale_factor = vae_scale_factor
        self.device = device

    def encode_regional_prompts(self, joint_attention_kwargs, encode_prompt_fn, num_images_per_prompt):
        """
        解析 `regional_prompts` 和 `regional_masks`，生成 `regional_inputs`
        """
        regional_inputs = []
        if 'regional_prompts' in joint_attention_kwargs and 'regional_masks' in joint_attention_kwargs:
            for regional_prompt, regional_mask in zip(joint_attention_kwargs['regional_prompts'], joint_attention_kwargs['regional_masks']):
                regional_prompt_embeds,regional_pooled_prompt_embeds,regional_text_ids = encode_prompt_fn(
                    prompt=regional_prompt,
                    prompt_2=regional_prompt,
                    prompt_embeds=None,
                    pooled_prompt_embeds=None,
                    device=self.device,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=512,
                    lora_scale=None,
                )
                regional_inputs.append((regional_mask, regional_prompt_embeds))

        return regional_inputs

    def create_attention_mask(self, regional_inputs):
        """
        生成 `regional_attention_mask` 控制 Prompt 作用范围
        """
        H, W = self.height // self.vae_scale_factor, self.width // self.vae_scale_factor
        H, W = H // 2, W // 2
        hidden_seq_len = H * W

        conds = []
        masks = []
        for mask, cond in regional_inputs:
            if mask is not None:  # Resize masks to match image size
                mask = F.interpolate(mask[None, None, :, :], (H, W), mode='nearest-exact').flatten().unsqueeze(1).repeat(1, cond.size(1))
            else:
                mask = torch.ones((H * W, cond.size(1)), device=self.device)
            masks.append(mask)
            conds.append(cond)

        regional_embeds = torch.cat(conds, dim=1)
        encoder_seq_len = regional_embeds.shape[1]

        # Initialize the attention mask
        regional_attention_mask = torch.zeros(
            (encoder_seq_len + hidden_seq_len, encoder_seq_len + hidden_seq_len),
            device=masks[0].device,
            dtype=torch.bool
        )

        num_of_regions = len(masks)
        each_prompt_seq_len = encoder_seq_len // num_of_regions

        # Initialize self-attend mask
        self_attend_masks = torch.zeros((hidden_seq_len, hidden_seq_len), device=masks[0].device, dtype=torch.bool)

        # Initialize union mask
        union_masks = torch.zeros((hidden_seq_len, hidden_seq_len), device=masks[0].device, dtype=torch.bool)

        for i in range(num_of_regions):
            # Text attends to itself
            regional_attention_mask[i * each_prompt_seq_len:(i + 1) * each_prompt_seq_len,
                                    i * each_prompt_seq_len:(i + 1) * each_prompt_seq_len] = True

            # Text attends to corresponding regional image
            regional_attention_mask[i * each_prompt_seq_len:(i + 1) * each_prompt_seq_len, encoder_seq_len:] = masks[i].transpose(-1, -2)

            # Regional image attends to corresponding text
            regional_attention_mask[encoder_seq_len:, i * each_prompt_seq_len:(i + 1) * each_prompt_seq_len] = masks[i]

            # Regional image attends to corresponding regional image
            img_size_masks = masks[i][:, :1].repeat(1, hidden_seq_len)
            img_size_masks_transpose = img_size_masks.transpose(-1, -2)
            self_attend_masks = torch.logical_or(self_attend_masks,
                                                 torch.logical_and(img_size_masks, img_size_masks_transpose))

            # Update union
            union_masks = torch.logical_or(union_masks,
                                           torch.logical_or(img_size_masks, img_size_masks_transpose))

        background_masks = torch.logical_not(union_masks)

        # Finalize the mask
        background_and_self_attend_masks = torch.logical_or(background_masks, self_attend_masks)

        regional_attention_mask[encoder_seq_len:, encoder_seq_len:] = background_and_self_attend_masks

        return regional_attention_mask, regional_embeds, encoder_seq_len

