import torch
import yaml, os
from transformers import T5Tokenizer
from typing import List
from diffusers.pipelines import FluxPipeline
from typing import List, Union, Optional, Dict, Any, Callable
from .transformer import tranformer_forward
from .condition import Condition
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
import torch.nn.functional as F
from diffusers.pipelines.flux.pipeline_flux import (
    FluxPipelineOutput,
    calculate_shift,
    retrieve_timesteps,
    np,
)
from .regional import RegionalProcessor
from .regional import CudaTimer

def save_noise_pred(noise_pred, step, save_dir="noise_preds"):
    """
    保存每一步的 noise_pred。
    :param noise_pred: 需要存储的噪声张量。
    :param step: 当前步数。
    :param save_dir: 存储目录。
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"noise_pred_{step}.pt")
    torch.save(noise_pred, save_path)
    print(f"Saved noise_pred at step {step} to {save_path}")

def load_noise_pred(step, save_dir="noise_preds"):
    """
    加载存储的 noise_pred。
    :param step: 需要加载的步数。
    :param save_dir: 存储目录。
    :return: 加载的 noise_pred 张量。
    """
    load_path = os.path.join(save_dir, f"noise_pred_{step}.pt")
    if os.path.exists(load_path):
        noise_pred = torch.load(load_path)
        print(f"Loaded noise_pred from {load_path}")
        return noise_pred
    else:
        print(f"No noise_pred found at {load_path}")
        return None



def get_config(config_path: str = None):
    config_path = config_path or os.environ.get("XFL_CONFIG")
    if not config_path:
        return {}
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def prepare_params(
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = 512,
    width: Optional[int] = 512,
    num_inference_steps: int = 28,
    timesteps: List[int] = None,
    guidance_scale: float = 3.5,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    **kwargs: dict,
):
    return (
        prompt,
        prompt_2,
        height,
        width,
        num_inference_steps,
        timesteps,
        guidance_scale,
        num_images_per_prompt,
        generator,
        latents,
        prompt_embeds,
        pooled_prompt_embeds,
        output_type,
        return_dict,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs,
        max_sequence_length,
    )


def seed_everything(seed: int = 42):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def generate(
    pipeline: FluxPipeline,
    conditions: List[Condition] = None,
    config_path: str = None,
    model_config: Optional[Dict[str, Any]] = {},
    condition_scale: float = 1.0,
    default_lora: bool = False,
    mask_image: PipelineImageInput = None,
    use_attention: bool = True,
    load_noise_path: str = None,
    save_noise_path: str = None,
    padding_mask_crop: Optional[int] = None,
    mask_inject_steps: int = 5,
    layers_list: List[int] = list[range(57)],
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    **params: dict,
):
    model_config = model_config or get_config(config_path).get("model", {})
    if condition_scale != 1:
        for name, module in pipeline.transformer.named_modules():
            if not name.endswith(".attn"):
                continue
            module.c_factor = torch.ones(1, 1) * condition_scale

    self = pipeline
    (
        prompt,
        prompt_2,
        height,
        width,
        num_inference_steps,
        timesteps,
        guidance_scale,
        num_images_per_prompt,
        generator,
        latents,
        prompt_embeds,
        pooled_prompt_embeds,
        output_type,
        return_dict,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs,
        max_sequence_length,
    ) = prepare_params(**params)

    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    latent_channels = self.vae.config.latent_channels if getattr(self, "vae", None) else 16
    mask_processor = VaeImageProcessor(
        vae_scale_factor=self.vae_scale_factor * 2,
        vae_latent_channels=latent_channels,
        do_normalize=False,
        do_binarize=True,
        do_convert_grayscale=True,
    )

    def prepare_mask_latents(
            mask,
            batch_size,
            num_channels_latents,
            num_images_per_prompt,
            height,
            width,
            dtype,
            device,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(mask, size=(height, width))
        mask = mask.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)

        mask = self._pack_latents(
            mask.repeat(1, num_channels_latents, 1, 1),
            batch_size,
            num_channels_latents,
            height,
            width,
        )

        return mask

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    lora_scale = (
        self.joint_attention_kwargs.get("scale", None)
        if self.joint_attention_kwargs is not None
        else None
    )
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )
    if use_attention :
      #3. Prepare attention mask
      # Initialize `RegionalProcessor`
      regional_processor = RegionalProcessor(height, width, self.vae_scale_factor, device)
      # Encode regional prompts
      regional_inputs = regional_processor.encode_regional_prompts(joint_attention_kwargs, self.encode_prompt,num_images_per_prompt)
      # Create regional attention mask
      regional_attention_mask, regional_embeds, encoder_seq_len = regional_processor.create_attention_mask(regional_inputs)
    else:
      regional_embeds = prompt_embeds
      
    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels // 4
    latents, latent_image_ids = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    latents_old = latents

    # 4.1. Prepare conditions
    condition_latents, condition_ids, condition_type_ids = ([] for _ in range(3))
    use_condition = conditions is not None or []
    if use_condition:
        assert len(conditions) <= 1, "Only one condition is supported for now."
        if not default_lora:
            pipeline.set_adapters(conditions[0].condition_type)
        for condition in conditions:
            tokens, ids, type_id = condition.encode(self)
            condition_latents.append(tokens)  # [batch_size, token_n, token_dim]
            condition_ids.append(ids)  # [token_n, id_dim(3)]
            condition_type_ids.append(type_id)  # [token_n, 1]
        condition_latents = torch.cat(condition_latents, dim=1)
        condition_ids = torch.cat(condition_ids, dim=0)
        condition_type_ids = torch.cat(condition_type_ids, dim=0)
    #4.2 Prepare mask
    if load_noise_path:
      if padding_mask_crop is not None:
          crops_coords = mask_processor.get_crop_region(mask_image, width, height, pad=padding_mask_crop)
          resize_mode = "fill"
      else:
          crops_coords = None
          resize_mode = "default"
      print(type(mask_image))
      mask_condition = mask_processor.preprocess(
          mask_image, height=height, width=width, resize_mode=resize_mode, crops_coords=crops_coords
      )
      mask = prepare_mask_latents(
          mask_condition,
          batch_size,
          num_channels_latents,
          num_images_per_prompt,
          height,
          width,
          prompt_embeds.dtype,
          device,
      )

    # 5. Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.base_image_seq_len,
        self.scheduler.config.max_image_seq_len,
        self.scheduler.config.base_shift,
        self.scheduler.config.max_shift,
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        timesteps,
        sigmas,
        mu=mu,
    )
    num_warmup_steps = max(
        len(timesteps) - num_inference_steps * self.scheduler.order, 0
    )
    self._num_timesteps = len(timesteps)

    # 6. Denoising loop
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            # handle guidance
            if self.transformer.config.guidance_embeds:
                guidance = torch.tensor([guidance_scale], device=device)
                guidance = guidance.expand(latents.shape[0])
            else:
                guidance = None

            noise_pred = tranformer_forward(
                self.transformer,
                model_config=model_config,
                # Inputs of the condition (new feature)
                condition_latents=condition_latents if use_condition else None,
                condition_ids=condition_ids if use_condition else None,
                condition_type_ids=condition_type_ids if use_condition else None,
                # Inputs to the original transformer
                hidden_states=latents,
                # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=regional_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs={
                    'single_inject_blocks_interval': joint_attention_kwargs['single_inject_blocks_interval'] if 'single_inject_blocks_interval' in joint_attention_kwargs else len(self.transformer.single_transformer_blocks),
                    'double_inject_blocks_interval': joint_attention_kwargs['double_inject_blocks_interval'] if 'double_inject_blocks_interval' in joint_attention_kwargs else len(self.transformer.transformer_blocks),
                    'regional_attention_mask': regional_attention_mask,
                    'layers_list': layers_list,
                } if use_attention else None,
                return_dict=False,
            )[0]
            if load_noise_path :
                noise_pred_old = load_noise_pred(i, os.path.join("noise_preds", load_noise_path))
                #noise_pred = noise_pred * mask #ce shi
                with CudaTimer("Noise Blending"):
                  noise_pred = noise_pred * mask + noise_pred_old * (1. - mask)
            if save_noise_path :
                save_noise_pred(noise_pred, i, os.path.join("noise_preds", save_noise_path))

            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()

    if output_type == "latent":
        image = latents

    else:
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (
            latents / self.vae.config.scaling_factor
        ) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    if condition_scale != 1:
        for name, module in pipeline.transformer.named_modules():
            if not name.endswith(".attn"):
                continue
            del module.c_factor

    if not return_dict:
        return (image,)

    CudaTimer.print_stats()
    
    return FluxPipelineOutput(images=image)
