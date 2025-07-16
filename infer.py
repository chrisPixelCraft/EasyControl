import torch
from PIL import Image
from src.pipeline import FluxPipeline
from src.transformer_flux import FluxTransformer2DModel
from src.lora_helper import set_single_lora, set_multi_lora

from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="Xiaojiu-Z/EasyControl", filename="models/canny.safetensors", local_dir="./")

def clear_cache(transformer):
    for name, attn_processor in transformer.attn_processors.items():
        attn_processor.bank_kv.clear()

# Clear GPU cache before loading models
torch.cuda.empty_cache()

# Initialize model with memory optimizations
device = "cuda"
base_path = "black-forest-labs/FLUX.1-dev"  # Path to your base model

# Load pipeline without immediately moving to GPU
pipe = FluxPipeline.from_pretrained(
    base_path,
    torch_dtype=torch.bfloat16,
    device_map=None  # Don't auto-move to GPU
)

# Load transformer separately with memory optimization
transformer = FluxTransformer2DModel.from_pretrained(
    base_path,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    device_map=None  # Don't auto-move to GPU
)

# Replace transformer and move to GPU in one step
pipe.transformer = transformer
pipe.to(device)

# Load control models
lora_path = "./models"
control_models = {
    "canny": f"{lora_path}/canny.safetensors",
    "depth": f"{lora_path}/depth.safetensors",
    "hedsketch": f"{lora_path}/hedsketch.safetensors",
    "pose": f"{lora_path}/pose.safetensors",
    "seg": f"{lora_path}/seg.safetensors",
    "inpainting": f"{lora_path}/inpainting.safetensors",
    "subject": f"{lora_path}/subject.safetensors",
}

# Single spatial condition control example
path = control_models["canny"]
set_single_lora(pipe.transformer, path, lora_weights=[1], cond_size=512)

# Generate image
prompt = "A nice car on the beach"

spatial_image = Image.open("./test_imgs/canny.png")

image = pipe(
    prompt,
    height=768,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=25,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(5),
    spatial_images=[spatial_image],
    subject_images=[],
    cond_size=512,
).images[0]

# Clear cache after generation
clear_cache(pipe.transformer)

image.save("output.png")
