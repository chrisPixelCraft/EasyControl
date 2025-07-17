import spaces
import os
import json
import time

# Fix for cuDNN Frontend error with CLIP model
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "0"
os.environ["PYTORCH_DISABLE_CUDNN_SDPA"] = "1"

import torch
from PIL import Image
from tqdm import tqdm
import gradio as gr

from safetensors.torch import save_file
from src.core.pipeline import FluxPipeline
from src.core.transformer_flux import FluxTransformer2DModel
from src.core.lora_helper import set_single_lora, set_multi_lora, unset_lora

# Optional: Apply attention fallback patch for additional safety
try:
    from src.attention_fix import patch_transformers_clip_attention
    patch_transformers_clip_attention()
    print("Applied CLIP attention fallback patch")
except ImportError:
    print("Attention fallback patch not available, relying on environment variables")

class ImageProcessor:
    def __init__(self, path):
        device = "cuda"
        self.pipe = FluxPipeline.from_pretrained(path, torch_dtype=torch.bfloat16, device=device)
        transformer = FluxTransformer2DModel.from_pretrained(path, subfolder="transformer", torch_dtype=torch.bfloat16, device=device)
        self.pipe.transformer = transformer
        self.pipe.to(device)

    def clear_cache(self, transformer):
        for name, attn_processor in transformer.attn_processors.items():
            attn_processor.bank_kv.clear()

    @spaces.GPU()
    def process_image(self, prompt='', subject_imgs=[], spatial_imgs=[], height=768, width=768, output_path=None, seed=42):
        image = self.pipe(
            prompt,
            height=int(height),
            width=int(width),
            guidance_scale=3.5,
            num_inference_steps=25,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(seed),
            subject_images=subject_imgs,
            spatial_images=spatial_imgs,
            cond_size=512,
        ).images[0]
        self.clear_cache(self.pipe.transformer)
        if output_path:
            image.save(output_path)
        return image

# Initialize the image processor
base_path = "black-forest-labs/FLUX.1-dev"
lora_base_path = "EasyControl/models"
style_lora_base_path = "Shakker-Labs"
processor = ImageProcessor(base_path)

# Define the Gradio interface
def single_condition_generate_image(prompt, subject_img, spatial_img, height, width, seed, control_type, style_lora=None):
    # Set the control type
    if control_type == "subject":
        lora_path = os.path.join(lora_base_path, "subject.safetensors")
    elif control_type == "depth":
        lora_path = os.path.join(lora_base_path, "depth.safetensors")
    elif control_type == "seg":
        lora_path = os.path.join(lora_base_path, "seg.safetensors")
    elif control_type == "pose":
        lora_path = os.path.join(lora_base_path, "pose.safetensors")
    elif control_type == "inpainting":
        lora_path = os.path.join(lora_base_path, "inpainting.safetensors")
    elif control_type == "hedsketch":
        lora_path = os.path.join(lora_base_path, "hedsketch.safetensors")
    elif control_type == "canny":
        lora_path = os.path.join(lora_base_path, "canny.safetensors")
    set_single_lora(processor.pipe.transformer, lora_path, lora_weights=[1], cond_size=512)

    # Set the style LoRA
    if style_lora=="None":
        pass
    else:
        if style_lora == "Simple_Sketch":
            processor.pipe.unload_lora_weights()
            style_lora_path = os.path.join(style_lora_base_path, "FLUX.1-dev-LoRA-Children-Simple-Sketch")
            processor.pipe.load_lora_weights(style_lora_path, weight_name="FLUX-dev-lora-children-simple-sketch.safetensors")
        if style_lora == "Text_Poster":
            processor.pipe.unload_lora_weights()
            style_lora_path = os.path.join(style_lora_base_path, "FLUX.1-dev-LoRA-Text-Poster")
            processor.pipe.load_lora_weights(style_lora_path, weight_name="FLUX-dev-lora-Text-Poster.safetensors")
        if style_lora == "Vector_Style":
            processor.pipe.unload_lora_weights()
            style_lora_path = os.path.join(style_lora_base_path, "FLUX.1-dev-LoRA-Vector-Journey")
            processor.pipe.load_lora_weights(style_lora_path, weight_name="FLUX-dev-lora-Vector-Journey.safetensors")

    # Process the image
    subject_imgs = [subject_img] if subject_img else []
    spatial_imgs = [spatial_img] if spatial_img else []
    image = processor.process_image(prompt=prompt, subject_imgs=subject_imgs, spatial_imgs=spatial_imgs, height=height, width=width, seed=seed)
    return image

# Define the Gradio interface
def multi_condition_generate_image(prompt, subject_img, spatial_img, height, width, seed):
    subject_path = os.path.join(lora_base_path, "subject.safetensors")
    inpainting_path = os.path.join(lora_base_path, "inpainting.safetensors")
    set_multi_lora(processor.pipe.transformer, [subject_path, inpainting_path], lora_weights=[[1],[1]],cond_size=512)

    # Process the image
    subject_imgs = [subject_img] if subject_img else []
    spatial_imgs = [spatial_img] if spatial_img else []
    image = processor.process_image(prompt=prompt, subject_imgs=subject_imgs, spatial_imgs=spatial_imgs, height=height, width=width, seed=seed)
    return image

# Define the Gradio interface components
control_types = ["subject", "depth", "pose", "inpainting", "hedsketch", "seg", "canny"]
style_loras = ["Simple_Sketch", "Text_Poster", "Vector_Style", "None"]

# Example data
single_examples = [
    ["A SKS in the library", Image.open("./test_imgs/subject1.png"), None, 1024, 1024, 5, "subject", None],
    ["In a picturesque village, a narrow cobblestone street with rustic stone buildings, colorful blinds, and lush green spaces, a cartoon man drawn with simple lines and solid colors stands in the foreground, wearing a red shirt, beige work pants, and brown shoes, carrying a strap on his shoulder. The scene features warm and enticing colors, a pleasant fusion of nature and architecture, and the camera's perspective on the street clearly shows the charming and quaint environment., Integrating elements of reality and cartoon.", None, Image.open("./test_imgs/spatial1.png"), 1024, 1024, 1, "pose", "Vector_Style"],
]
multi_examples = [
    ["A SKS on the car", Image.open("./test_imgs/subject2.png"), Image.open("./test_imgs/spatial2.png"), 1024, 1024, 7],
]


# Create the Gradio Blocks interface
with gr.Blocks() as demo:
    gr.Markdown("# Image Generation with EasyControl")
    gr.Markdown("Generate images using EasyControl with different control types and style LoRAs.")

    with gr.Tab("Single Condition Generation"):
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt")
                subject_img = gr.Image(label="Subject Image", type="pil")  # 上传图像文件
                spatial_img = gr.Image(label="Spatial Image", type="pil")  # 上传图像文件
                height = gr.Slider(minimum=256, maximum=1536, step=64, label="Height", value=768)
                width = gr.Slider(minimum=256, maximum=1536, step=64, label="Width", value=768)
                seed = gr.Number(label="Seed", value=42)
                control_type = gr.Dropdown(choices=control_types, label="Control Type")
                style_lora = gr.Dropdown(choices=style_loras, label="Style LoRA")
                single_generate_btn = gr.Button("Generate Image")
            with gr.Column():
                single_output_image = gr.Image(label="Generated Image")

        # Add examples for Single Condition Generation
        gr.Examples(
            examples=single_examples,
            inputs=[prompt, subject_img, spatial_img, height, width, seed, control_type, style_lora],
            outputs=single_output_image,
            fn=single_condition_generate_image,
            cache_examples=False,  # 缓存示例结果以加快加载速度
            label="Single Condition Examples"
        )


    with gr.Tab("Multi-Condition Generation"):
        with gr.Row():
            with gr.Column():
                multi_prompt = gr.Textbox(label="Prompt")
                multi_subject_img = gr.Image(label="Subject Image", type="pil")  # 上传图像文件
                multi_spatial_img = gr.Image(label="Spatial Image", type="pil")  # 上传图像文件
                multi_height = gr.Slider(minimum=256, maximum=1536, step=64, label="Height", value=768)
                multi_width = gr.Slider(minimum=256, maximum=1536, step=64, label="Width", value=768)
                multi_seed = gr.Number(label="Seed", value=42)
                multi_generate_btn = gr.Button("Generate Image")
            with gr.Column():
                multi_output_image = gr.Image(label="Generated Image")

        # Add examples for Multi-Condition Generation
        gr.Examples(
            examples=multi_examples,
            inputs=[multi_prompt, multi_subject_img, multi_spatial_img, multi_height, multi_width, multi_seed],
            outputs=multi_output_image,
            fn=multi_condition_generate_image,
            cache_examples=False,  # 缓存示例结果以加快加载速度
            label="Multi-Condition Examples"
        )


    # Link the buttons to the functions
    single_generate_btn.click(
        single_condition_generate_image,
        inputs=[prompt, subject_img, spatial_img, height, width, seed, control_type, style_lora],
        outputs=single_output_image
    )
    multi_generate_btn.click(
        multi_condition_generate_image,
        inputs=[multi_prompt, multi_subject_img, multi_spatial_img, multi_height, multi_width, multi_seed],
        outputs=multi_output_image
    )

# Launch the Gradio app
demo.queue().launch()
