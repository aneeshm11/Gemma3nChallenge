import os
import sys
import tempfile
import json
import math
import timm
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import gradio as gr
from huggingface_hub import snapshot_download
from typing import List, Union, Dict
import torchvision.transforms as transforms


# Vision Model
class TimmCNNModel(nn.Module):
    def __init__(self, num_classes: int = 8, model_name: str = "efficientnet_b0"):
        super().__init__()
        
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=True,
            num_classes=0,
        )
        
        self.feature_dim = self.backbone.num_features
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        logits = self.classifier(features)
        return logits

# Projector Model
class Projector_4to3d(nn.Module):
    def __init__(self, cnn_dim: int = 1280, llm_dim: int = 2048, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cnn_dim = cnn_dim
        self.llm_dim = llm_dim
        
        # Spatial positional embeddings for 8x8 grid
        self.spatial_pos_embed = nn.Parameter(torch.randn(64, cnn_dim))
        
        # Multi-scale feature processing
        self.spatial_conv = nn.Conv2d(cnn_dim, cnn_dim // 2, 1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Enhanced projection layers
        self.input_proj = nn.Sequential(
            nn.Linear(cnn_dim, llm_dim),
            nn.LayerNorm(llm_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-head self-attention for spatial reasoning
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=llm_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-attention for text-image alignment
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=llm_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(llm_dim)
        self.norm2 = nn.LayerNorm(llm_dim)
        
        # Enhanced FFN
        self.ffn = nn.Sequential(
            nn.Linear(llm_dim, llm_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_dim * 4, llm_dim),
            nn.Dropout(dropout)
        )
        
        self.norm3 = nn.LayerNorm(llm_dim)
        
        # Token compression layer
        self.compress_tokens = nn.Parameter(torch.randn(32, llm_dim))
        self.token_compression = nn.MultiheadAttention(
            embed_dim=llm_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
    
    def forward(self, cnn_features: torch.Tensor, text_embeddings: torch.Tensor = None) -> torch.Tensor:
        batch_size = cnn_features.shape[0]
        
        # Multi-scale processing
        spatial_features = self.spatial_conv(cnn_features)
        global_context = self.global_pool(cnn_features).flatten(1)
        
        # Flatten spatial features and add positional encoding
        x = einops.rearrange(cnn_features, "b c h w -> b (h w) c")
        pos_embeddings = self.spatial_pos_embed.unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos_embeddings
        
        # Project to LLM dimension
        x = self.input_proj(x)
        
        # Self-attention for spatial reasoning
        attended_x, spatial_attn_weights = self.spatial_attention(x, x, x)
        x = self.norm1(x + attended_x)
        
        # Cross-attention with text (if available)
        if text_embeddings is not None:
            text_embeddings_float = text_embeddings.float()
            cross_attended, cross_attn_weights = self.cross_attention(x, text_embeddings_float, text_embeddings_float)
            x = self.norm2(x + cross_attended)
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        
        # Optional token compression
        compress_queries = self.compress_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        compressed_x, _ = self.token_compression(compress_queries, x, x)
        
        return compressed_x

# Main VLM Model
class Model(nn.Module):
    def __init__(self, image_model, language_model, projector, tokenizer, prompt="Describe this image:"):
        super().__init__()
        self.image_model = image_model 
        self.language_model = language_model
        self.projector = projector
        self.tokenizer = tokenizer
        self.eos_token = tokenizer.eos_token
        self.prompt = prompt
        
        device = next(self.language_model.parameters()).device
        
        self.image_model.to(device)
        self.projector.to(device)
        
        # Create prompt embeddings
        prompt_tokens = tokenizer(text=prompt, return_tensors="pt").input_ids.to(device)
        prompt_embeddings = language_model.get_input_embeddings()(prompt_tokens).detach()
        self.register_buffer('prompt_embeddings', prompt_embeddings)
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def generate(self, patches: torch.Tensor, generator_kwargs: dict[str, Union[int, float]]):
        device = self.device
        patches = patches.to(device)
        
        image_features = self.image_model.backbone.forward_features(patches)
        patch_embeddings = self.projector(image_features)
        patch_embeddings = patch_embeddings.to(torch.bfloat16)
        
        embeddings = torch.cat([
            self.prompt_embeddings.expand(patches.size(0), -1, -1),
            patch_embeddings,
        ], dim=1)
        
        prompt_mask = torch.ones(patches.size(0), self.prompt_embeddings.size(1), device=device)
        patch_mask = torch.ones(patches.size(0), patch_embeddings.size(1), device=device)
        attention_mask = torch.cat([prompt_mask, patch_mask], dim=1)
        
        return self.language_model.generate(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            **generator_kwargs
        )

vlm_model = None
tokenizer = None
transform = None

def download_and_load_models():
    global vlm_model, tokenizer, transform
    
    print("Starting model download and initialization...")
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("CUDA available - using GPU")
    else:
        device = torch.device("cpu")
        print("CUDA not available - using CPU")
    
    repo_id = "aneeshm44/regfinal"
    print(f"Downloading from repo: {repo_id}")
    
    local_dir = tempfile.mkdtemp(prefix="regfinal_")
    print(f"Local directory: {local_dir}")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            allow_patterns=[
                "llmweights/*",
                "imagemodelweights/finalcheckpoint.pth",
                "projectorweights/projector.pth"
            ],
            local_dir_use_symlinks=False,
        )
        print("Download completed successfully")
    except Exception as e:
        print(f"Download failed: {e}")
        raise e
    
    llm_path = os.path.join(local_dir, "llmweights")
    image_weights_path = os.path.join(local_dir, "imagemodelweights", "finalcheckpoint.pth")
    projector_weights_path = os.path.join(local_dir, "projectorweights", "projector.pth")
    
    print("Loading language model...")
    try:
        language_model = AutoModelForCausalLM.from_pretrained(
            llm_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        language_model.eval()
        language_model.to(device)
        
        tokenizer = AutoTokenizer.from_pretrained(llm_path)
        print("Language model loaded successfully")
    except Exception as e:
        print(f"Language model loading failed: {e}")
        raise e
    
    print("Loading vision model...")
    try:
        image_model = TimmCNNModel(num_classes=8)
        weights = torch.load(image_weights_path, map_location=device)
        image_model.load_state_dict(weights['model_state_dict'])
        
        for param in image_model.parameters():
            param.requires_grad = False
        image_model.eval()
        image_model.to(device)
        print("Vision model loaded successfully")
    except Exception as e:
        print(f"Vision model loading failed: {e}")
        raise e
    
    print("Loading projector...")
    try:
        projector = Projector_4to3d(cnn_dim=1280, llm_dim=2048, num_heads=8)
        weights = torch.load(projector_weights_path, map_location=device)
        projector.load_state_dict(weights)
        
        for param in projector.parameters():
            param.requires_grad = False
        projector.eval()
        projector.to(device)
        print("Projector loaded successfully")
    except Exception as e:
        print(f"Projector loading failed: {e}")
        raise e
    
    print("Creating VLM model...")
    try:
        vlm_model = Model(image_model, language_model, projector, tokenizer, prompt="Describe this image:")
        vlm_model = vlm_model.to(device)
        print("VLM model created successfully")
    except Exception as e:
        print(f"VLM model creation failed: {e}")
        raise e
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    print("All models loaded successfully!")

def tensor_to_pil_image(tensor):
    img_tensor = tensor.squeeze(0)
    img_tensor = torch.clamp(img_tensor, 0, 1)
    
    img_array = img_tensor.permute(1, 2, 0).numpy()
    img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array)

def on_image_upload(image):
    if image is not None:
        return "Image processed, click 'Generate Report' to produce report."
    else:
        return "Models are loaded, upload the Image to get started."

def describe_image(image, temperature, top_p, max_tokens, progress=gr.Progress()):
    global vlm_model, tokenizer, transform
    
    if vlm_model is None:
        return "Models not loaded yet. Please wait for initialization to complete.", None
    
    if image is None:
        return "Please upload an image.", None
    
    try:
        progress(0.1, desc="Starting image processing...")
        
        # Preprocess image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif hasattr(image, 'convert'):
            image = image.convert('RGB')
        
        progress(0.3, desc="Applying image transformations...")
        
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        processed_image = tensor_to_pil_image(image_tensor)
        
        progress(0.5, desc="Setting up generation parameters...")
        
        # Generation parameters
        generator_kwargs = {
            "max_new_tokens": int(max_tokens),
            "do_sample": True,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "pad_token_id": tokenizer.eos_token_id
        }
        
        progress(0.7, desc="Generating pathology report...")
        
        # Generate description
        with torch.no_grad():
            output_ids = vlm_model.generate(image_tensor, generator_kwargs)
            text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        progress(0.9, desc="Finalizing report...")
            
        if "Describe this image:" in text:
            description = text.split("Describe this image:")[-1].strip()
        else:
            description = text.strip()
            
        result_text = description if description else "Unable to generate description."
        
        progress(1.0, desc="Complete!")
        
        return result_text, processed_image
        
    except Exception as e:
        return f"Error processing image: {str(e)}", None

def reset_interface():
    return None, "Models are loaded, upload the WSI file to get started.", None

try:
    download_and_load_models()
    initial_status = "Models are loaded, upload the WSI file to get started."
except Exception as e:
    initial_status = f"Failed to load models: {str(e)}"

def create_interface():
    with gr.Blocks(title="WSI Pathology Report using Gemma3n") as demo:
        gr.Markdown("# WSI Pathology Report using Gemma3n")
        gr.Markdown("Upload a pathology WSI to get concise a report")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload WSI file")
                
                # Generation parameters
                with gr.Row():
                    temperature_slider = gr.Slider(
                        minimum=0.1, 
                        maximum=1.0, 
                        value=0.6, 
                        step=0.1, 
                        label="Temperature",
                        info="Lower values give consistent results and Higher values produce creative results"
                    )
                    
                    top_p_slider = gr.Slider(
                        minimum=0.1, 
                        maximum=1.0, 
                        value=0.9, 
                        step=0.1, 
                        label="Top-p",
                        info="Lower values use a more focused vocabulary for sampling compared to a more diverse vocabulary in Higher values"
                    )
                    
                    max_tokens_slider = gr.Slider(
                        minimum=10, 
                        maximum=200, 
                        value=100, 
                        step=10, 
                        label="Max Tokens for generation"
                    )
                
                with gr.Row():
                    submit_btn = gr.Button("Generate Report", variant="primary")
                    reset_btn = gr.Button("Reset", variant="secondary")
            
            with gr.Column():
                output_text = gr.Textbox(
                    label="Pathology Report", 
                    lines=8,
                    value=initial_status,
                    show_copy_button=True
                )
                
                processed_image = gr.Image(
                    label="Processed WSI",
                    show_download_button=True
                )
        
        image_input.change(
            fn=on_image_upload,
            inputs=[image_input],
            outputs=[output_text]
        )
        
        submit_btn.click(
            fn=describe_image,
            inputs=[image_input, temperature_slider, top_p_slider, max_tokens_slider],
            outputs=[output_text, processed_image],
            show_progress=True
        )
        
        reset_btn.click(
            fn=reset_interface,
            inputs=[],
            outputs=[image_input, output_text, processed_image]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )