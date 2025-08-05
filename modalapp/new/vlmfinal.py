import modal
from pathlib import Path

# ========================================= PARAMETERS AND ARGS =========================================

number = 11.7

class CFG:
    app_name = f"vlm-training-app-{number}"
    volume_name = f"vlm-training-vol-{number}"

    gpu_type = "L40S:1"
    timeout = 9 * 60 * 60  

    hf_api_key = ""
    hf_username = "aneeshm44"
    hf_dataset_name = f"gemma-vlm-trial-{number}"

    EPOCHS = 13
    GRAD_ACC = 1
    TRAIN_BATCH_SIZE = 10
    TRAIN_START = 0
    TRAIN_END = 5000

    LABEL_MASK = -100
    MAX_LENGTH = 50 
    
    VM_LR = 2e-4
    LLM_LR = 2e-5

    base_model_repo = "aneeshm44/gemma3"

# ========================================= MODAL APP CONFIGURATION =========================================

image = modal.Image.debian_slim().pip_install(
    "timm",
    "einops", 
    "lightning",
    "torch",
    "peft",
    "datasets",
    "huggingface_hub",
    "numpy",
    "pandas",
    "tqdm",
    "requests",  
    "transformers",
    )

app = modal.App(name=CFG.app_name)
output_vol = modal.Volume.from_name(CFG.volume_name, create_if_missing=True)

VOL_MOUNT_PATH = Path("/vol")

@app.function(
    gpu=CFG.gpu_type,
    timeout=CFG.timeout,
    volumes={VOL_MOUNT_PATH: output_vol},
    image=image,
)
def train_and_upload():

    # ----------------------------------------- IMPORTS ------------------------------------------
    import os
    import sys
    
    import json, math, timm, einops
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import numpy as np
    import lightning as LIGHTNING
    from lightning.pytorch.callbacks import Callback
    from typing import List, Union, Dict
    from huggingface_hub import snapshot_download, login, create_repo, upload_folder
    import platform
    import subprocess
    from peft import LoraConfig, get_peft_model, TaskType
    from tqdm import tqdm
    import torch._dynamo
    torch._dynamo.disable()

    # ----------------------------------------- HUGGING FACE SETUP AND KERNEL INFORMATION ------------------------------------------

    HF_API_KEY = CFG.hf_api_key
    login(HF_API_KEY)
    
    username, model_repo_name = CFG.hf_username, CFG.hf_dataset_name
    base_model_repo = CFG.base_model_repo

    print("STARTING VLM TRAINING WORKFLOW...")    

    local_model_path = VOL_MOUNT_PATH / "base_model"    
    local_model_path.mkdir(exist_ok=True)

    print("SYSTEM INFORMATION:")
    print(f"KERNEL: {platform.release()}")
    print(f"PYTHON: {platform.python_version()}")
    print(f"PYTORCH: {torch.__version__}")
    print("="*100)

    def run_nvidia_smi():
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            print("NVIDIA-SMI OUTPUT:\n", result.stdout)
            print("="*100)
        except subprocess.CalledProcessError as e:
            print("NVIDIA-SMI FAILED:\n", e.stderr)
            print("="*100)
        except FileNotFoundError:
            print("NVIDIA-SMI NOT FOUND.")
            print("="*100)
    
    print("INITIAL GPU STATUS:")
    print("="*100)
    run_nvidia_smi()

    print("DOWNLOADING BASE MODEL...")

    try:
        downloaded_path = snapshot_download(
            repo_id=base_model_repo,
            local_dir=str(local_model_path),
            repo_type="dataset",
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"BASE MODEL DOWNLOADED TO: {downloaded_path}")
        print("STARTING TRAINING PROCESS...")
        print("="*100)

        # ----------------------------------------- TRAINING WORKFLOW ------------------------------------------

        def start_training():
            
            tokenizer = AutoTokenizer.from_pretrained(str(local_model_path), trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                str(local_model_path),
                trust_remote_code=True,
                device_map="cuda:0",
                low_cpu_mem_usage=True,
                # torch_dtype=torch.bfloat16,      # uncomment for bfloat16 training. By default model weights are loaded in float32
            )
            
            num_layers = 30
            target_modules = []

            for i in range(num_layers):
                target_modules.extend([
                    f"language_model.layers.{i}.self_attn.q_proj",
                    f"language_model.layers.{i}.self_attn.k_proj",
                    f"language_model.layers.{i}.self_attn.v_proj",
                    f"language_model.layers.{i}.self_attn.o_proj",
                    f"language_model.layers.{i}.mlp.gate_proj",
                    f"language_model.layers.{i}.mlp.up_proj",
                    f"language_model.layers.{i}.mlp.down_proj"
                ])
            
            lora_config = LoraConfig(
                r=256,
                lora_alpha=256*4,
                target_modules=target_modules,
                lora_dropout=0.04,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            language_model = get_peft_model(model, lora_config) 
            # language_model = language_model.to(dtype=torch.bfloat16)  # uncomment these for bfloat16 training
            language_model.print_trainable_parameters()

            try:
                l = []
                for param in language_model.parameters():
                    l.append(param.dtype)
                print(set(l))
            except Exception as e:
                print(f"Error checking parameter types: {str(e)}")
            

            # ----------------------------------------- IMAGE MODEL SETUP ------------------------------------------

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

            image_model = TimmCNNModel(num_classes=8)
            
            # Download and load pre-trained weights
            files_path = VOL_MOUNT_PATH / "reg_files"
            files_path.mkdir(exist_ok=True)
    
            downloaded_path = snapshot_download(
                repo_id="aneeshm44/regdata",
                repo_type="dataset",
                local_dir=str(files_path),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
            weights = torch.load(files_path / "finalcheckpoint_cnn_timm.pth", map_location='cuda')
            image_model.load_state_dict(weights['model_state_dict'])
            
            for param in image_model.parameters():
                param.requires_grad = False
                
            print("IMAGE MODEL LOADED")

            # ----------------------------------------- PROJECTOR SETUP ------------------------------------------            
            class Projector_4to3d(nn.Module):
                def __init__(self, cnn_dim: int = 1280, llm_dim: int = 2048, num_heads: int = 8, dropout: float = 0.04):
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
                    
                    # Cross-attention with text (if available during training)
                    if text_embeddings is not None:
                        text_embeddings_float = text_embeddings.float()
                        cross_attended, cross_attn_weights = self.cross_attention(x, text_embeddings_float, text_embeddings_float)
                        x = self.norm2(x + cross_attended)
                    
                    # FFN
                    ffn_out = self.ffn(x)
                    x = self.norm3(x + ffn_out)
                    
                    # Token compression
                    compress_queries = self.compress_tokens.unsqueeze(0).expand(batch_size, -1, -1)
                    compressed_x, _ = self.token_compression(compress_queries, x, x)
                    
                    return compressed_x
            
            projector = Projector_4to3d(cnn_dim=1280, llm_dim=2048, num_heads=8)
            
            print("PROJECTOR LOADED!")

            # ----------------------------------------- DATASET SETUP ------------------------------------------
            
            patches = np.load(files_path / "images.npy", mmap_mode="r")
            
            with open(files_path / "labels.json", "r") as f:
                raw_texts = json.load(f)

            texts = []
            for t in raw_texts:
                texts.append(t.replace("\n", "").strip())

            

            print(f"Patches shape: {np.shape(patches)}, Texts length: {len(texts)}")

            class REGDataset(Dataset):
                def __init__(self, patches_mmap, texts: list[str], start_idx: int, end_idx: int):
                    self.patches_mmap = patches_mmap
                    self.texts = texts
                    self.start_idx = start_idx
                    self.end_idx = end_idx
                
                def __len__(self):
                    return self.end_idx - self.start_idx
                
                def __getitem__(self, idx):
                    actual_idx = self.start_idx + idx
                    patches = (self.patches_mmap[actual_idx])
                    texts = self.texts[actual_idx]
                    return torch.tensor(patches), texts

            train_data = REGDataset(patches, texts, CFG.TRAIN_START, CFG.TRAIN_END)
            train_dl = DataLoader(train_data, CFG.TRAIN_BATCH_SIZE, pin_memory=True, shuffle=True, drop_last=True)
            
            print("DATASET LOADED!")

            # ----------------------------------------- MAIN MODEL DEFINITION ------------------------------------------
            
            class Model(nn.Module):
                def __init__(self, image_model, language_model, projector, tokenizer, prompt="Describe the medical image:"):
                    super().__init__()
                    self.image_model = image_model 
                    self.language_model = language_model
                    self.projector = projector
                    self.tokenizer = tokenizer
                    self.eos_token = tokenizer.eos_token
                    self.prompt = prompt
                    
                    device = next(self.language_model.parameters()).device
                    
                    self.image_model = self.image_model.to(device)
                    self.projector = self.projector.to(device)
                    
                    # Create prompt embeddings
                    prompt_tokens = tokenizer(text=prompt, return_tensors="pt").input_ids.to(device)
                    prompt_embeddings = language_model.get_input_embeddings()(prompt_tokens).detach()
                    self.register_buffer('prompt_embeddings', prompt_embeddings)
                    
                    # Contrastive learning components
                    self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
                    self.image_projection_head = nn.Sequential(
                        nn.Linear(2048, 512),
                        nn.ReLU(),
                        nn.Linear(512, 256)
                    ).to(device)
                    self.text_projection_head = nn.Sequential(
                        nn.Linear(2048, 512),
                        nn.ReLU(),
                        nn.Linear(512, 256)
                    ).to(device)
                
                @property
                def device(self):
                    return next(self.parameters()).device
                
                def forward(self, patches: torch.Tensor, texts: List[str], compute_contrastive: bool = True):
                    device = self.device
                    patches = patches.to(device)
                    
                    image_features = self.image_model.backbone.forward_features(patches)
                    
                    tokenized = self.tokenizer(
                        text=[text for text in texts],
                        padding=True,
                        truncation=True,
                        max_length=CFG.MAX_LENGTH,
                        return_tensors="pt",
                    )
                    tokenized = {k: v.to(device) for k, v in tokenized.items()}
                    
                    text_embeddings = self.language_model.get_input_embeddings()(tokenized["input_ids"])
                    patch_embeddings = self.projector(image_features, text_embeddings)

                    # patch_embeddings = patch_embeddings.to(torch.bfloat16)  # uncomment these for bfloat16 training
                    # text_embeddings = text_embeddings.to(torch.bfloat16)    # uncomment these for bfloat16 training

                    
                    # Concatenate embeddings
                    embeddings = torch.cat([
                        self.prompt_embeddings.expand(patches.size(0), -1, -1),
                        patch_embeddings,
                        text_embeddings,
                    ], dim=1)
                    
                    # Create attention mask
                    prompt_mask = torch.ones(patches.size(0), self.prompt_embeddings.size(1), device=device)
                    patch_mask = torch.ones(patches.size(0), patch_embeddings.size(1), device=device)
                    attention_mask = torch.cat([prompt_mask, patch_mask, tokenized["attention_mask"]], dim=1)
                    
                    # Create labels
                    prompt_labels = torch.full((patches.size(0), self.prompt_embeddings.size(1)), CFG.LABEL_MASK, device=device)
                    patch_labels = torch.full((patches.size(0), patch_embeddings.size(1)), CFG.LABEL_MASK, device=device)
                    text_labels = tokenized["input_ids"].clone()
                    labels = torch.cat([prompt_labels, patch_labels, text_labels], dim=1)
                    labels[attention_mask == 0] = CFG.LABEL_MASK
                    
                    llm_output = self.language_model(
                        inputs_embeds=embeddings,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    total_loss = llm_output.loss
                    loss_dict = {"language_loss": llm_output.loss}
                    
                    if compute_contrastive:
                        # Contrastive loss between image and text
                        image_global = patch_embeddings.mean(dim=1)
                        text_global = text_embeddings.mean(dim=1)
                        
                        # Project to contrastive space
                        image_proj = self.image_projection_head(image_global.float())
                        text_proj = self.text_projection_head(text_global.float())
                        
                        # Normalize
                        image_proj = F.normalize(image_proj, dim=-1)
                        text_proj = F.normalize(text_proj, dim=-1)
                        
                        # Compute contrastive loss
                        logits = torch.matmul(image_proj, text_proj.t()) * self.temperature.exp()
                        labels_contrastive = torch.arange(len(logits), device=device)
                        
                        contrastive_loss = (F.cross_entropy(logits, labels_contrastive) + 
                                          F.cross_entropy(logits.t(), labels_contrastive)) / 2
                        
                        total_loss = total_loss + 0.4 * contrastive_loss    # changed from 0.1 to 0.4
                        loss_dict["contrastive_loss"] = contrastive_loss
                    
                    # Attention regularization loss
                    if hasattr(self.projector, 'spatial_attention'):
                        attn_entropy_loss = 0.0
                        loss_dict["attention_entropy_loss"] = attn_entropy_loss
                    
                    return {
                        "loss": total_loss,
                        "logits": llm_output.logits,
                        "loss_breakdown": loss_dict
                    }
                
                def generate(self, patches: torch.Tensor, generator_kwargs: dict[str, Union[int, float]]):
                    device = self.device
                    patches = patches.to(device)
                    self.image_model = self.image_model.to(device)
                    self.projector = self.projector.to(device)

                    
                    image_features = self.image_model.backbone.forward_features(patches)
                    patch_embeddings = self.projector(image_features)
                    
                    # patch_embeddings = patch_embeddings.to(torch.bfloat16) # for bfloat16 inference
                    
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

            class LightningModule(LIGHTNING.LightningModule):
                def __init__(self, model: Model):
                    super().__init__()
                    self.model = model
                    self.automatic_optimization = False  
                
                def training_step(self, batch, batch_idx):
                    opt = self.optimizers()
                    sch = self.lr_schedulers()
                    
                    patches, texts = batch
                    
                    output = self.model(patches, texts, compute_contrastive=True)
                    
                    total_loss = output["loss"]
                    loss_breakdown = output["loss_breakdown"]
                    
                    self.manual_backward(total_loss)
                    
                    self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
                    
                    opt.step()
                    opt.zero_grad()
                    sch.step()
                    
                    self.log("train_loss", total_loss, prog_bar=True)
                    for loss_name, loss_value in loss_breakdown.items():
                        self.log(f"{loss_name}", loss_value)
                    
                    return total_loss        
                
                def configure_optimizers(self):
                    params = [
                        {"params": self.model.projector.parameters(), "lr": CFG.VM_LR, "weight_decay": 1e-4},
                        {"params": self.model.image_projection_head.parameters(), "lr": CFG.VM_LR, "weight_decay": 1e-4},
                        {"params": self.model.text_projection_head.parameters(), "lr": CFG.VM_LR, "weight_decay": 1e-4},
                        {"params": [self.model.temperature], "lr": CFG.VM_LR, "weight_decay": 0.0},
                        {"params": [p for p in self.model.language_model.parameters() if p.requires_grad], "lr": CFG.LLM_LR, "weight_decay": 1e-5}
                    ]

                    optimizer = torch.optim.AdamW(params, eps=1e-8)
                    
                    # Cosine annealing with warmup
                    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                        optimizer,
                        max_lr=[param_group["lr"] for param_group in optimizer.param_groups],
                        total_steps=self.trainer.estimated_stepping_batches,
                        pct_start=0.1,  # 10% warmup
                        anneal_strategy='cos',
                        div_factor=25.0,
                        final_div_factor=1000.0,
                    )
                    
                    return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

            # Initialize loss tracking arrays
            step_train_losses = []
            step_language_losses = []
            step_contrastive_losses = []
            step_attention_entropy_losses = []
            
            class PrintLossCallback(Callback):
                def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                    metrics = trainer.callback_metrics
                    
                    if 'train_loss' in metrics:
                        step_train_losses.append(float(metrics['train_loss']))
                    if 'language_loss' in metrics:
                        step_language_losses.append(float(metrics['language_loss']))
                    if 'contrastive_loss' in metrics:
                        step_contrastive_losses.append(float(metrics['contrastive_loss']))
                    if 'attention_entropy_loss' in metrics:
                        step_attention_entropy_losses.append(float(metrics['attention_entropy_loss']))
            
                def on_train_epoch_end(self, trainer, pl_module):
                    metrics = trainer.callback_metrics
                    text = f"\nEpoch {trainer.current_epoch} Summary:"
                    for key, value in metrics.items():
                        text += f"  {key}: {value:.4f}"
                            
                    print(text)
                
            vlm_model = Model(image_model, language_model, projector, tokenizer)
            vlm_model = vlm_model.to(torch.device("cuda:0"))
            lightning_module = LightningModule(vlm_model)

            # ----------------------------------------- TRAINING ------------------------------------------
            print("STARTING TRAINING...")
            
            trainer = LIGHTNING.Trainer(
                max_epochs=CFG.EPOCHS,
                accumulate_grad_batches=CFG.GRAD_ACC,
                enable_progress_bar=True,
                log_every_n_steps=100,
                enable_checkpointing=False,
                callbacks=[PrintLossCallback()]
            )

            trainer.fit(
                model=lightning_module,
                train_dataloaders=train_dl, 
                datamodule=None,
            )

            print("TRAINING COMPLETED!")

            # ----------------------------------------- SAVE MODELS ------------------------------------------
            print("SAVING TRAINED MODELS...")
            
            lmweights_lora_path = VOL_MOUNT_PATH / "lmweights_lora"
            lmweights_merged_path = VOL_MOUNT_PATH / "lmweights_merged"
            vmweights_path = VOL_MOUNT_PATH / "vmweights"
            lmweights_lora_path.mkdir(exist_ok=True)
            lmweights_merged_path.mkdir(exist_ok=True)
            vmweights_path.mkdir(exist_ok=True)

            try:
                import shutil
                if local_model_path.exists():
                    shutil.rmtree(local_model_path)
                if files_path.exists():
                    shutil.rmtree(files_path)
            except Exception as e:
                print(f"Error removing old model files: {str(e)}")
            
            lightning_module.model.language_model.save_pretrained(str(lmweights_lora_path))
            
            try:
                merged_model = lightning_module.model.language_model.merge_and_unload()

                merged_model.save_pretrained(
                    str(lmweights_merged_path),
                    safe_serialization=True
                )
                
                tokenizer.save_pretrained(str(lmweights_merged_path))
                
                
            except Exception as e:
                print(f"Error saving merged model: {str(e)}")
                print("Falling back to LoRA-only save...")
            
            torch.save(lightning_module.model.projector.state_dict(), str(vmweights_path / "projector.pth"))
            
            # Save loss arrays
            np.save(str(VOL_MOUNT_PATH / "step_train_losses.npy"), np.array(step_train_losses))
            np.save(str(VOL_MOUNT_PATH / "step_language_losses.npy"), np.array(step_language_losses))
            np.save(str(VOL_MOUNT_PATH / "step_contrastive_losses.npy"), np.array(step_contrastive_losses))
            np.save(str(VOL_MOUNT_PATH / "step_attention_entropy_losses.npy"), np.array(step_attention_entropy_losses))

            print("MODELS SAVED!")
            print("="*100)

            # ----------------------------------------- INFERENCE TEST ------------------------------------------
            print("RUNNING INFERENCE TEST...")
            
            start_idx = 5000
            end_idx = len(patches)
            
            patches_batch = np.array(patches[start_idx:end_idx])
            patches_batch = torch.tensor(patches_batch)
            
            generator_kwargs = {
                "max_new_tokens": 200,
                "do_sample": True,
                "temperature": 0.4,
                "top_p": 0.9,
                "pad_token_id": tokenizer.eos_token_id
            }
            
            batch_size = 20
            generated_ids = []
            
            for i in tqdm(range(0, len(patches_batch), batch_size), desc="Running inference"):
                batch_chunk = patches_batch[i:i+batch_size].to(torch.device("cuda:0"))
                batch_chunk = batch_chunk.to(device=torch.device("cuda:0"), dtype=torch.float32)
                chunk_ids = vlm_model.generate(batch_chunk, generator_kwargs)
                generated_ids.extend(chunk_ids)

            generated_texts = []
            for new_tokens in generated_ids:
                text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                generated_texts.append(text.strip())

            with open(str(VOL_MOUNT_PATH / "inference_results.json"), "w") as f:
                json.dump(generated_texts, f, indent=2)

            print("INFERENCE TEST COMPLETED!")
        start_training()

        # ----------------------------------------- UPLOAD TO HUGGING FACE ------------------------------------------
        
        try:
            create_repo(
                repo_id=f"{username}/{model_repo_name}",
                repo_type="dataset",
                exist_ok=True,
                private=False
            )
            
            upload_folder(
                repo_id=f"{username}/{model_repo_name}",
                folder_path=str(VOL_MOUNT_PATH),
                repo_type="dataset",
                commit_message=f"Upload VLM training results {number}"
            )
            
            print(f"UPLOADED TO: https://huggingface.co/datasets/{username}/{model_repo_name}")
            
        except Exception as e:
            print(f"UPLOAD FAILED: {str(e)}")
        
        print("="*100)

        # ----------------------------------------- CLEANUP ------------------------------------------
        output_vol.commit()

        print("FINAL GPU STATUS:")
        print("="*100)
        run_nvidia_smi()

    except Exception as e:
        print(f"ERROR DURING WORKFLOW: {str(e)}")
        raise e
    
    print("VLM TRAINING WORKFLOW FINISHED SUCCESSFULLY!")
    print("="*100)

@app.local_entrypoint()
def main():
    train_and_upload.remote()

if __name__ == "__main__":

    train_and_upload.remote()
