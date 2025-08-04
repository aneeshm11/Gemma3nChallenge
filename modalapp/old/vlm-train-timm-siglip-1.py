import modal
from pathlib import Path

# ========================================= PARAMETERS AND ARGS =========================================

number = 8.3

class CFG:

    app_name = f"training-app-trail-{number}"
    volume_name = f"training-app-trial-{number}"

    gpu_type = "L40S:1"
    timeout = 1 * 30 * 60  

    hf_api_key = ""
    hf_username = "aneeshm44"
    hf_dataset_name = f"gemma-3n-timm-trial-{number}"

    TRAIN_START = 0
    TRAIN_END = 2380
    TRAIN_BATCH_SIZE = 5

    EPOCHS = 20
    GRAD_ACC = 1
    
    MAX_LENGTH = 50
    LABEL_MASK = -100  # for loss calculation
    
    # SigLIP specific parameters
    SIGLIP_TEMP = 0.07
    SIGLIP_WEIGHT = 1.0  # Weight for contrastive loss vs generation loss
    GLOBAL_EMBED_DIM = 512*4  # Dimension for global image/text representations

    base_model_repo = "aneeshm44/gemma3"

# ========================================= MODAL APP CONFIGURATION =========================================

image = modal.Image.debian_slim().pip_install(
    "transformers",
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

# ========================================= TRAINING FUNCTION =========================================

def train_and_upload():

    # ----------------------------------------- IMPORTS ------------------------------------------
    import platform
    import os
    import pathlib
    import random
    import numpy as np
    import torch
    import torch.nn as nn
    import json
    import torch.nn.functional as F
    import timm
    import einops 
    from torch.utils.data import Dataset, DataLoader
    from torch import Tensor
    from tqdm.auto import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer , GenerationConfig , BitsAndBytesConfig
    import lightning as LIGHTNING
    from lightning.pytorch.callbacks import Callback
    import shutil
    from typing import List, Union , Dict
    from peft import LoraConfig, get_peft_model, TaskType
    from huggingface_hub import snapshot_download, login, create_repo, upload_folder

    # ----------------------------------------- HUGGING FACE SETUP AND KERNEL INFORMATION ------------------------------------------

    HF_API_KEY = "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"
    login(HF_API_KEY)
    
    username, model_repo_name = CFG.hf_username, CFG.hf_dataset_name
    base_model_repo = CFG.base_model_repo

    print("STARTING COMPLETE TRAINING WORKFLOW...")    

    local_model_path = VOL_MOUNT_PATH / "base_model"
    trained_language_model_path = VOL_MOUNT_PATH / "trained_lm"
    trained_vision_model_path = VOL_MOUNT_PATH / "trained_vm"
    
    local_model_path.mkdir(exist_ok=True)
    trained_language_model_path.mkdir(exist_ok=True)
    trained_vision_model_path.mkdir(exist_ok=True)

    print("SYSTEM INFORMATION:")
    print(f"KERNEL: {platform.release()}")
    print(f"PYTHON: {platform.python_version()}")
    print(f"PYTORCH: {torch.__version__}")
    print("="*100)

    import subprocess
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
        print("="*100)
        
        print("STARTING TRAINING PROCESS...")
        print("="*100)

    # ----------------------------------------- TRAINING WORKFLOW ------------------------------------------

        def start_training():
            
            device = torch.device("cuda:0")
            language_model = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                trust_remote_code=True,
                device_map="cuda:0",                
                low_cpu_mem_usage=True,
            )

            target_modules = []
            num_layers = 30
            
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
                r=128,
                lora_alpha=128*4,
                target_modules=target_modules,
                lora_dropout=0.06,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            language_model1 = get_peft_model(language_model, lora_config)

            tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token
            
            class Projection(nn.Module):
                def __init__(self, d_in: int, d_out: int, p: float = 0.5, last_layer=False) -> None:
                    super().__init__()
                    self.linear1 = nn.Linear(d_in, d_out, bias=False)
                    self.linear2 = nn.Linear(d_out, d_out, bias=False)
                    self.layer_norm = nn.Identity() if last_layer else nn.LayerNorm(d_out)
                    self.drop = nn.Dropout(p)
            
                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    embed1 = self.linear1(x)
                    embed2 = self.drop(self.linear2(F.gelu(embed1)))
                    embeds = self.layer_norm(embed1 + embed2)
                    return embeds
            
            def projection_layers(d_in: int, d_out: int, layers: int) -> list[nn.Module]:
                return [Projection(d_in, d_in), nn.GELU(), nn.LayerNorm(d_in)] * (layers - 1) + [
                    Projection(d_in, d_out, last_layer=True)
                ]
            
            path = "timm/tf_efficientnet_b7.aa_in1k"
            image_model = timm.create_model(path, pretrained=True).to(device) 
            config = timm.data.resolve_data_config({}, model=image_model)
            transform = timm.data.transforms_factory.create_transform(**config, is_training=False)
            
            files_path = VOL_MOUNT_PATH / "reg_files"
            files_path.mkdir(exist_ok=True)
    
            downloaded_path = snapshot_download(
                repo_id="aneeshm44/reg-1",
                repo_type="dataset",
                local_dir=str(files_path),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            # print(os.listdir(files_path))
    
            patches = np.load(files_path / "small1.npy", mmap_mode='r')
    
            with open(files_path / "filenames1.json", "r") as f:
                texts = json.load(f)
                
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
            
            class ImprovedModel(nn.Module):
                def __init__(self, image_model, language_model, tokenizer, prompt="Describe the medical image:"):
                    super().__init__()
                    self.image_model = image_model 
                    self.language_model = language_model 
                    self.tokenizer = tokenizer
                    self.eos_token = tokenizer.eos_token
                    self.prompt = prompt

                    self.config = timm.data.resolve_data_config({}, model=self.image_model)
                    self.transform = timm.data.transforms_factory.create_transform(**self.config, is_training=False)
                    
                    # Patch-level projector for text generation conditioning
                    self.patch_projector = nn.Sequential(
                        *projection_layers(image_model.num_features, 2048, 6)
                    ).to(device)
                    
                    # Global projectors for SigLIP contrastive learning
                    self.image_global_projector = nn.Sequential(
                        nn.Linear(image_model.num_features, CFG.GLOBAL_EMBED_DIM),
                        nn.ReLU(),
                        nn.Linear(CFG.GLOBAL_EMBED_DIM, CFG.GLOBAL_EMBED_DIM),
                        nn.LayerNorm(CFG.GLOBAL_EMBED_DIM)
                    ).to(device)
                    
                    self.text_global_projector = nn.Sequential(
                        nn.Linear(2048, CFG.GLOBAL_EMBED_DIM),
                        nn.ReLU(), 
                        nn.Linear(CFG.GLOBAL_EMBED_DIM, CFG.GLOBAL_EMBED_DIM),
                        nn.LayerNorm(CFG.GLOBAL_EMBED_DIM)
                    ).to(device)

                    # Global average pooling for image features
                    self.global_pool = nn.AdaptiveAvgPool2d(1)

                    # Tokenize and get embeddings of prompt
                    prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")
                    prompt_embeddings = language_model.get_input_embeddings()(prompt_tokens).detach() 
                    self.register_buffer('prompt_embeddings', prompt_embeddings)

                @property
                def device(self):
                    return next(self.parameters()).device

                def siglip_loss(self, image_features, text_features, temperature=CFG.SIGLIP_TEMP):
                    """
                    SigLIP loss implementation - more efficient than CLIP for small batches
                    """
                    # Normalize features
                    image_features = F.normalize(image_features, dim=-1)
                    text_features = F.normalize(text_features, dim=-1)
                    
                    # Compute similarity matrix
                    logits = torch.matmul(image_features, text_features.T) / temperature
                    
                    # Create positive and negative labels
                    batch_size = image_features.size(0)
                    
                    # Positive pairs (diagonal)
                    positive_mask = torch.eye(batch_size, device=image_features.device, dtype=torch.bool)
                    positive_logits = logits[positive_mask]
                    
                    # Negative pairs (off-diagonal)  
                    negative_logits = logits[~positive_mask]
                    
                    # SigLIP sigmoid loss
                    positive_loss = -F.logsigmoid(positive_logits).mean()
                    negative_loss = -F.logsigmoid(-negative_logits).mean()
                    
                    return positive_loss + negative_loss

                def forward(self, patches: torch.Tensor, texts: List[str]):
                    device = self.device
                    patches = patches.to(device)
                        
                    # Extract image features
                    image_features = self.image_model.forward_features(patches)
                    
                    # Global image representation for SigLIP
                    global_image_features = self.global_pool(image_features).flatten(1)
                    global_image_embed = self.image_global_projector(global_image_features)
                    
                    # Patch-level features for text generation
                    patch_features = einops.rearrange(image_features, "bs dim w h -> bs (w h) dim")
                    patch_embeddings = self.patch_projector(patch_features)

                    # Tokenize text
                    tokenized = self.tokenizer(
                        [text + self.tokenizer.eos_token for text in texts],
                        padding=True,
                        truncation=True,
                        max_length=CFG.MAX_LENGTH,
                        return_tensors="pt",
                    )
                    tokenized = {k: v.to(device) for k, v in tokenized.items()}
                    text_embeddings = self.language_model.get_input_embeddings()(tokenized["input_ids"])
                    
                    # Global text representation for SigLIP (mean pooling of text embeddings)
                    text_mask = tokenized["attention_mask"].unsqueeze(-1).float()
                    global_text_features = (text_embeddings * text_mask).sum(dim=1) / text_mask.sum(dim=1)
                    global_text_embed = self.text_global_projector(global_text_features)

                    # Concatenate embeddings for text generation (prompt + patches + text)
                    generation_embeddings = torch.cat([
                        self.prompt_embeddings.to(device).expand(patches.size(0), -1, -1),
                        patch_embeddings,
                        text_embeddings,
                    ], dim=1)

                    # Create attention mask
                    prompt_mask = torch.ones(patches.size(0), self.prompt_embeddings.size(1), device=device)
                    patch_mask = torch.ones(patches.size(0), patch_embeddings.size(1), device=device)
                    attention_mask = torch.cat([prompt_mask, patch_mask, tokenized["attention_mask"]], dim=1)

                    # Create labels (only compute loss on text tokens, mask prompt and patches)
                    prompt_labels = torch.full((patches.size(0), self.prompt_embeddings.size(1)), CFG.LABEL_MASK, device=device)
                    patch_labels = torch.full((patches.size(0), patch_embeddings.size(1)), CFG.LABEL_MASK, device=device)
                    text_labels = tokenized["input_ids"].clone()
                    text_labels[tokenized["attention_mask"] == 0] = CFG.LABEL_MASK  # Mask padding tokens

                    labels = torch.cat([prompt_labels, patch_labels, text_labels], dim=1)

                    # Forward pass for text generation
                    lm_outputs = self.language_model(
                        inputs_embeds=generation_embeddings,
                        attention_mask=attention_mask,
                        labels=labels
                    )

                    # Compute SigLIP contrastive loss
                    contrastive_loss = self.siglip_loss(global_image_embed, global_text_embed)

                    # Combine losses
                    generation_loss = lm_outputs.loss
                    total_loss = generation_loss + CFG.SIGLIP_WEIGHT * contrastive_loss

                    # Return custom output object
                    return type('ModelOutput', (), {
                        'loss': total_loss,
                        'generation_loss': generation_loss,
                        'contrastive_loss': contrastive_loss,
                        'logits': lm_outputs.logits
                    })()

                def generate(self, patches: torch.Tensor, generator_kwargs: dict[str, Union[int, float]]):
                    device = self.device
                    patches = patches.to(device)

                    # Extract patch-level features for conditioning
                    image_features = self.image_model.forward_features(patches)
                    patch_features = einops.rearrange(image_features, "bs dim w h -> bs (w h) dim")
                    patch_embeddings = self.patch_projector(patch_features)

                    # Create generation embeddings (prompt + patches only, no text)
                    embeddings = torch.cat([
                        self.prompt_embeddings.to(device).expand(patches.size(0), -1, -1),
                        patch_embeddings,
                    ], dim=1)

                    # Create attention mask
                    prompt_mask = torch.ones(patches.size(0), self.prompt_embeddings.size(1), device=device)
                    patch_mask = torch.ones(patches.size(0), patch_embeddings.size(1), device=device)
                    attention_mask = torch.cat([prompt_mask, patch_mask], dim=1)

                    return self.language_model.generate(
                        inputs_embeds=embeddings,
                        attention_mask=attention_mask,
                        **generator_kwargs
                    )

            class ImprovedLightningModule(LIGHTNING.LightningModule):
                def __init__(self, model: ImprovedModel):
                    super().__init__()
                    self.model = model
                    self.lr = 2e-5
                    self.lr1 = 2e-4
                    
                def training_step(self, batch, batch_idx):
                    patches, texts = batch
                    outputs = self.model(patches, texts)
                    
                    # Log individual losses for monitoring
                    self.log("train_loss", outputs.loss, prog_bar=True)
                    self.log("generation_loss", outputs.generation_loss , prog_bar=True)
                    self.log("contrastive_loss", outputs.contrastive_loss , prog_bar=True)
                    
                    return outputs.loss

                def configure_optimizers(self) -> torch.optim.Optimizer:
                    params = [
                        {"params": self.model.language_model.parameters(), "lr": self.lr1},
                        {"params": self.model.image_model.parameters(), "lr": self.lr},
                        {"params": self.model.patch_projector.parameters(), "lr": self.lr},
                        {"params": self.model.image_global_projector.parameters(), "lr": self.lr},
                        {"params": self.model.text_global_projector.parameters(), "lr": self.lr},
                    ]
                    optimizer = torch.optim.Adam(params)
                    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                        optimizer,
                        max_lr=[param_group["lr"] for param_group in optimizer.param_groups],
                        total_steps=self.trainer.estimated_stepping_batches,
                    )
                    return [optimizer], [scheduler]
                                
            class PrintLossCallback(Callback):
                def on_train_epoch_end(self, trainer, pl_module):
                    metrics = trainer.callback_metrics
                    train_loss = metrics.get("train_loss")
                    gen_loss = metrics.get("generation_loss") 
                    cont_loss = metrics.get("contrastive_loss")
                    
                    if train_loss is not None:
                        print(f"Epoch {trainer.current_epoch} - Total Loss: {train_loss.item():.4f}")
                    if gen_loss is not None:
                        print(f"Epoch {trainer.current_epoch} - Generation Loss: {gen_loss.item():.4f}")
                    if cont_loss is not None:
                        print(f"Epoch {trainer.current_epoch} - Contrastive Loss: {cont_loss.item():.4f}")

                def on_train_end(self, trainer, pl_module):
                    metrics = trainer.callback_metrics
                    train_loss = metrics.get("train_loss")
                    gen_loss = metrics.get("generation_loss")
                    cont_loss = metrics.get("contrastive_loss")
                    
                    print("="*50)
                    print("FINAL TRAINING RESULTS:")
                    if train_loss is not None:
                        print(f"Final Total Loss: {train_loss.item():.4f}")
                    if gen_loss is not None:
                        print(f"Final Generation Loss: {gen_loss.item():.4f}")
                    if cont_loss is not None:
                        print(f"Final Contrastive Loss: {cont_loss.item():.4f}")
                    print("="*50)

            trainer = LIGHTNING.Trainer(
                max_epochs=CFG.EPOCHS,
                accumulate_grad_batches=CFG.GRAD_ACC,
                gradient_clip_val=1.0,
                enable_progress_bar=True,                           
                log_every_n_steps=50,
                enable_checkpointing=False,
                callbacks=[PrintLossCallback()]
            )
            
            model = ImprovedModel(image_model, language_model1, tokenizer)
            lightning_module = ImprovedLightningModule(model)
            
            print("STARTING TRAINING WITH SIGLIP + MASKED GENERATION LOSS...")
            print("="*100)
            
            trainer.fit(
                model=lightning_module,
                train_dataloaders=train_dl,
            )
            
            print("TRAINING COMPLETED! STARTING INFERENCE ON 10 SAMPLES...")
            print("="*100)
            
            patches_batch = np.array(patches[-10:]) 
            patches_batch = torch.tensor(patches_batch)
            
            generator_kwargs = {
                "max_new_tokens": 50,
                "do_sample": True,
                "temperature": 0.4,
                "pad_token_id": tokenizer.eos_token_id
            }
            
            print("GENERATING CAPTIONS FOR 10 SAMPLE IMAGES:")
            print("="*100)
            
            with torch.no_grad():
                generated_ids = model.generate(patches_batch, generator_kwargs)

            generated_texts = []
            from tqdm import tqdm
            
            for i in tqdm(range(generated_ids.shape[0])):
                prompt_length = model.prompt_embeddings.shape[1]
                patch_length = patches_batch.shape[1] * patches_batch.shape[2] * patches_batch.shape[3] // model.image_model.num_features
                start_idx = prompt_length + patch_length
                
                new_tokens = generated_ids[i, start_idx:]
                text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                generated_texts.append(text.strip())
                
                print(f"Sample {i+1}: {text.strip()}")
                print(text)
                print("-" * 80)

            print("="*100)
            print("INFERENCE COMPLETED!")
            print("="*100)

            # Save trained models
            lightning_module.model.language_model.save_pretrained(trained_language_model_path)
            tokenizer.save_pretrained(trained_language_model_path)
            
            # Save vision components
            torch.save(lightning_module.model.image_model.state_dict(), str(trained_vision_model_path / "vision_model.pth"))
            torch.save(lightning_module.model.patch_projector.state_dict(), str(trained_vision_model_path / "patch_projector.pth"))
            torch.save(lightning_module.model.image_global_projector.state_dict(), str(trained_vision_model_path / "image_global_projector.pth"))
            torch.save(lightning_module.model.text_global_projector.state_dict(), str(trained_vision_model_path / "text_global_projector.pth"))
            
            # Clean up dataset files
            # shutil.rmtree(files_path)

        train_result = start_training()

        # ----------------------------------------- CLEANUP ------------------------------------------
        # shutil.rmtree(trained_language_model_path)
        # shutil.rmtree(trained_vision_model_path)
        output_vol.commit()

        print("FINAL GPU STATUS:")
        print("="*100)
        run_nvidia_smi()

    except Exception as e:
        print(f"ERROR DURING WORKFLOW: {str(e)}")
        print("="*100)
        raise e
    
    print("COMPLETE SIGLIP TRAINING WORKFLOW FINISHED SUCCESSFULLY!")
    print("="*100)

@app.local_entrypoint()
def main():
    train_and_upload.remote()

if __name__ == "__main__":
    train_and_upload.remote()