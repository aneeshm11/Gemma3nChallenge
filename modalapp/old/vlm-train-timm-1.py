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
    TRAIN_END = 700
    TRAIN_BATCH_SIZE = 5

    VAL_START = 600
    VAL_END = 700
    VAL_BATCH_SIZE = 5
    
    EPOCHS = 5*4
    GRAD_ACC = 1
    
    MAX_LENGTH = 50
    LABEL_MASK = -100  # for loss calculation
    # VM_LR = 1e-4
    # LM_LR = 1e-4

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
    # from pytorch_lightning.callbacks import ModelCheckpoint
    from lightning.pytorch.callbacks import Callback

    import shutil
    from typing import List, Union , Dict
    # LIGHTNING.seed_everything(42)
    from peft import LoraConfig, get_peft_model, TaskType
    from huggingface_hub import snapshot_download, login, create_repo, upload_folder



    # ----------------------------------------- HUGGING FACE SETUP AND KERNEL INFORMATION ------------------------------------------

    HF_API_KEY        = "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"
    login(HF_API_KEY)
    
    username, model_repo_name = CFG.hf_username, CFG.hf_dataset_name
    base_model_repo = CFG.base_model_repo

    print("STARTING COMPLETE TRAINING WORKFLOW...")    

    local_model_path            = VOL_MOUNT_PATH / "base_model"
    trained_language_model_path = VOL_MOUNT_PATH / "trained_lm"
    trained_vision_model_path   = VOL_MOUNT_PATH / "trained_vm"
    # lora_adapters_path          = VOL_MOUNT_PATH / "lora_adapters"
    
    local_model_path.mkdir(exist_ok=True)
    trained_language_model_path.mkdir(exist_ok=True)
    trained_vision_model_path.mkdir(exist_ok=True)
    # lora_adapters_path.mkdir(exist_ok=True)


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
            
            # local_model_path = "meta-llama/Llama-3.2-3B-Instruct"

            device = torch.device("cuda:0")
            language_model = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                trust_remote_code=True,
                device_map="cuda:0",                
                low_cpu_mem_usage=True,
                # max_memory=max_memory_map
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

                        # target_modules= [
                        #     "q_proj",
                        #     "k_proj",
                        #     "v_proj",
                        #     "o_proj",
                        # ],
                    
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
            
            
            # path = "timm/mobilenetv4_conv_medium.e500_r256_in1k"
            path = "timm/tf_efficientnet_b7.aa_in1k"
            image_model = timm.create_model(path, pretrained=True).to(device) 
            config = timm.data.resolve_data_config({}, model=image_model)
            transform = timm.data.transforms_factory.create_transform(**config, is_training=False)
            

            files_path = VOL_MOUNT_PATH / "reg_files"
            files_path.mkdir(exist_ok=True)
    
            downloaded_path = snapshot_download(
                repo_id="aneeshm44/reg",
                repo_type="dataset",
                local_dir=str(files_path),
                local_dir_use_symlinks=False,
                resume_download=True
            )
    
            patches = np.load(files_path / "small.npy" , mmap_mode='r')
    
            with open(files_path / "filenames.json", "r") as f:
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
            val_data   = REGDataset(patches, texts, CFG.VAL_START, CFG.VAL_END)
            
            train_dl = DataLoader(train_data, CFG.TRAIN_BATCH_SIZE, pin_memory=True, shuffle=True, drop_last=True)
            valid_dl = DataLoader(val_data, CFG.VAL_BATCH_SIZE, pin_memory=True, shuffle=False, drop_last=False)
            
            class Model(nn.Module):
                def __init__(self, image_model, language_model, tokenizer, prompt="Describe the medical image:"):
                    super().__init__()
                    self.image_model = image_model 
                    self.language_model = language_model 
                    self.tokenizer = tokenizer
                    self.eos_token = tokenizer.eos_token
                    self.prompt = prompt
            
                    self.config = timm.data.resolve_data_config({}, model=self.image_model)
                    self.transform = timm.data.transforms_factory.create_transform(**self.config, is_training=False)
                    self.projector = nn.Sequential(
                        *projection_layers(image_model.num_features, 2048, 6)
                    ).to(device) 
            
                    # tokenize get embeddings of prompt
                    prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")
                    prompt_embeddings = language_model.get_input_embeddings()(prompt_tokens).detach() 
                    self.register_buffer('prompt_embeddings', prompt_embeddings)
            
                @property
                def device(self):
                    return next(self.parameters()).device
            
                def forward(self, patches: torch.Tensor, texts: List[str]):
                    device = self.device
                    patches = patches .to(device)
                        
                    image_features = self.image_model.forward_features(patches)
                    image_features = einops.rearrange(image_features, "bs dim w h -> bs (w h) dim")
                    patch_embeddings = self.projector(image_features)

                    tokenized = self.tokenizer(
                        [text + self.tokenizer.eos_token for text in texts],
                        padding=True,
                        truncation=True,
                        max_length=CFG.MAX_LENGTH,
                        return_tensors="pt",
                    )
                    tokenized = {k: v.to(device) for k, v in tokenized.items()}
            
                    text_embeddings = self.language_model.get_input_embeddings()(tokenized["input_ids"]) 
            
                    #  inputs are embeddings of ->  prompt + image patches + text
                    embeddings = torch.cat([
                        self.prompt_embeddings.to(device).expand(patches.size(0), -1, -1),
                        patch_embeddings,
                        text_embeddings,
                    ], dim=1)
            
                    # attention mask
                    prompt_mask = torch.ones(patches.size(0), self.prompt_embeddings.size(1), device=device)
                    patch_mask = torch.ones(patches.size(0), patch_embeddings.size(1), device=device)
                    attention_mask = torch.cat([prompt_mask, patch_mask, tokenized["attention_mask"]], dim=1)
            
                    # labels
                    prompt_labels = torch.full((patches.size(0), self.prompt_embeddings.size(1)), CFG.LABEL_MASK, device=device)
                    patch_labels = torch.full((patches.size(0), patch_embeddings.size(1)), CFG.LABEL_MASK, device=device)
                    text_labels = tokenized["input_ids"].clone()
            
                    labels = torch.cat([prompt_labels, patch_labels, text_labels], dim=1)
                    labels[attention_mask == 0] = CFG.LABEL_MASK  #  padding
            
                    return self.language_model(
                        inputs_embeds=embeddings,
                        attention_mask=attention_mask,
                        labels=labels
                    )
            
                def generate(self, patches: torch.Tensor, generator_kwargs: dict[str, Union[int, float]]):
                    device = self.device
                    patches = patches .to(device)
            
                    image_features = self.image_model.forward_features(patches)
                    image_features = einops.rearrange(image_features, "bs dim w h -> bs (w h) dim")
                    patch_embeddings = self.projector(image_features)
            
                    embeddings = torch.cat([
                        self.prompt_embeddings.to(device).expand(patches.size(0), -1, -1),
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
                    self.lr = 2e-4
                    
                def training_step(self, batch, batch_idx):
                    patches, texts = batch
                    patches = patches   
                    out = self.model(patches, texts)
                    self.log("train_loss", out.loss)
                    return out.loss
            
                def configure_optimizers(self) -> torch.optim.Optimizer:
                    params = [
                        {"params": self.model.language_model.parameters(), "lr": self.lr },
                        {"params": self.model.image_model.parameters(), "lr": self.lr },
                        {"params": self.model.projector.parameters(), "lr": self.lr},
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
                    loss = trainer.callback_metrics.get("train_loss")
                    if loss is not None:
                        print(f"Epoch {trainer.current_epoch} - Train Loss: {loss.item():.4f}")
            
                def on_validation_epoch_end(self, trainer, pl_module):
                    val_loss = trainer.callback_metrics.get("val_loss")
                    if val_loss is not None:
                        print(f"Epoch {trainer.current_epoch} - Val Loss: {val_loss.item():.4f}")
            
                def on_train_end(self, trainer, pl_module):
                    train_loss = trainer.callback_metrics.get("train_loss")
                    val_loss = trainer.callback_metrics.get("val_loss")
                    if train_loss is not None:
                        print(f"Final Train Loss: {train_loss.item():.4f}")
                    if val_loss is not None:
                        print(f"Final Val Loss: {val_loss.item():.4f}")
            
            
            trainer = LIGHTNING.Trainer(
                max_epochs=  CFG.EPOCHS,
                accumulate_grad_batches= CFG.GRAD_ACC,
                gradient_clip_val=1.0,
                enable_progress_bar=True,                           
                log_every_n_steps=100,
                enable_checkpointing=False,
                callbacks=[PrintLossCallback()]
            
            )
            
            model = Model(image_model, language_model1, tokenizer )
            lightning_module = LightningModule(model)
            
            
            trainer.fit(
                model = lightning_module,
                train_dataloaders = train_dl, 
                # val_dataloaders = valid_dl ,
                datamodule = None,
            )
            
            patches_batch = np.array(patches[-50:])
            patches_batch = torch.tensor(patches_batch)
            patches_batch.shape
            
            
            generator_kwargs = {
                "max_new_tokens": 50,
                "do_sample": True,
                "temperature": 0.4,
                "pad_token_id": tokenizer.eos_token_id
            }
            
            
            generated_ids = model.generate(patches_batch, generator_kwargs)

            generated_texts = []
            import tqdm
            from tqdm import tqdm
            for i in tqdm(range(generated_ids.shape[0])):
            
                prompt_length = model.prompt_embeddings.shape[1]
                patch_length = patches_batch.shape[1]
                start_idx = prompt_length + patch_length
                
                new_tokens = generated_ids[i, :]
                
                text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                print(text.strip())
                print("="*100)




            lightning_module.model.language_model.save_pretrained(trained_language_model_path)
            tokenizer.save_pretrained(trained_language_model_path)
            shutil.rmtree(files_path)


            torch.save(lightning_module.model.image_model.state_dict(), str(trained_vision_model_path / "vision_model.pth"))
            torch.save(lightning_module.model.projector.state_dict(), str(trained_vision_model_path / "projector_model.pth"))


        train_result = start_training()

        # ----------------------------------------- UPLOADING MODEL TO HUGGINGFACE ------------------------------------------

        print("UPLOADING TRAINED FILES TO HUGGINGFACE...")
        
        # dataset_repo_id = f"{username}/{model_repo_name}"
        
        # try:
        #     create_repo(
        #         repo_id=dataset_repo_id,
        #         repo_type="dataset",
        #         exist_ok=True
        #     )
        # except Exception as e:
        #     print(f"REPOSITORY CREATION ERROR: {e}")
        #     print("="*100)
        
        # shutil.rmtree(local_model_path)

        # upload_folder(
        #     folder_path= VOL_MOUNT_PATH,
        #     repo_id=dataset_repo_id,
        #     repo_type="dataset",
        #     commit_message=f"Upload trained model files for trial {number}",
        # )
        
        # ----------------------------------------- DELETING ALL FILES ON MODAL TO SAVE STORAGE COSTS ------------------------------------------

        shutil.rmtree(trained_language_model_path)
        shutil.rmtree(trained_vision_model_path)
        output_vol.commit()

        # ----------------------------------------- COMPLETION ------------------------------------------

        print("FINAL GPU STATUS:")
        print("="*100)
        run_nvidia_smi()

    except Exception as e:
        print(f"ERROR DURING WORKFLOW: {str(e)}")
        print("="*100)
        raise e
    
    print("COMPLETE TRAINING WORKFLOW FINISHED SUCCESSFULLY!")
    print("="*100)

@app.local_entrypoint()
def main():
    train_and_upload.remote()

if __name__ == "__main__":
    train_and_upload.remote()




