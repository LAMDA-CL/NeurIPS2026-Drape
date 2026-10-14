# Adopted from LLaVA and FastChat
# Modified for DRAPE (Dynamic Generative Tuning)
import sys

import copy
import json
import logging
import os
import pathlib
import random
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from DRAPE.peft.tuners.drape import DRAPEModel
from DRAPE.checkpoint import ADAPTER_WEIGHTS_NAME, PROJECTOR_WEIGHTS_NAME
import torch
import torch.distributed as dist
import transformers
from torch.utils.data import Dataset

import deepspeed
from DRAPE.peft import get_peft_model, DRAPEConfig
from DRAPE.peft.utils import WEIGHTS_NAME, set_peft_model_state_dict

from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

# replace_llama_attn_with_flash_attn()

# LLaVA Imports
from llava import conversation as conversation_lib
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
)
from llava.mm_utils import tokenizer_image_token
from llava.model import *
from llava.train.llava_trainer import LLaVATrainer

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="lmsys/vicuna-7b-v1.5")
    previous_task_model_path: Optional[str] = field(default=None)
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-2)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "pad"
    memory_data_path: str = field(
        default=None, metadata={"help": "Path to the memory data."}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length."},
    )
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    is_SAT: bool = field(
        default=False,
        metadata={"help": "whether SAT data (different group_by_modality_length)"},
    )
    # === DRAPE Specific Arguments ===
    drape_enable: bool = field(default=True, metadata={"help": "Enable DRAPE Tuning"})
    drape_num_tokens: int = field(default=10, metadata={"help": "Number of generated soft prompts per instance"})
    drape_hidden_size: int = field(default=512, metadata={"help": "Hidden size of the DRAPE generator"})
    drape_dropout: float = field(default=0.1, metadata={"help": "Dropout for DRAPE generator"})
    drape_generator_type: str = field(default="CROSS_ATTENTION", metadata={"help": "Architecture: CROSS_ATTENTION or MLP_MIXER"})
    drape_skip_vision_attn: bool = field(default=False, metadata={"help": "Ablation: skip vision cross-attention"})
    drape_pool_mode: str = field(default="segment", metadata={"help": "Ablation: text pooling mode: segment or mean"})
    drape_query_mode: str = field(
        default="segpool",
        metadata={"help": "Prompt query mode: segpool or learnable_mm"},
    )
    drape_static_prompt: bool = field(default=False, metadata={"help": "Ablation: use per-task static prompt table"})
    drape_prefix_tuning: bool = field(default=False, metadata={"help": "Use KV cache prefix injection instead of input-level soft prompt"})
    drape_target_layers: Optional[str] = field(default=None, metadata={"help": "Comma-separated layer indices for prefix KV injection, e.g. '0,8,16,24'. None = all layers"})
    # === Continual Learning Arguments ===
    enable_continual_learning: bool = field(default=False, metadata={"help": "Enable continual learning with null-space projection and CLIP routing"})
    current_task_name: Optional[str] = field(default=None, metadata={"help": "Name of the current task (e.g., 'RS', 'Med', 'AD')"})
    clip_model_name: str = field(default="openai/clip-vit-base-patch32", metadata={"help": "CLIP model for task routing"})
    router_num_centers: int = field(default=1, metadata={"help": "Number of centers per task for CLIP router"})
    router_kmeans_iters: int = field(default=20, metadata={"help": "K-means iterations when router_num_centers > 1"})
    router_k1_center_mode: str = field(default="learned", metadata={"help": "K=1 center mode: mean or learned"})
    null_space_threshold: float = field(default=0.99, metadata={"help": "SVD explained variance threshold for null-space projection"})
    train_mm_projector_with_null_space: bool = field(default=True, metadata={"help": "Apply null-space gradient projection to mm_projector gradients during continual learning"})


# -----------------------------------------------------------------------------
# Data Preprocessing (Standard LLaVA Logic - Kept Intact)
# -----------------------------------------------------------------------------
def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

from llava.train.train import (
    preprocess_multimodal,
    preprocess_v1,
    preprocess,
    LazySupervisedDataset,
    DataCollatorForSupervisedDataset,
)

_ROUTE_IMAGE_TAGS = ("<image>", "<im_start>", "<im_end>", "<im_patch>")


def _extract_route_text_from_sample(sample: Dict) -> str:
    """Extract question-only text for router (no system prompt/template)."""
    convs = sample.get("conversations", [])
    if isinstance(convs, list):
        for c in convs:
            if isinstance(c, dict) and c.get("from") == "human":
                text = c.get("value", "")
                if not isinstance(text, str):
                    continue
                for tag in _ROUTE_IMAGE_TAGS:
                    text = text.replace(tag, " ")
                text = " ".join(text.split()).strip()
                if text:
                    return text
    return ""


class DRAPELazySupervisedDataset(LazySupervisedDataset):
    """Add route_text so router can consume question-only text during training."""

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        data_dict = super().__getitem__(i)
        sample = self.list_data_dict[i]
        data_dict["route_text"] = _extract_route_text_from_sample(sample)
        return data_dict


@dataclass
class DRAPEDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Keep standard collation and append route_texts as a list[str]."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(instances)
        if "route_text" in instances[0]:
            batch["route_texts"] = [ins.get("route_text", "") for ins in instances]
        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make DRAPE dataset/collator with route_texts for router feature caching."""
    train_dataset = DRAPELazySupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args
    )
    data_collator = DRAPEDataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )

def _is_valid_path(path):
    """Check if path is a valid previous model path (not None or sentinel)."""
    return path is not None and path not in ("", "None", "null", "NO_PREVIOUS_MODEL")


def load_model_from_previous_task(model, previous_task_model_path):
    """
    Load DRAPE generator weights and mm_projector from a previous task checkpoint.
    At this point `model` is already a DRAPEModel, so we load directly into its generator.
    """
    if not _is_valid_path(previous_task_model_path):
        return

    # 1. Load DRAPE generator weights
    adapter_path = os.path.join(previous_task_model_path, ADAPTER_WEIGHTS_NAME)
    if os.path.exists(adapter_path):
        rank0_print(f"Loading DRAPE generator from {adapter_path}")
        weights = torch.load(
            adapter_path,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        model.drape_generator.load_state_dict(weights, strict=False)
    else:
        rank0_print(f"Warning: {ADAPTER_WEIGHTS_NAME} not found at {previous_task_model_path}")

    # 2. Load mm_projector weights if available
    projector_path = os.path.join(previous_task_model_path, PROJECTOR_WEIGHTS_NAME)
    if os.path.exists(projector_path):
        rank0_print(f"Loading mm_projector from {projector_path}")
        projector_weights = torch.load(projector_path, map_location="cpu")
        base_model = model.model if hasattr(model, "model") else model
        base_model.load_state_dict(projector_weights, strict=False)


def train():
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    # 1. Setup Compute Dtype
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # 2. BitsAndBytes Config (For 4bit/8bit Base Model)
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            ),
        ))

    # 3. Load Base Model (LLaVA)
    rank0_print("Loading LLaVA Base Model...")    
    if model_args.vision_tower is not None:
        if "mpt" in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(
                model_args.model_name_or_path, trust_remote_code=True
            )
            config.attn_config["attn_impl"] = training_args.mpt_attn_impl
            model = LlavaMPTForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args,
            )
        else:
            config = transformers.AutoConfig.from_pretrained(
                    model_args.model_name_or_path
                )
            config.mm_vision_tower = model_args.vision_tower
            if not hasattr(config, "mm_hidden_size") or getattr(config, "mm_hidden_size", None) is None:
                vt_cfg = transformers.AutoConfig.from_pretrained(
                    model_args.vision_tower,
                    trust_remote_code=True,
                    local_files_only=True,
                )
                config.mm_hidden_size = int(getattr(vt_cfg, "hidden_size"))
            if not hasattr(config, "mm_projector_type"):
                config.mm_projector_type = model_args.mm_projector_type
            if not hasattr(config, "mm_vision_select_layer"):
                config.mm_vision_select_layer = model_args.mm_vision_select_layer
            if not hasattr(config, "mm_vision_select_feature"):
                config.mm_vision_select_feature = model_args.mm_vision_select_feature
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                low_cpu_mem_usage=True,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args,
            )
            mm_projector = getattr(model.get_model(), "mm_projector", None)
            if mm_projector is not None and any(getattr(p, "is_meta", False) for p in mm_projector.parameters()):
                from llava.model.multimodal_projector.builder import build_vision_projector
                rank0_print("Rebuilding mm_projector to materialize meta parameters...")
                model.get_model().mm_projector = build_vision_projector(config)
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args,
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    # 4. Prepare for k-bit training if needed
    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # 5. Initialize Tokenizer & Conversation
    if "mpt" in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                model_args.version
            ]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                "vicuna_v1"
            ]

    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = (
        model_args.mm_use_im_start_end
    )
    model.config.mm_projector_lr = training_args.mm_projector_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token

    # 6. Initialize Vision Tower
    rank0_print("Initializing Vision Tower...")
    if model_args.vision_tower is not None:  # add vision tower here
        model.get_model().initialize_vision_modules(
            model_args=model_args, fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(
            dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            device=training_args.device,
        )

        # 7. DRAPE Integration (The Core Change)
        if training_args.drape_enable:
            rank0_print("Initializing DRAPE Configuration...")
            
            # Get Vision Dim
            if hasattr(vision_tower, "config"):
                raw_vision_dim = vision_tower.config.hidden_size
            elif hasattr(vision_tower, "hidden_size"):
                raw_vision_dim = vision_tower.hidden_size
            else:
                # Fallback for CLIP-L/14 if config attribute is missing
                raw_vision_dim = 1024 
                
            # Get Text Dim
            text_dim = model.config.hidden_size
            
            vision_dim = 4096

            # B. Config Initialization
            # Parse target_layers from comma-separated string
            target_layers = None
            if training_args.drape_target_layers is not None:
                target_layers = [int(x.strip()) for x in training_args.drape_target_layers.split(",")]

            # Resolve previous_task_model_path
            prev_model_dir = model_args.previous_task_model_path if _is_valid_path(model_args.previous_task_model_path) else None

            drape_config = DRAPEConfig(
                num_virtual_tokens=training_args.drape_num_tokens,
                generator_type=training_args.drape_generator_type,
                generator_hidden_size=training_args.drape_hidden_size,
                generator_dropout=training_args.drape_dropout,
                skip_vision_attn=training_args.drape_skip_vision_attn,
                pool_mode=training_args.drape_pool_mode,
                query_mode=training_args.drape_query_mode,
                static_prompt=training_args.drape_static_prompt,
                vision_input_dim=vision_dim,
                text_input_dim=text_dim,
                task_type="CAUSAL_LM",

                # Continual Learning parameters
                enable_continual_learning=training_args.enable_continual_learning,
                current_task_name=training_args.current_task_name,
                clip_model_name=training_args.clip_model_name,
                router_num_centers=training_args.router_num_centers,
                router_kmeans_iters=training_args.router_kmeans_iters,
                router_k1_center_mode=training_args.router_k1_center_mode,
                null_space_threshold=training_args.null_space_threshold,
                apply_null_space_projection=training_args.train_mm_projector_with_null_space,
                previous_task_model_dir=prev_model_dir,
            )

            # C. Wrap Model
            rank0_print("Wrapping Base Model with DRAPEModel...")
            # Note: This calls the get_peft_model -> finds "DRAPE" in mapping -> instantiates DRAPEModel
            model = get_peft_model(model, drape_config)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = (
            model_args.tune_mm_mlp_adapter
        )
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        # In CL mode, mm_projector is trainable by default.
        if training_args.enable_continual_learning:
            if training_args.train_mm_projector_with_null_space:
                rank0_print("[CL] Enabling mm_projector training with null-space gradient projection")
            else:
                rank0_print("[CL] Enabling mm_projector training without null-space gradient projection")
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        # IMPORTANT:
        # DRAPEModel initializes CL hooks before mm_projector is unfrozen in this trainer.
        # Refresh null-space hooks here so they attach to the now-trainable mm_projector params.
        if hasattr(model, "_setup_null_space_hooks"):
            model._setup_null_space_hooks()

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(
                dtype=compute_dtype, device=training_args.device
            )

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = (
            model_args.mm_use_im_start_end
        )
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    if local_rank==0:
        model.print_trainable_parameters()
    for name, param in model.named_parameters():
        if param.requires_grad:
            rank0_print(name)
            
    if _is_valid_path(model_args.previous_task_model_path):
        # load model from previous task
        load_model_from_previous_task(model, model_args.previous_task_model_path)
        
        
        
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = LLaVATrainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    # 10. Start Training
    rank0_print("Starting Training...")
    trainer.train()

    # 11. Save State
    trainer.save_state()
    model.config.use_cache = True

    # 12. Continual Learning Post-Training Steps
    if training_args.enable_continual_learning and training_args.current_task_name:
        rank0_print(f"[CL] Running post-training continual learning steps for task '{training_args.current_task_name}'...")
        _run_continual_learning_post_training(
            model, tokenizer, data_args, training_args, training_args.current_task_name
        )

    # 13. Save DRAPE Model
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        rank0_print(f"Saving DRAPE model to {training_args.output_dir}")
        model.config.save_pretrained(training_args.output_dir)
        # This calls DRAPEModel.save_pretrained which saves 'adapter_model.bin' + CL state
        model.base_model.save_pretrained(training_args.output_dir)



def _run_continual_learning_post_training(model, tokenizer, data_args, training_args, task_name):
    """
    Post-training steps for continual learning:
    1. Compute and update the task center using CLIP text encoder
    2. Compute null-space projection for mm_projector
    3. Save all CL state
    """
    device = training_args.device
    dist_ready = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if dist_ready else 1
    rank = dist.get_rank() if dist_ready else 0

    # 1. Update task center with fused (text + image) features from training-time CPU cache.
    if (
        hasattr(model, "update_task_center_from_embeddings")
        and hasattr(model, "clip_router")
        and model.clip_router is not None
    ):
        rank0_print(f"[CL] Computing fused task center for '{task_name}' from training cache...")
        fused_embeddings = None
        if hasattr(model, "pop_cached_fused_embeddings"):
            fused_embeddings = model.pop_cached_fused_embeddings(task_name)
            if fused_embeddings is not None and fused_embeddings.numel() > 0:
                rank0_print(
                    f"[CL] Using local cached fused embeddings before gather: {fused_embeddings.shape[0]} samples"
                )

        if dist_ready:
            local_payload = None
            if fused_embeddings is not None and fused_embeddings.numel() > 0:
                local_payload = fused_embeddings.detach().float().cpu()
            gathered = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, local_payload)
            if rank == 0:
                gathered_tensors = [
                    t for t in gathered if torch.is_tensor(t) and t.numel() > 0
                ]
                if len(gathered_tensors) > 0:
                    fused_embeddings = torch.cat(gathered_tensors, dim=0)
                    rank0_print(
                        f"[CL] Gathered fused embeddings across {world_size} ranks: {fused_embeddings.shape[0]} samples"
                    )
                else:
                    fused_embeddings = None
            else:
                fused_embeddings = None

        if rank == 0 and fused_embeddings is not None and fused_embeddings.numel() > 0:
            model.update_task_center_from_embeddings(task_name, fused_embeddings, device)
            rank0_print(f"[CL] Fused task center updated with {fused_embeddings.shape[0]} samples")
        elif rank == 0:
            rank0_print("[CL] Warning: No cached fused embeddings found; skip task center update")

    # 2. Compute null-space projection for mm_projector (optional)
    if rank == 0 and hasattr(model, "compute_and_save_null_space"):
        if training_args.train_mm_projector_with_null_space:
            rank0_print(f"[CL] Computing null-space projection...")
            model.compute_and_save_null_space(training_args.output_dir)
        else:
            rank0_print("[CL] Null-space projection disabled: skip covariance/projection update.")

    # 3. Null-space hook application is controlled by apply_null_space_projection.
    #    If enabled, gradient hooks are already active during training.
    rank0_print(f"[CL] Post-training continual learning steps completed for task '{task_name}'")

if __name__ == "__main__":
    train()
