#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 bram-w, The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import contextlib
import io
import logging
import math
import os
import random
import shutil
from pathlib import Path
import pandas as pd
from datasets import Dataset
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from datasets import Image as D_Image
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import pandas as pd
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers
from diffusers.utils.import_utils import is_xformers_available

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.25.0.dev0")

logger = get_logger(__name__)


VALIDATION_PROMPTS = [
    "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
    "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
]


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def log_validation(args, unet, vae, accelerator, weight_dtype, epoch, global_step, is_final_validation=False,reference_model=None,fixed_noise=None):
    logger.info(f"Running validation... \n Generating images with prompts:\n" f" {VALIDATION_PROMPTS}.")

    if is_final_validation:
        if args.mixed_precision == "fp16":
            vae.to(weight_dtype)

    # create pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        unet=accelerator.unwrap_model(unet),
        revision=args.revision,
        variant=args.variant,
        safety_checker=None,
        torch_dtype=weight_dtype,
    )

    pipeline = pipeline.to(accelerator.device)
    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    images = []
    context = contextlib.nullcontext() if is_final_validation else torch.cuda.amp.autocast()

    for idx,prompt in enumerate(VALIDATION_PROMPTS):
        with context:
            if not args.fixed_noise:
                image = pipeline(prompt, num_inference_steps=25, generator=generator).images[0]
            else:
                image = pipeline(prompt, num_inference_steps=25, latents=fixed_noise[idx][:1].to(device=accelerator.device,dtype=weight_dtype)).images[0]
            images.append(image)
            
    tracker_key = "test" if is_final_validation else "validation"
    if accelerator.is_main_process:
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in images])
                tracker.writer.add_images(tracker_key, np_images, epoch, dataformats="NHWC")
            if tracker.name == "wandb":
                tracker.log(
                    {
                        tracker_key: [
                            wandb.Image(image, caption=f"{i}: {VALIDATION_PROMPTS[i]}") for i, image in enumerate(images)
                        ]
                    }
                )

    # Also log images without the LoRA params for comparison.
    if is_final_validation:
        if args.use_lora:
            pipeline.disable_lora()
        else:
            pipeline.unet = accelerator.unwrap_model(reference_model)
        no_lora_images = [
            pipeline(prompt, num_inference_steps=25, generator=generator).images[0] for prompt in VALIDATION_PROMPTS
        ]
        if accelerator.is_main_process:
            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in no_lora_images])
                    tracker.writer.add_images("test_without_lora", np_images, epoch, dataformats="NHWC")
                if tracker.name == "wandb":
                    tracker.log(
                        {
                            "test_without_lora": [
                                wandb.Image(image, caption=f"{i}: {VALIDATION_PROMPTS[i]}")
                                for i, image in enumerate(no_lora_images)
                            ]
                        }
                    )


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_split_name",
        type=str,
        default="validation",
        help="Dataset split to be used during training. Helpful to specify for conducting experimental runs.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--run_validation",
        default=False,
        action="store_true",
        help="Whether to run validation inference in between training and also after training. Helps to track progress.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=200,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="diffusion-dpo-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--vae_encode_batch_size",
        type=int,
        default=8,
        help="Batch size to use for VAE encoding of the images for efficient processing.",
    )
    parser.add_argument(
        "--no_hflip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--random_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to random crop the input images to the resolution. If not set, the images will be center-cropped."
        ),
    )
    parser.add_argument(
        "--use_ema",
        default=False,
        action="store_true",
        help=(
            "Whether to random crop the input images to the resolution. If not set, the images will be center-cropped."
        ),
    )
    parser.add_argument(
        "--reference_ema",
        default=False,
        action="store_true",
        help=(
            "Whether to random crop the input images to the resolution. If not set, the images will be center-cropped."
        ),
    )
    parser.add_argument(
        "--reference_ema_momentum",
        default=0.999,
        type=float,
        help=(
            "Whether to random crop the input images to the resolution. If not set, the images will be center-cropped."
        ),
    )    
    parser.add_argument(
        "--ema_momentum",
        default=0.998,
        type=float,
        help=(
            "Whether to random crop the input images to the resolution. If not set, the images will be center-cropped."
        ),
    )
    
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--beta_dpo",
        type=int,
        default=1000,
        help="DPO KL Divergence penalty.",
    )
    parser.add_argument(
        "--use_lora",
        action='store_true'
    )
    parser.add_argument(
        "--lambda_d_kto",
        type=float,
        default=1.0,
        help="KTO lambda_d",
    )
    parser.add_argument(
        "--lambda_u_kto",
        type=float,
        default=1.0,
        help="KTO lambda_u",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    
    parser.add_argument(
        "--eval_metrics",
        type=str,
        nargs='+',
        default= ['pick'],
    )
    
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--tracker_name",
        type=str,
        default="diffusion-kto",
        help=("The name of the tracker to report results to."),
    )
    parser.add_argument(
        "--dataloader",
        type=str,
        choices=['default','pick'],
        default='pick',
        help=("dataloader"),
    )
    parser.add_argument(
        "--policy",
        type=str,
        default='gt_label_wi_l',
        help=("dataloader"),
    )
    parser.add_argument(
        "--positive_ratio",
        type=float,
        default=0.5,
        help="postive sample ratio",
    )
    
    parser.add_argument(
        "--halo",
        type=str,
        default='sigmoid',
    )
    
    parser.add_argument(
        "--loss",
        type=str,
        choices=['sft','dpo','kto','kto_gp'],
        default='kto',
        help=("dataloader"),
    )
    
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default='',
        help=("dataloader"),
    )
    
    parser.add_argument(
        "--fixed_noise",
        action='store_true'
    )
    parser.add_argument(
        "--pick_split",
        type=str,
        default='train',
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0.2,
        help="postive sample ratio",
    )
    
    parser.add_argument(
        "--track_score",
        type=str,
        choices=[None,"laion","pick"],
        default=None,
        help=("Track score metric for validation images"),
    ) 
    
    parser.add_argument(
        "--h_pos",
        type=float,
        default=1.0,
       
    ) 
    parser.add_argument(
        "--h_neg",
        type=str,
        default=1.0,
       
    ) 
    

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None:
        raise ValueError("Must provide a `dataset_name`.")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def tokenize_captions(tokenizers, examples):
    captions = []
    caption_key = "coco_caption" if "coco_caption" in examples else "caption" 
    for idx,caption in enumerate(examples[caption_key]):
        if random.random() < args.proportion_empty_prompts:
            captions.append("")
        else:
            captions.append(caption)

    tokens_one = tokenizers[0](
        captions, truncation=True, padding="max_length", max_length=tokenizers[0].model_max_length, return_tensors="pt"
    ).input_ids
    tokens_two = tokenizers[1](
        captions, truncation=True, padding="max_length", max_length=tokenizers[1].model_max_length, return_tensors="pt"
    ).input_ids

    return tokens_one, tokens_two


@torch.no_grad()
def encode_prompt(text_encoders, text_input_ids_list):
    prompt_embeds_list = []
    text_input_ids = text_input_ids_list[0]
    text_input_ids = text_input_ids.to(text_encoders[0].device)
    attention_mask = None
    prompt_embeds = text_encoders[0](text_input_ids, attention_mask=attention_mask)
    prompt_embeds = prompt_embeds[0]
    return prompt_embeds,prompt_embeds

def rand_decision(p):
    return np.random.rand() < p
from diffusers.training_utils import EMAModel, compute_snr

def make_weights_for_balanced_classes(dataset):  
    nclasses = 2                      
    count = [0] * nclasses  
    labels = []  
    all_scores = []                                                  
    for item in tqdm(dataset):    
        label = int(item['label'])
        count[label] += 1  
        labels.append(label)                                           
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/(float(count[i])+1e-9)
    weight_per_class[1] *= args.positive_ratio
    weight_per_class[0] *= 1- args.positive_ratio                              
    weight = [0] * len(labels)                                              
    for idx, val in enumerate(labels):                                          
        weight[idx] = weight_per_class[val]  
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    hist, bin_edges = np.histogram(all_scores)
    hist_s =hist.sum()
    print('---------------DEBUG DATASET INFO') 
    print(f'label: pos {n_pos}  neg {n_neg} ({count})')
    print(f'weight: ({weight_per_class})')
    print(f'Distribution of scores: ')
    for i,v in enumerate(hist):
        print(f'({bin_edges[i],bin_edges[i+1]}) : {v} ({v/(hist_s+1e-9)}) ')
    print('--------------------------------')                               
    return weight                                                               

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
            args.pretrained_model_name_or_path, args.revision
    )
    tokenizer_two = tokenizer_one
    text_encoder_cls_two = text_encoder_cls_one


    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = text_encoder_one

    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )
   

    # We only train the additional adapter LoRA layers
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.train()
    fixed_noise = None

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet and text_encoders to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    # The VAE is always in float32 to avoid NaN losses.

    # Set up LoRA.
    if args.use_lora:
        unet_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        reference_model = torch.nn.Identity()
        unet.requires_grad_(False)
        unet.add_adapter(unet_lora_config)
    else:
        reference_model = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
        )
        reference_model = EMAModel(reference_model.parameters(), model_cls=UNet2DConditionModel, model_config=reference_model.config,decay=args.reference_ema_momentum)#.cpu()
    unet.train()
    
    # Create EMA for the unet.
    ema_unet = None
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
        )
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config,decay=args.ema_momentum)
        ema_unet.to(accelerator.device)


    if args.mixed_precision == "fp16" and args.use_lora:
        for param in unet.parameters():
            # only upcast trainable parameters (LoRA) into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder atten layers
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            if not args.use_lora:
                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))
                return
            unet_lora_layers_to_save = None
            models = [accelerator.unwrap_model(x) for x in models]
            for model in models:
                if isinstance(model, type(accelerator.unwrap_model(unet))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                try:
                    weights.pop()
                except:
                    pass


    def load_model_hook(models, input_dir):
        unet_ = None
        if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                del load_model

        if args.use_lora:
            raise NotImplementedError
            # Not support lora yet
        else:
            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    print('------------ Data Init --------------------')
    if not args.train_data_dir:
        train_dataset = load_dataset(
            args.dataset_name,
            cache_dir=args.cache_dir,
            split=args.dataset_split_name,
        )
    elif args.dataloader == 'pick':
        from datasets import Dataset
        pick_split = args.pick_split
        assert os.path.exists(ds:=os.path.join(args.train_data_dir,f'{pick_split}_processed_v3.csv')),f"unknown split {pick_split}"
        data = pd.read_csv(ds)
        data.caption = data.caption.fillna("")

        def resolve_path(examples):
            examples["file_name"] = list([os.path.join(args.train_data_dir,x) for x in examples["file_name"]])
            return examples

        if args.policy == 'gt_label_w_l':
            data = data[~(data.split=='intersections')]
            data['label'] = data.split == 'exclusive_win'
        elif args.policy == 'gt_label_wi_l': # move intersection to win
            data['label'] = (data.split == 'exclusive_win')|(data.split == 'intersections')
        elif args.policy == 'gt_label_w_il': # move intersection to lose
            data['label'] = (data.split == 'exclusive_win')
        else:
            raise NotImplemented

        print(f'--------------pick a pick policy: {args.policy}')
        num_wins = data['label'].sum()
        print("wins: {} loses: {} total: {}".format(num_wins,len(data)-num_wins,len(data)))
        print('-----------------------------------')
        data = Dataset.from_pandas(data).map(resolve_path, batched=True)
        train_dataset = data

    elif os.path.exists(os.path.join(args.train_data_dir,'metadata.csv')):
        ds = os.path.join(args.train_data_dir,'metadata.csv')
        from datasets import Dataset
        data = pd.read_csv(ds)
        def resolve_path(examples):
            examples["file_name"] = list([os.path.join(args.train_data_dir,x) for x in examples["file_name"]])
            return examples
        data = Dataset.from_pandas(data).map(resolve_path, batched=True)
        train_dataset = data
    else:
        train_dataset = load_dataset(
                "imagefolder",
                data_dir=args.train_data_dir,
                cache_dir=args.cache_dir,
        )['train']

    print('------------Data Loaded --------------------')
    # Preprocessing the datasets.
    train_resize = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR)
    train_crop = transforms.RandomCrop(args.resolution) if args.random_crop else transforms.CenterCrop(args.resolution)
    train_flip = transforms.RandomHorizontalFlip(p=1.0)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.5], [0.5])

    def preprocess_train(examples):
        all_pixel_values = []
        try:
            images = [img.convert("RGB") for img in examples["image"]]
        except:
                images = [Image.open(img).convert("RGB") for img in examples["file_name"]]
        original_sizes = [(image.height, image.width) for image in images]
        crop_top_lefts = []
        pixel_values = [to_tensor(image) for image in images]
        combined_pixel_values = []
        all_labels = []
        for idx,img in enumerate(pixel_values):
            if args.policy.startswith('gt_label'):
                label = examples['label'][idx]
            else:
                raise NotImplementedError
                

            all_labels.append(label)

            # Resize.
            combined_im = train_resize(img)

            # Cropping.
            if not args.random_crop:
                y1 = max(0, int(round((combined_im.shape[1] - args.resolution) / 2.0)))
                x1 = max(0, int(round((combined_im.shape[2] - args.resolution) / 2.0)))
                combined_im = train_crop(combined_im)
            else:
                y1, x1, h, w = train_crop.get_params(combined_im, (args.resolution, args.resolution))
                combined_im = crop(combined_im, y1, x1, h, w)

            # Flipping.
            if random.random() < 0.5:
                x1 = combined_im.shape[2] - x1
                combined_im = train_flip(combined_im)

            crop_top_left = (y1, x1)
            crop_top_lefts.append(crop_top_left)
            combined_im = normalize(combined_im)
            combined_pixel_values.append(combined_im)

        examples["pixel_values"] = combined_pixel_values
        examples["original_sizes"] = original_sizes
        examples["crop_top_lefts"] = crop_top_lefts
        examples['labels'] = all_labels
        tokens_one, tokens_two = tokenize_captions([tokenizer_one, tokenizer_two], examples)
        examples["input_ids_one"] = tokens_one
        examples["input_ids_two"] = tokens_two
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            train_dataset = train_dataset.shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        if args.policy.startswith('gt_label'):
            data_weights = make_weights_for_balanced_classes(train_dataset)
            data_weights = torch.tensor(data_weights)
        train_dataset = train_dataset.with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        original_sizes = [example["original_sizes"] for example in examples]
        crop_top_lefts = [example["crop_top_lefts"] for example in examples]
        input_ids_one = torch.stack([example["input_ids_one"] for example in examples])
        input_ids_two = torch.stack([example["input_ids_two"] for example in examples])
        labels = torch.tensor([example["labels"] for example in examples])

        return {
            "pixel_values": pixel_values,
            "input_ids_one": input_ids_one,
            "input_ids_two": input_ids_two,
            "original_sizes": original_sizes,
            "crop_top_lefts": crop_top_lefts,
            "labels":labels
        }

    if  args.policy.startswith('gt_label'):
        sampler = torch.utils.data.sampler.WeightedRandomSampler(data_weights, len(data_weights))
        sample_kwargs = dict(sampler=sampler)
    else:
        sample_kwargs = dict(shuffle=True,)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        **sample_kwargs,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    unet.train()
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # (batch_size, 2*channels, h, w) -> (2*batch_size, channels, h, w)
                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                feed_pixel_values =  pixel_values #torch.cat(pixel_values.chunk(2, dim=1))

                latents = []
                for i in range(0, feed_pixel_values.shape[0], args.vae_encode_batch_size):
                    latents.append(
                        vae.encode(feed_pixel_values[i : i + args.vae_encode_batch_size]).latent_dist.sample()
                    )
                latents = torch.cat(latents, dim=0)
                latents = latents * vae.config.scaling_factor
                if args.pretrained_vae_model_name_or_path is None:
                    latents = latents.to(weight_dtype)
                if not args.use_lora:
                    latents.to(weight_dtype)
                    
                latents = latents.detach()
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)#.chunk(2)[0]#.repeat(2, 1, 1, 1)
                if fixed_noise is None:
                    fixed_noise =  list([torch.randn_like(latents).detach().cpu() for _ in VALIDATION_PROMPTS])

                # Sample a random timestep for each image
                bsz = latents.shape[0] #// 2
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device, dtype=torch.long
                )

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(latents, noise, timesteps)

                # time ids
                def compute_time_ids(original_size, crops_coords_top_left):
                    target_size = (args.resolution, args.resolution)
                    add_time_ids = list(original_size + crops_coords_top_left + target_size)
                    add_time_ids = torch.tensor([add_time_ids])
                    add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
                    return add_time_ids

                add_time_ids = torch.cat(
                    [compute_time_ids(s, c) for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])]
                )

                # Get the text embedding for conditioning
                prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    [text_encoder_one, text_encoder_two], [batch["input_ids_one"], batch["input_ids_two"]]
                )
                prompt_embeds = prompt_embeds
                pooled_prompt_embeds = pooled_prompt_embeds

                model_pred = unet(
                    noisy_model_input.detach(),
                    timesteps,
                    prompt_embeds,
                ).sample

                
                

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Compute losses.
                model_losses = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                model_losses = model_losses.mean(dim=list(range(1, len(model_losses.shape))))

                # For logging
                raw_model_loss = model_losses.mean() 

                # Reference model predictions.
                if args.use_lora:
                    accelerator.unwrap_model(unet).disable_adapters()
                    ref_unet = unet
                else:
                    reference_model.store(unet.parameters())
                    reference_model.copy_to(unet.parameters())
                    ref_unet = unet
                if args.loss in ['sft']:
                    ref_loss = model_losses.detach()
                    raw_ref_loss = ref_loss.mean()
                else:
                    with torch.no_grad():
                        ref_unet.to(accelerator.device,dtype=weight_dtype)
                        ref_preds = ref_unet(
                            noisy_model_input.to(weight_dtype),
                            timesteps,
                            prompt_embeds,
                            added_cond_kwargs={"time_ids": add_time_ids, "text_embeds": pooled_prompt_embeds},
                        ).sample
                        ref_loss = F.mse_loss(ref_preds.float(), target.float(), reduction="none")
                        ref_loss = ref_loss.mean(dim=list(range(1, len(ref_loss.shape))))
                        raw_ref_loss = ref_loss.mean()
                    if args.use_lora:
                        del ref_unet
                    else:
                        reference_model.restore(unet.parameters())
                        del ref_unet

                # Re-enable adapters.
                if args.use_lora:
                    accelerator.unwrap_model(unet).enable_adapters()
                policy_KL_logps = - model_losses
                reference_KL_logps = -ref_loss
                g_term = policy_KL_logps - reference_KL_logps
                # Final loss.
                scale_term = args.beta_dpo
                labels = batch['labels']
                kl_gpu = g_term.mean().detach()
                kl = accelerator.reduce(kl_gpu,reduction='mean') 
                kl = kl.clamp(min=0).detach()
                g_term = g_term - kl 
                label_sgn = 2 * labels - 1;
                labels_binary = labels == 1
                label_scale_g = label_sgn * scale_term * g_term
                if args.halo == 'sigmoid':
                    h =  torch.sigmoid(label_scale_g)
                elif args.halo == 'loss_averse':
                    h = torch.nn.functional.logsigmoid(label_scale_g)
                elif args.halo == 'risk_seeking_1':
                    h =  -label_sgn * torch.nn.functional.logsigmoid(-scale_term * g_term)
                elif args.halo == 'risk_seeking_2':
                    h =   label_sgn  * torch.exp(scale_term * g_term/100)-1
                elif args.halo == 'risk_seeking_3':
                    h =  -torch.nn.functional.logsigmoid(-label_scale_g)
                else:
                    raise NotImplemented
                w_y = args.lambda_d_kto * labels_binary + args.lambda_u_kto * ~labels_binary
                l_kto = w_y * (1 - h)
                acc = ( label_scale_g > 0).sum().detach() / len(label_scale_g)
                n_pos = labels_binary.sum().item()
                n_neg = (len(labels_binary) - n_pos)
                with torch.no_grad():
                    diff_pos = g_term[labels_binary].sum() / (labels_binary.sum() + 1e-6)
                    diff_neg = g_term[~labels_binary].sum() / ((~labels_binary).sum() + 1e-6)
                    
                if args.loss == 'sft':
                    loss = (model_losses * labels).mean() 
                elif args.loss == 'cft':
                    loss = model_losses.mean()
                elif args.loss == 'kto':
                    loss = l_kto.mean()
                elif args.loss == 'kto_gp':
                    base_factor = (w_y*label_scale_g).mean(0,keepdim=True)
                    base_factor_all =  accelerator.reduce(kl_gpu,reduction='mean') 
                    world_size = accelerator.num_processes
                    inner_term = (base_factor_all - base_factor/world_size).detach() + base_factor/world_size
                    h = torch.sigmoid(inner_term)
                    l_kto = (1 - h)
                    loss = l_kto.mean()
                else:
                    raise NotImplemented
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if args.use_ema and (global_step+1)% 20 == 0:
                        ema_unet.step(unet.parameters())
                    if args.reference_ema and  (global_step+1)% 20 == 0:
                        reference_model.step(unet.parameters())
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                else:
                    print("Here")
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if 1:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    try:
                                        if os.path.exists(removing_checkpoint):
                                            shutil.rmtree(removing_checkpoint)
                                    except:
                                        pass

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.run_validation and global_step % args.validation_steps == 0 or global_step==1:
                        if args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())
                        if accelerator.is_main_process:
                            log_validation(
                                args, unet=unet, vae=vae, accelerator=accelerator, weight_dtype=weight_dtype,
                                epoch=epoch,global_step=global_step,
                                reference_model=reference_model, fixed_noise=fixed_noise,
                            )
                        if args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_unet.restore(unet.parameters())

            logs = {
                "loss": loss.detach().item(),
                "raw_model_loss": raw_model_loss.detach().item(),
                "ref_loss": raw_ref_loss.detach().item(),
                "likelyhood":g_term.detach().mean().item(),
                "kl":kl.detach().item(),
                "diff_pos":diff_pos.detach().item(),
                "diff_neg":diff_neg.detach().item(),
                "n_pos":n_pos,
                "n_neg":n_neg,
                "acc":acc.item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))

        # Final validation?
        if args.run_validation:
            log_validation(
                args,
                unet=None,
                vae=vae,
                accelerator=accelerator,
                weight_dtype=weight_dtype,
                epoch=epoch,
                global_step=global_step,
                is_final_validation=True,
                fixed_noise=fixed_noise,
            )

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
