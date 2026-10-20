# Dynamic Cross-Modal Prompt Generation for Multimodal Continual Instruction Tuning

<div align="center">
    <div><b>Tao Hu</b>&emsp;<b>Da-Wei Zhou</b></div>
    <div>School of Artificial Intelligence, Nanjing University</div>
    <div>State Key Laboratory for Novel Software Technology, Nanjing University</div>
</div>

<div align="center">
    <a href="https://arxiv.org/abs/2605.10765">
        <img src="https://img.shields.io/badge/Paper-arXiv-red" alt="arXiv">
    </a>
    &nbsp;
    <a href="https://github.com/LAMDA-CL/NeurIPS2026-Drape">
        <img src="https://img.shields.io/badge/Codebase-DRAPE-blue" alt="DRAPE">
    </a>
</div>

The code repository for "[Dynamic Cross-Modal Prompt Generation for Multimodal Continual Instruction Tuning](https://arxiv.org/abs/2605.10765)". If you use this work, please cite:

```bibtex
@inproceedings{hu2026dynamic,
  title     = {Dynamic Cross-Modal Prompt Generation for Multimodal Continual Instruction Tuning},
  author    = {Hu, Tao and Zhou, Da-Wei},
  booktitle = {NeurIPS},
  year      = {2026}
}
```

## Introduction

Multimodal Large Language Models (MLLMs) achieve strong performance through instruction tuning, yet real-world deployment often requires continual capability expansion across sequential tasks. In such scenarios, Multimodal Continual Instruction Tuning (MCIT) aims to acquire new capabilities while limiting catastrophic forgetting. Existing methods mainly follow a module-composition paradigm: they maintain task-level prompts or LoRA experts and dynamically route or aggregate a subset of them at inference. However, samples within the same task can still differ substantially in visual scenes, question intents, and reasoning demands. This motivates instance-level adaptation to individual query-image pairs rather than only selecting or combining task-level modules. To this end, we propose DRAPE (Dynamic Cross-Modal Prompt Generation), a prompt-learning framework that synthesizes continuous instance-specific soft prompts for MCIT. Instead of selecting prompts from a fixed pool, DRAPE derives prompt queries from the textual instruction and cross-attends to visual patch features, producing prompts conditioned on query-image pairs that are prepended to the frozen LLM. To mitigate forgetting during sequential updates, DRAPE applies null-space gradient projection to the shared projector and uses CLIP-based prototype routing for task-label-free generator selection at inference. Extensive experiments on MCIT benchmarks show that DRAPE achieves state-of-the-art performance among representative prompt-based and LoRA-based continual-learning baselines.

## Requirements

**Environment.** From the repository root, install the dependencies in a Linux/CUDA environment:

```bash
conda create -n drape python=3.10 -y
conda activate drape
pip install --upgrade pip
pip install -e .
pip install -e ".[train]" packaging
pip install flash-attn==2.3.3 --no-build-isolation
pip install nltk==3.9.1 pycocotools==2.0.8 pycocoevalcap==1.2 "openai>=1.0,<2.0"
```

Dependencies are defined in [pyproject.toml](pyproject.toml). Caption evaluation also requires a Java runtime for METEOR.

**Pre-trained weights.** Download [LLaVA-v1.5-7B](https://huggingface.co/liuhaotian/llava-v1.5-7b) and [CLIP ViT-L/14-336](https://huggingface.co/openai/clip-vit-large-patch14-336) into `data/LLaVa/` and `data/CLIP/`. Include `mm_projector.bin` in the LLaVA directory.

**Datasets.** This repository uses **CoIN** and **UCIT**:

- **CoIN:** follow the [image preparation guide](https://github.com/zackschen/CoIN#dataset) and download [instruction files](https://huggingface.co/datasets/Zacks-Chen/CoIN/tree/main/Instructions_Original). Place images in `data/datasets/` and instructions in `data/instructions/Instructions_Original/`.
- **UCIT:** follow the [HiDe-LLaVA preparation guide](https://github.com/Ghy0501/HiDe-LLaVA#ucit-benchmark) and download [instruction files](https://huggingface.co/datasets/HaiyangGuo/UCIT). Place images in `data/dataset_UCIT/` and instructions in `data/instructions/UCIT/`.

Preserve the image paths and benchmark annotations expected by the instruction files. Update paths in `scripts/drape/` to match your local setup.

## Running

From the repository root:

```bash
# CoIN
bash scripts/drape/Train_CoIN/train_all.sh
bash scripts/drape/Eval_CoIN/Eval_all.sh

# UCIT
bash scripts/drape/Train_UCIT/train_all.sh
bash scripts/drape/Eval_UCIT/Eval_all.sh
```

Scripts use GPUs `0,1,2,3` by default and save checkpoints to `checkpoints/llava_drape_{CoIN,UCIT}_TaskN`.

## Acknowledgement

This implementation builds on [LLaVA](https://github.com/haotian-liu/LLaVA). We thank the authors for their contributions to the open-source community.
