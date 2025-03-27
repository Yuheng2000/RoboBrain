<div align="center">
<img src="./assets/logo.jpg" width="400"/>
</div>

# [CVPR 2025] RoboBrain: A Unified Brain Model for Robotic Manipulation from Abstract to Concrete.



<p align="center">
        </a>&nbsp&nbsp⭐️ <a href="https://superrobobrain.github.io/">Project</a></a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://superrobobrain.github.io/">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://superrobobrain.github.io/">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="http://arxiv.org/abs/2502.21257">Paper</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://superrobobrain.github.io/">WeChat</a>
</p>


Recent advancements in Multimodal Large Language Models (MLLMs) have shown remarkable capabilities across various multimodal contexts. However, their application in robotic scenarios, particularly for long-horizon manipulation tasks, reveals significant limitations. These limitations arise from the current MLLMs lacking three essential robotic brain capabilities: **(1) Planning Capability**, which involves decomposing complex manipulation instructions into manageable sub-tasks; **(2) Affordance Perception**, the ability to recognize and interpret the affordances of interactive objects; and **(3) Trajectory Prediction**, the foresight to anticipate the complete manipulation trajectory necessary for successful execution. To enhance the robotic brain's core capabilities from abstract to concrete, we introduce ShareRobot, a high-quality heterogeneous dataset that labels multi-dimensional information such as task planning, object affordance, and end-effector trajectory. ShareRobot's diversity and accuracy have been meticulously refined by three human annotators. Building on this dataset, we developed RoboBrain, an MLLM-based model that combines robotic and general multi-modal data, utilizes a multi-stage training strategy, and incorporates long videos and high-resolution images to improve its robotic manipulation capabilities. Extensive experiments demonstrate that RoboBrain achieves state-of-the-art performance across various robotic tasks, highlighting its potential to advance robotic brain capabilities.

<div align="center">
<img src="./assets/overview.png" />
</div>

## 🚀 Features
This repository supports:
- **`Data Preparation`**: Please refer to [Dataset Preparation](#Dataset) for how to prepare the dataset.
- **`Training for RoboBrain`**: Please refer to [Training Section](#Training) for the usage of training scripts.
- **`Evaluation for RoboBrain`**: Please refer to [Evaluation Section](#Evaluation) for how to prepare the benchmarks.
- **`Support VLLM Inference`**: Please see [Inference  Section](#Inference), now we support inference with [VLLM](https://github.com/vllm-project/vllm).
- **`ShareRobot Generation`**: Please refer to [ShareRobot](https://github.com/FlagOpen/ShareRobot) for details.


## 🗞️ News

- **`2025-03-26`**: 🔥 We have released the [RoboBrain](https://superrobobrain.github.io/) repository.

- **`2025-02-27`**: 🌍 Our [RoboBrain](https://superrobobrain.github.io/) was accepted to CVPR2025.


## 🤖 Models


- **[`Base Planning Model`](https://superrobobrain.github.io/)**: The model was trained on general datasets in Stages 1–2 and on the Robotic Planning dataset in Stage 3, which is designed for Planning prediction.
- **[`A-LoRA for Affordance`](https://superrobobrain.github.io/)**: Based on the Base Planning Model, Stage 4 involves LoRA-based training with our Affordance dataset to predict affordance.
- **[`T-LoRA for Trajectory`](https://superrobobrain.github.io/)**: Based on the Base Planning Model, Stage 4 involves LoRA-based training with our Trajectory dataset to predict trajectory.

<div align="center">
<img src="./assets/training.png" />
</div>

| Models | Checkpoint | Description | 
|----------|----------------|----------------|
| Base Planning Model   | [Planning Checkpoint](https://superrobobrain.github.io/)   | Used for Planning prediction in our paper | 
| A-LoRA for Affordance | [Affordance Checkpoint](https://superrobobrain.github.io/) | Used for Affordance prediction in our paper | 
| T-LoRA for Trajectory | [Trajectory Checkpoint](https://superrobobrain.github.io/) | Used for  Trajectory prediction in our paper | 


## 🛠️ Setup

```bash
conda create -n robobrain python=3.10
conda activate robobrain
pip install -r requirements.txt
```

## 🤖 Training

### 1. Data Preparation

```bash
datasets:
    - yaml_path: /path/to/stage_1.yaml
        - json_path: xxx.json
        - json_path: xxx.json

    - yaml_path: /path/to/stage_1_5.yaml
        - json_path: xxx.json
        - json_path: xxx.json

    - yaml_path: /path/to/stage_2_si.yaml
        - json_path: xxx.json
        - json_path: xxx.json

    - yaml_path: /path/to/stage_2_ov.yaml
        - json_path: xxx.json
        - json_path: xxx.json

    - yaml_path: /path/to/stage_3_planning.yaml
        - json_path: xxx.json
        - json_path: xxx.json

    - yaml_path: /path/to/stage_4_affordance.yaml
        - json_path: xxx.json
        - json_path: xxx.json

    - yaml_path: /path/to/stage_4_trajectory.yaml
        - json_path: xxx.json
        - json_path: xxx.json
```

### 2. Training 
```bash
# Training on Stage 1:
bash scripts/train/stage_1_0_pretrain.sh

# Training on Stage 1.5:
bash scripts/train/stage_1_5_direct_finetune.sh

# Training on Stage 2_si:
bash scripts/train/stage_2_0_resume_finetune_si.sh

# Training on Stage 2_ov:
bash scripts/train/stage_2_0_resume_finetune_ov.sh

# Training on Stage 3_plan:
bash scripts/train/stage_3_0_resume_finetune_robo.sh

# Training on Stage 4_aff:
bash scripts/train/stage_4_0_resume_finetune_lora_a.sh

# Training on Stage 4_traj:
bash scripts/train/stage_4_0_resume_finetune_lora_t.sh
```

## 🤖 Evaluation

### 1. Data Preparation

```bash

```

### 2. Evaluation for Robotic Benchmarks
```bash

```

### 3. Evaluation for General Benchmarks
```bash

```

## 🤖 Inference

### Option 1: HF inference
```bash

```

### Option 2: VLLM inference
```bash

```


## 📑 Citation
If you find this project useful, welcome to cite us.
```bib
@article{ji2025robobrain,
  title={RoboBrain: A Unified Brain Model for Robotic Manipulation from Abstract to Concrete},
  author={Ji, Yuheng and Tan, Huajie and Shi, Jiayu and Hao, Xiaoshuai and Zhang, Yuan and Zhang, Hengyuan and Wang, Pengwei and Zhao, Mengdi and Mu, Yao and An, Pengju and others},
  journal={arXiv preprint arXiv:2502.21257},
  year={2025}
}
```
