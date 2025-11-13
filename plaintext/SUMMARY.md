根据你提供的 plaintext 文件夹结构和项目背景（关于将同态加密应用于 LoRA 的论文开源代码），我对该文件夹的内容做如下详细说明：

### 1. 顶层脚本与配置
- **eval_128_padding.py / eval_lora_128_padding.py / eval.py / eval_GLUE.sh**  
  这些是模型评估相关的脚本，分别用于不同的评测场景（如 GLUE 基准、LoRA 相关评测等）。
- **pretrain.py**  
  预训练主脚本，负责模型的训练流程。
- **load_local_model.py**  
  加载本地模型的工具脚本，便于模型复现或微调。
- **saving_fine-tuning_data.py / upload_processed_dataset.py**  
  数据处理与上传相关脚本，支持微调和数据集管理。
- **sentense_test_eval.py**  
  句子级别的测试评估脚本。
- **requirements.txt / pyproject.toml / setup.cfg / MANIFEST.in**  
  Python 项目依赖与打包配置文件。

### 2. 文档与元数据
- **README.md / model_card.md / data_card.md / CITATION.cff**  
  项目说明、模型卡、数据卡和引用信息，便于用户了解项目用途和引用方式。
- **LICENSE.MD**  
  项目许可证。
- **index.html**  
  可能用于项目主页或文档展示。

### 3. cramming 子模块
- `cramming/` 目录下包含了项目的核心代码，结构如下：
  - **architectures/**  
    包含多种 Transformer 架构实现（如 Bert、Funnel、LoRA、Recurrent 等），以及相关组件（attention、embeddings、losses、lora_layers 等）。
  - **backend/**  
    后端相关代码，包括对 DeepSpeed 的集成、优化器实现（如 AdaHessian、Sophia、Shampoo 等）、批处理和调度器等。
  - **config/**  
    配置文件（yaml），涵盖模型结构、预训练、评测等，支持多种架构和数据配置。
  - **data/**  
    数据处理相关脚本，包括去重、预处理、分词等。
  - **scripts/**  
    各类实验脚本，支持架构消融、数据消融、预处理、复现等实验流程。

### 4. 功能与用途总结
- 该文件夹主要负责：
  - 模型的定义与实现（支持多种 Transformer 变体和 LoRA 相关结构）
  - 数据处理与预处理
  - 训练、评测、微调流程的自动化
  - 优化器和后端集成（支持高效训练和分布式训练）
  - 项目配置、文档和元数据管理

### 5. 与同态加密和 LoRA 的关系
- 虽然 plaintext 文件夹本身未直接包含同态加密代码，但它实现了 LoRA 相关的 Transformer 架构和训练流程，为同态加密的集成和实验提供了基础框架。
- 项目整体架构支持在 LoRA 等模型上进行同态加密相关实验，便于论文复现和进一步研究。


结合 README.md 的内容和 `scripts` 目录下的脚本，整个项目的代码分为三大部分，每部分的功能和流程如下：

## 1. 预训练（Pre-training）

**主要文件/脚本：**
- `pretrain.py`
- 配置文件：`cramming/config/arch/v1/crammed-bert.yaml`、`cramming/config/impl/_default.yaml`、`cramming/config/cfg_pretrain.yaml`
- 部分辅助脚本：`scripts/preprocessing.sh`、`scripts/reproducing_bert.sh`

**流程说明：**
1. 数据预处理（可选）：  
   使用 `scripts/preprocessing.sh` 对原始数据进行清洗、格式化，生成可用于预训练的数据集。
2. 预训练启动：  
   运行 `python pretrain.py name={your_name} arch=crammed-bert train=bert-o4 data=pile-readymade`，在明文环境下训练 BERT 模型。  
   训练参数（如层数、步数、注意力头数等）可在相关 yaml 配置文件中调整。
3. 训练结果保存：  
   训练完成后，模型权重（如 `model.safetensors`）和日志会保存在 `outputs/{your_name}` 目录下。

---

## 2. 微调数据保存（Saving fine-tuning dataset）

**主要文件/脚本：**
- `saving_fine-tuning_data.py`
- `upload_processed_dataset.py`
- `convert.py`（位于 ciphertext 文件夹）
- 配置文件：`cramming/config/eval/GLUE_sane`

**流程说明：**
1. 权重准备：  
   将预训练得到的 `model.safetensors` 文件复制到 `pre-trained_weights/{your_name}` 目录。
2. 保存微调训练数据：  
   运行 `python saving_fine-tuning_data.py eval.user_name={your_name} eval.save_train_data=True`，将 GLUE 等微调数据保存为适合同态加密处理的格式。
3. 数据格式转换：  
   进入 ciphertext 文件夹，运行 `convert.py`，将保存的数据转换为同态加密友好的格式。
4. 保存微调评估数据：  
   运行 `python saving_fine-tuning_data.py eval.user_name={your_name} eval.save_train_data=False`，保存评估数据，并再次用 `convert.py` 转换格式。
5. 数据集选择：  
   可在 `cramming/config/eval/GLUE_sane` 的 `defaults/tasks` 中选择具体任务（如 cola, mrpc, qnli 等）。

---

## 3. 微调（Fine-tuning）

### 3-1. 明文微调（Plaintext）

**主要文件/脚本：**
- `eval_128_padding.py`（全参数微调）
- `eval_lora_128_padding.py`（LoRA 微调）
- 配置文件：`cramming/config/eval/GLUE_sane`

**流程说明：**
1. 运行全参数微调：  
   `python eval_128_padding.py eval=GLUE_sane name={your_name} eval.checkpoint=latest ...`
2. 运行 LoRA 微调：  
   `python eval_lora_128_padding.py eval=GLUE_sane name={your_name} eval.checkpoint=latest ...`
3. LoRA 参数调整：  
   可在配置文件中设置 `lora_rank` 和 `lora_alpha`，控制 LoRA 的低秩分解参数。

### 3-2. 密文微调（Ciphertext）

**主要文件/脚本：**
- 进入 ciphertext 文件夹，参考其 README.md 和相关脚本（如 `convert.py`）

**流程说明：**
1. 按照 ciphertext 文件夹的说明，使用同态加密相关代码进行微调实验。
2. 该部分实现了在加密环境下的模型微调，核心在于数据格式转换和加密计算。

---

## 4. scripts 目录作用

`scripts` 目录下的 shell 脚本主要用于自动化实验流程，包括：
- 架构消融实验（如 `architecture_ablations_c5_o3.sh`）
- 数据消融实验（如 `data_ablations_a4000.sh`、`data_ablations_a6000.sh`）
- 评测基线（如 `eval_baselines.sh`）
- 复现与预处理（如 `reproducing_bert.sh`、`preprocessing.sh`）
- 扩展实验（如 `scaling_law_cb_o4_a4000.sh` 等）

这些脚本通常会批量调用 Python 主程序，设置不同参数，便于大规模实验和论文复现。

---

## 总结

- **预训练阶段**：在明文下训练 BERT，生成权重。
- **微调数据保存阶段**：将微调数据转换为同态加密友好格式，便于后续加密微调。
- **微调阶段**：支持明文和密文两种环境，明文下可直接微调，密文下需数据转换和加密计算。
- **scripts 目录**：批量自动化实验，便于消融、复现和扩展。

如需某个脚本或流程的代码细节解读，可进一步指定文件名。