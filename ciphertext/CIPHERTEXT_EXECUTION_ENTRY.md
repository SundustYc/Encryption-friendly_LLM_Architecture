# Ciphertext 文件夹执行入口分析

## 概述
ciphertext 文件夹是全同态加密下的 LoRA 安全微调系统的核心实现，采用 C++17 + CUDA + HEaaN库开发。

## 编译和执行流程

### 1. 编译构建
```bash
cd ciphertext
mkdir build
cd build
cmake ..
make
```

编译后的可执行文件位于 `build/bin/` 目录。

### 2. 执行入口（可执行程序）

#### a) **keygen** - 密钥生成
**源文件**: `examples/keygen.cpp`

**功能**: 生成全同态加密的公钥和私钥

**执行方式**:
```bash
export HELLM_KEY_PATH=./keys
./bin/keygen
```

**关键代码**:
```cpp
int main() {
    std::string key_path{std::getenv("HELLM_KEY_PATH")};
    std::filesystem::create_directory(key_path);
    std::filesystem::create_directory(key_path + "/PK");
    
    HELLM::HEMMer hemmer{HELLM::HEMMer::genHEMMer()};
    hemmer.save(key_path);
}
```

---

#### b) **convert** - 权重转换
**源文件**: `examples/convert2.cpp`

**功能**: 将 PyTorch 训练好的明文模型权重转换为全同态加密格式

**执行方式**:
```bash
export HELLM_KEY_PATH=./keys
./bin/convert [-g]  # -g 参数用于生成模式
```

**关键参数**:
- 输入: `./data_2ly_mrpc/converted_weights_mrpc.pth` (PyTorch TorchScript格式)
- 输出: 加密后的权重文件
- 支持的任务: MRPC, RTE, COLA, STSB, SST2, QNLI

---

#### c) **eval** - 评估推理
**源文件**: `examples/bert-test.cpp`

**功能**: 在加密域中执行 BERT 推理（不涉及梯度计算）

**执行方式**:
```bash
mpirun -np 6 ./bin/eval
```

**关键配置**:
```cpp
static const std::string LORA_TYPE = "qkv";  // 应用 LoRA 的位置
static const std::string data_path = "./data_2ly_mrpc/";
const int prompt_len = 128;

// MRPC 有 51 个评估样本
for (int j = 0; j < 51; ++j) {
    // 推理循环
}
```

**关键特性**:
- 使用 MPI 多进程并行
- 支持多GPU加速
- 加密推理不泄露任何明文信息

---

#### d) **train** - 加密训练（LoRA安全微调）
**源文件**: `examples/backward-bert-multi.cpp`

**功能**: 在加密域中执行反向传播和权重更新，实现 LoRA 安全微调

**执行方式**:
```bash
mpirun -np 8 ./bin/train
```

**关键配置**:
```cpp
static const std::string LORA_TYPE = "qkv";
const std::string weight_pth = "./data_2ly_mrpc/";
const int num_gpu = 8;
const int batch_size = 16 / 8 = 2;

// 任务配置
auto task = "M";  // M=MRPC, R=RTE, C=COLA, S=STSB, T=SST2, Q=QNLI
const HELLM::u64 one_epo_step = 458;  // MRPC 每个 epoch 的步数
const int num_data = 3668;  // MRPC 总数据样本数
```

**核心流程**:
1. 初始化 LoRA 权重并加密
2. 对每个批次数据进行加密推理
3. 在加密域计算损失函数
4. 执行反向传播计算梯度（全程加密）
5. 使用 AdamW 优化器更新 LoRA 权重（加密方式）
6. Bootstrap 操作恢复噪声等级

---

#### e) **其他工具程序**

- **backward-time.cpp**: 性能基准测试
- **bert-matmul.cpp**: 矩阵乘法验证
- **bert-matmul-backward.cpp**: 反向传播矩阵乘法验证

---

## 核心类和模块

### 1. **HEMMer** (`include/HELLM/HEMMer.hpp`)
全同态加密管理器，封装所有加密操作

```cpp
// 单GPU模式
HELLM::HEMMer hemmer = HELLM::HEMMer::genHEMMer();

// 多GPU模式（MPI）
HELLM::HEMMer hemmer = HELLM::HEMMer::genHEMMerMultiGPU();
```

**主要方法**:
- `encrypt2()`: 加密明文张量
- `decrypt2()`: 解密密文张量
- `bootstrap()`: 恢复噪声等级
- `getEval()`: 获取同态评估器

### 2. **LoRA模块** (`src/LoRA.cpp`)
实现加密域 LoRA 微调

**核心功能**:
- `generateInitialLoraWeight()`: 初始化 LoRA 权重
- `zeroGrad()`: 梯度清零
- `AdamW()`: AdamW 优化器（加密版本）
- `optimizerStep_bert()`: 优化步骤
- `approxInverseSqrt_*()`: 近似求平方根倒数

### 3. **TransformerBlock** (`src/TransformerBlock.cpp`)
Transformer 层的加密实现

**关键方法**:
- 前向传播：加密注意力计算
- 反向传播：梯度计算

### 4. **TorchTransformerBlock** (`include/HELLM/TorchTransformerBlock.hpp`)
PyTorch 模型和加密系统的桥接

---

## 多GPU并行架构

### MPI 分布式设置
```
GPU 0: LoRA-Q 层 (rank=0)
GPU 1: LoRA-K 层 (rank=1)  
GPU 2: LoRA-V 层 (rank=2)
GPU 3: LoRA-Q 层 (rank=3)
GPU 4: LoRA-K 层 (rank=4)
GPU 5: LoRA-V 层 (rank=5)
GPU 6: 头层权重 (rank=6)
GPU 7: 头层权重 (rank=7)
```

**总体流程** (在 `backward-bert-multi.cpp`):
```
if (rank == 0):
    生成初始 LoRA 权重 (layer 0 和 1)
    初始化优化器状态

for each epoch:
    for each batch:
        Forward: 加密推理
        Compute Loss: 加密损失计算
        Backward: 加密反向传播
        Optimizer: AdamW 权重更新 (带 Bootstrap)
```

---

## 关键配置文件

### `include/HELLM/ModelArgs.hpp`
包含所有超参数和模型配置：
- `DIM = 768`: 隐藏层维度
- `N_HEAD = 12`: 注意力头数
- `HEAD_DIM = 64`: 每个头的维度
- `LOW_DIM = 8`: LoRA 秩
- `MAX_SEQ_LEN = 128`: 最大序列长度
- `LEARNING_RATE = 2e-4`: 学习率
- `BETA_1 = 0.9`, `BETA_2 = 0.999`: Adam 参数
- `WEIGHT_DECAY = 0.01`: 权重衰减

---

## 数据流

### 1. 初始化阶段
```
PyTorch 模型 (.pth)
    ↓
convert 程序
    ↓
加密权重文件 (.bin)
```

### 2. 训练阶段
```
明文输入数据
    ↓
HEMMer::encrypt2() [加密]
    ↓
TransformerBlock [加密推理]
    ↓
Loss 计算 [加密域]
    ↓
反向传播 [加密域]
    ↓
LoRA::AdamW() [加密参数更新]
    ↓
更新的加密权重 (.bin)
```

### 3. 推理阶段
```
明文输入
    ↓
加密输入
    ↓
TransformerBlock [加密推理]
    ↓
加密输出
    ↓
解密输出 [仅用于准确度评估]
```

---

## 执行命令汇总

### 完整工作流

1. **密钥生成** (一次性)
```bash
export HELLM_KEY_PATH=./keys
./bin/keygen
```

2. **权重转换** (一次性)
```bash
export HELLM_KEY_PATH=./keys
./bin/convert
```

3. **加密推理评估**
```bash
mpirun -np 6 ./bin/eval
```

4. **加密LoRA微调**
```bash
mpirun -np 8 ./bin/train
```

---

## 性能考虑

- **Bootstrap 操作**: 是性能瓶颈，用于恢复密文噪声等级
- **多GPU并行**: 通过 MPI 分散计算负载
- **精度管理**: 需要定期 Bootstrap 以维持可计算性
- **内存占用**: 加密数据量是明文的约 2-3 倍

---

## 参考文档

- `docs/CIPHERTEXT_DETAILED_FLOWCHART.md` - 详细执行流程
- `docs/CIPHERTEXT_CODE_REFERENCE.md` - 代码参考
- `docs/CIPHERTEXT_ANALYSIS.md` - 技术分析

