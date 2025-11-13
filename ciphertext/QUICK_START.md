# Ciphertext 快速启动指南

## 🎯 执行入口速查表

| 程序 | 源文件 | 功能 | 执行命令 |
|------|--------|------|---------|
| **keygen** | `examples/keygen.cpp` | 生成HE密钥对 | `./bin/keygen` |
| **convert** | `examples/convert2.cpp` | PyTorch→加密权重 | `./bin/convert` |
| **eval** | `examples/bert-test.cpp` | 加密推理 | `mpirun -np 6 ./bin/eval` |
| **train** | `examples/backward-bert-multi.cpp` | 加密LoRA微调 | `mpirun -np 8 ./bin/train` |

---

## 📋 执行顺序

### Step 1: 密钥生成
```bash
cd /root/autodl-tmp/Encryption-friendly_LLM_Architecture-master/ciphertext
cd build

export HELLM_KEY_PATH=./keys
./bin/keygen
```

### Step 2: 权重转换
```bash
export HELLM_KEY_PATH=./keys
./bin/convert
```

### Step 3: 推理评估
```bash
mpirun -np 6 ./bin/eval
```

### Step 4: 加密训练
```bash
mpirun -np 8 ./bin/train
```

---

## 🔧 编译

```bash
cd ciphertext
mkdir -p build
cd build
cmake ..
make -j8
```

编译后的二进制文件在 `build/bin/` 目录

---

## 📊 核心数据结构

### 加密张量 (CtxtTensor)
```cpp
// 在加密域中的张量
CtxtTensor encrypted_tensor = hemmer->encrypt2(plaintext_a, plaintext_b);

// 同态操作（直接在加密域进行）
hemmer->getEval().add(ctxt1.get(), ctxt2.get(), result.get());
hemmer->getEval().mult(ctxt.get(), scalar, result.get());
hemmer->getEval().square(ctxt.get(), result.get());
```

### LoRA权重存储
```
LoRA 权重 (加密):
  ├─ lora_wa_q / lora_wa_k / lora_wa_v  [下投影]
  ├─ lora_wb_q / lora_wb_k / lora_wb_v  [上投影]
  ├─ momentum_ma, momentum_mb           [Adam一阶矩]
  ├─ momentum_va, momentum_vb           [Adam二阶矩]
  └─ 存储在 weight_path 中的 .bin 文件
```

---

## 🎛️ 关键配置参数

在 `include/HELLM/ModelArgs.hpp` 中修改：

```cpp
// 模型结构
static const u64 DIM = 768;           // 隐藏层维度
static const u64 N_HEAD = 12;         // 注意力头数  
static const u64 HEAD_DIM = 64;       // 每头维度
static const u64 LOW_DIM = 8;         // LoRA秩

// 数据
static const u64 MAX_SEQ_LEN = 128;   // 序列长度

// 优化器
static const double LEARNING_RATE = 2e-4;
static const double BETA_1 = 0.9;     // Adam参数
static const double BETA_2 = 0.999;
static const double WEIGHT_DECAY = 0.01;
static const double OPTI_EPS = 1e-8;

// 任务特定（在backward-bert-multi.cpp中修改）
const char* task = "M";  // M/R/C/S/T/Q
const int num_gpu = 8;   // GPU数量
const int batch_size = 2; // 批次大小
```

---

## 🚀 多GPU并行

系统自动分配：
- **GPU 0,3**: LoRA-Q 层
- **GPU 1,4**: LoRA-K 层  
- **GPU 2,5**: LoRA-V 层
- **GPU 6,7**: 分类头层

使用 MPI 自动通信和同步。

---

## 📊 支持的任务

| 任务代码 | 任务名称 | 数据量 | 步数/epoch |
|---------|--------|-------|----------|
| R | RTE | 2490 | 39 |
| C | CoLA | 8551 | 167 |
| M | MRPC | 3668 | 458 |
| S | STS-B | 5749 | 90 |
| T | SST-2 | 67349 | 1048 |
| Q | QNLI | 104743 | 1636 |

---

## 🔍 常见问题排查

### 问题1: 找不到密钥文件
```
错误: Cannot load context from key_path
解决: export HELLM_KEY_PATH=./keys && ./bin/keygen
```

### 问题2: MPI进程数不匹配
```
错误: MPI rank mismatch
解决: eval 用 -np 6, train 用 -np 8
```

### 问题3: CUDA内存不足
```
解决: 在 ModelArgs.hpp 中减小 LOW_DIM 或 MAX_SEQ_LEN
```

### 问题4: Bootstrap 失败
```
错误: Level too low for bootstrap
解决: 在 LoRA.cpp 中检查 bootstrapIfNecessary() 的阈值设置
```

---

## 📁 文件组织

```
ciphertext/
├── examples/           # 5个主要入口程序
│   ├── keygen.cpp      # 密钥生成 ⭐
│   ├── convert2.cpp    # 权重转换 ⭐
│   ├── bert-test.cpp   # 推理评估 ⭐
│   └── backward-bert-multi.cpp  # 训练 ⭐
├── src/                # 核心实现
│   ├── LoRA.cpp        # LoRA微调 (350+行)
│   ├── TransformerBlock.cpp
│   └── HEMMer.cpp
├── include/HELLM/      # 头文件
│   ├── LoRA.hpp
│   ├── HEMMer.hpp
│   └── ModelArgs.hpp   # 模型配置
└── build/              # 编译输出
    └── bin/            # 可执行文件

```

---

## 🧮 关键数学操作

### 在加密域中实现的 AdamW
```cpp
// 在 LoRA.cpp::AdamW() 中
θ = θ * (1 - lr * weight_decay)
m = β₁*m + (1-β₁)*∇
v = β₂*v + (1-β₂)*∇²
m̂ = m / (1 - β₁^t)  
v̂ = v / (1 - β₂^t)
θ = θ - lr * m̂ / (√v̂ + ε)
```

### Bootstrap操作
```cpp
// 恢复加密数据的噪声等级，允许继续计算
hemmer->bootstrap(ciphertext);
// 用于处理数据相关的乘法深度问题
```

### 近似求平方根倒数
```cpp
// 使用 Chebyshev 多项式近似 1/√x
approxInverseSqrt_COLA(eval, btp, input, output, num_iter);
// 是AdamW中最复杂的操作
```

---

## 📖 学习路径

1. **理解加密**: 阅读 HEaaN 库文档
2. **查看入口**: 从 `examples/keygen.cpp` 开始
3. **追踪数据流**: 在 `backward-bert-multi.cpp` 中跟踪
4. **深入算法**: 研究 `src/LoRA.cpp` 的 AdamW 实现
5. **优化性能**: 分析 Bootstrap 和并行策略

---

## 🎓 输出文件位置

执行完成后生成：
- 密钥: `./keys/`
- 加密权重: `./data_2ly_mrpc/` (*.bin 文件)
- 梯度: `./he_path/` (MPI rank 特定)
- 日志: 终端输出 (准确度、损失等)

---

## 📞 关键源代码位置

| 功能 | 文件 | 行数 |
|------|------|------|
| 加密操作 | `src/HEMMer.cpp` | ~500 |
| LoRA初始化 | `src/LoRA.cpp:generateInitialLoraWeight()` | 25-80 |
| LoRA梯度 | `src/LoRA.cpp:zeroGrad()` | 120-145 |
| AdamW优化 | `src/LoRA.cpp:AdamW()` | 250-400 |
| 推理循环 | `examples/bert-test.cpp:main()` | 60-100 |
| 训练循环 | `examples/backward-bert-multi.cpp:main()` | 100-300 |

