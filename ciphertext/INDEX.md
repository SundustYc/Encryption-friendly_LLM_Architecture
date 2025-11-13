# Ciphertext 文件夹完整分析索引

本索引整理了对 ciphertext 全同态加密 LoRA 安全微调系统的完整分析文档。

---

## 📚 文档列表

### 1. **CIPHERTEXT_EXECUTION_ENTRY.md** ⭐ (入门必读)
**大小**: 6.4KB | **难度**: ⭐⭐ | **用途**: 快速了解执行入口

**内容**:
- 4个主要可执行程序的详细说明
- keygen / convert / eval / train 的完整工作流
- 多GPU并行架构 (8GPU MPI)
- 核心类和模块介绍
- 完整执行命令汇总

**快速导航**:
- 密钥生成: `keygen.cpp`
- 权重转换: `convert2.cpp`
- 推理评估: `bert-test.cpp` + `-np 6`
- 加密训练: `backward-bert-multi.cpp` + `-np 8`

---

### 2. **QUICK_START.md** ⭐ (速查表)
**大小**: 5.8KB | **难度**: ⭐ | **用途**: 快速上手

**内容**:
- 执行入口速查表 (4行表格)
- 分步执行顺序
- 编译命令
- 核心数据结构 (加密张量、LoRA权重存储)
- 关键配置参数 (ModelArgs.hpp)
- 多GPU并行分配表
- 支持的任务列表 (R/C/M/S/T/Q)
- 常见问题排查
- 文件夹组织结构

**适用场景**: "我要快速编译和运行"

---

### 3. **EXECUTION_FLOW.txt** �� (流程可视化)
**大小**: 17KB | **难度**: ⭐⭐⭐ | **用途**: 理解完整执行流程

**内容**:
- 3个阶段的详细流程图:
  - 阶段1: 初始化 (密钥生成 → 权重转换)
  - 阶段2: 评估推理 (加密推理 6个GPU)
  - 阶段3: 加密训练 (完整LoRA微调流程)
- 每个训练步骤的6个子步骤详解
- GPU内存分布 (8GPU × 16GB)
- 性能基准数据 (秒/批)
- 计算复杂度分析
- 精度管理 (Level概念)

**主要流程**:
```
Forward Pass (30-60s)
    ↓
Loss 计算 (5-10s)
    ↓
反向传播 (50-100s)
    ↓
AdamW 优化 (20-40s)
    ↓
Bootstrap (100-200s) ← 性能瓶颈
    ↓
梯度清零
```

---

### 4. **CODE_STRUCTURE.md** 🔍 (代码详解)
**大小**: 21KB | **难度**: ⭐⭐⭐ | **用途**: 深入代码实现

**内容**:
- 完整文件夹组织树
- 4个核心类的详细说明:
  1. **HEMMer** - 加密管理器
  2. **LoRA** - 微调模块
  3. **TransformerBlock** - Transformer实现
  4. **主程序入口** - 4个可执行程序源码
- 关键函数的伪代码实现
- ModelArgs.hpp 配置参数详解
- 数据流追踪 (2个例子)
- 代码阅读建议路径 (初/中/高三个阶段)
- 关键代码片段速查表

**核心代码亮点**:
- `HEMMer::encrypt2()` - 加密操作
- `LoRA::AdamW()` - 加密优化器 (~150行)
- `LoRA::approxInverseSqrt_*()` - 多项式近似
- `backward-bert-multi.cpp::main()` - 训练循环

---

## 🎯 快速导航

### "我要快速了解执行入口"
→ 阅读 **CIPHERTEXT_EXECUTION_ENTRY.md**

### "我要立即编译和运行"
→ 阅读 **QUICK_START.md** 中的编译和执行顺序部分

### "我要理解完整的执行流程"
→ 阅读 **EXECUTION_FLOW.txt**，特别是"阶段3"部分

### "我要研究代码实现细节"
→ 阅读 **CODE_STRUCTURE.md**，特别是关键函数部分

### "我要追踪特定功能"
→ 使用 CODE_STRUCTURE.md 中的"关键代码片段速查表"

---

## 🔄 学习路径建议

### 路径1: 快速上手 (30分钟)
1. QUICK_START.md - 编译和执行
2. CIPHERTEXT_EXECUTION_ENTRY.md - 了解4个程序
3. 运行 keygen → convert → eval

### 路径2: 理解原理 (2小时)
1. EXECUTION_FLOW.txt - 阶段1,2,3 流程
2. CIPHERTEXT_EXECUTION_ENTRY.md - 核心类介绍
3. CODE_STRUCTURE.md - 关键函数伪代码

### 路径3: 深入研究 (1天)
1. 上述路径2的全部内容
2. CODE_STRUCTURE.md - 所有核心类详解
3. CODE_STRUCTURE.md - 数据流追踪例子
4. 阅读源代码 (LoRA.cpp, TransformerBlock.cpp)

### 路径4: 扩展开发 (2天+)
1. 上述路径3的全部内容
2. 研究 ModelArgs.hpp 的超参数
3. 修改参数进行实验
4. 理解性能瓶颈 (Bootstrap操作)
5. 优化策略研究

---

## 📊 核心信息速查

### 4个执行程序
| 名称 | 源文件 | 功能 | 耗时 | GPU数 |
|------|--------|------|------|-------|
| keygen | examples/keygen.cpp | 密钥生成 | 1分钟 | 1 |
| convert | examples/convert2.cpp | 权重转换 | 10-30分钟 | 1 |
| eval | examples/bert-test.cpp | 推理评估 | ~10分钟 | 6 |
| train | examples/backward-bert-multi.cpp | 加密训练 | ~8小时/epoch | 8 |

### 支持的任务 (GLUE)
| 代码 | 任务 | 数据量 | Steps/epoch |
|------|------|-------|------------|
| R | RTE | 2490 | 39 |
| C | CoLA | 8551 | 167 |
| M | MRPC | 3668 | 458 |
| S | STS-B | 5749 | 90 |
| T | SST-2 | 67349 | 1048 |
| Q | QNLI | 104743 | 1636 |

### LoRA权重类型
- **q** - Query 投影的 LoRA
- **k** - Key 投影的 LoRA
- **v** - Value 投影的 LoRA
- **head** - 分类头的 LoRA

### 主要性能指标
| 操作 | 耗时 |
|------|------|
| 加密输入 (2×128) | 1-2秒 |
| 前向传播 (1层) | 30-60秒 |
| 反向传播 (1层) | 50-100秒 |
| Bootstrap (1次) | 100-200秒 ⚠️ 瓶颈 |
| 单批总耗时 | 200-500秒 |

---

## 🔐 全同态加密核心概念

### CKKS 方案特点
- **加法**: 开销小 (直接在多项式上)
- **乘法**: 开销大 (多项式乘法)
- **乘法深度**: 可进行的乘法操作数限制
- **噪声**: 计算中积累，最终影响解密精度

### Bootstrap 操作
- **用途**: 恢复乘法深度，允许继续计算
- **代价**: 非常耗时 (~100-200秒)
- **性能瓶颈**: 主要影响训练速度
- **策略**: 需要平衡精度和性能

### 多项式近似
- **ReLU**: 多项式近似
- **Softmax**: Chebyshev 多项式
- **倒数平方根**: Chebyshev 多项式 (AdamW中使用)

---

## 💡 关键技术创新

### 1. 加密域 LoRA 微调
- 仅微调低秩矩阵 (8维)
- 大大降低计算复杂度

### 2. 任务特定的学习率表
- 预计算各任务最优学习率
- 支持 6 个 GLUE 任务

### 3. 多GPU并行 (MPI)
- 8个GPU分别处理不同参数
- 使用 MPI AllReduce 聚合梯度

### 4. 加密域 Adam 优化器
- 一阶矩 (m) 更新
- 二阶矩 (v) 更新
- Chebyshev 多项式近似平方根倒数

---

## 📝 文件夹内容速查

### /ciphertext/examples/ (4个入口程序)
- **keygen.cpp** - 20行，密钥生成主程序
- **convert2.cpp** - 100+行，权重转换
- **bert-test.cpp** - 200+行，推理评估
- **backward-bert-multi.cpp** - 400+行，训练程序 ⭐ 最复杂

### /ciphertext/src/ (核心实现)
- **LoRA.cpp** - ~700行，LoRA 微调实现 ⭐⭐⭐
- **HEMMer.cpp** - ~500行，加密操作
- **TransformerBlock.cpp** - Transformer 层实现
- **其他**: MatrixUtils.cpp, Loss.cpp, 激活函数等

### /ciphertext/include/HELLM/ (头文件)
- **LoRA.hpp** - LoRA 类接口
- **HEMMer.hpp** - 加密管理器接口 ⭐
- **ModelArgs.hpp** - 2.2MB，所有超参数配置 ⭐
- **TransformerBlock.hpp** - Transformer 接口
- **其他**: 矩阵工具、多项式系数等

---

## ⚠️ 注意事项

### 内存占用
- 加密数据量 ≈ 明文 × 2-3倍
- 8个16GB GPU 接近满载
- 不能简单增加批次大小

### 计算复杂度
- Bootstrap 是主要瓶颈
- 需要定期 Bootstrap 恢复计算能力
- 单个批次 200-500 秒 (取决于 Bootstrap)

### 精度问题
- 需要 Chebyshev 多项式进行函数近似
- 任务特定的学习率表 (预先优化)
- 噪声积累可能影响收敛

### MPI 配置
- keygen: 1 GPU
- convert: 1 GPU
- eval: 6 GPU (必须)
- train: 8 GPU (必须)

---

## 🚀 快速命令

```bash
# 编译
cd ciphertext && mkdir build && cd build && cmake .. && make -j8

# 运行 (按顺序)
export HELLM_KEY_PATH=./keys

./bin/keygen
./bin/convert
mpirun -np 6 ./bin/eval
mpirun -np 8 ./bin/train
```

---

## 📖 进一步学习

### HEaaN 库
- 全同态加密的底层实现
- CKKS 方案
- 多项式运算

### BERT 微调
- Attention 机制
- LayerNorm, ReLU, Softmax
- 反向传播算法

### LoRA (Low-Rank Adaptation)
- 参数高效微调
- 低秩矩阵分解
- 用于大模型快速适配

### 分布式训练
- MPI 通信
- AllReduce 操作
- 多GPU同步

---

## 📞 快速索引

| 查找 | 文档 | 位置 |
|------|------|------|
| 执行命令 | QUICK_START.md | 执行顺序 |
| 程序说明 | CIPHERTEXT_EXECUTION_ENTRY.md | 执行入口 |
| 流程图 | EXECUTION_FLOW.txt | 阶段1,2,3 |
| 源代码 | CODE_STRUCTURE.md | 核心类详解 |
| 参数配置 | CODE_STRUCTURE.md | ModelArgs |
| 编译指导 | QUICK_START.md | 编译章节 |
| 问题排查 | QUICK_START.md | 常见问题 |

---

## �� 版本信息

**创建时间**: 2024年11月9日
**分析范围**: ciphertext 文件夹所有源代码
**覆盖程度**: 完整分析 (初/中/高三个学习阶段)

---

**文档维护**: 自动生成的分析文档
**下一步**: 选择上述文档开始学习！
