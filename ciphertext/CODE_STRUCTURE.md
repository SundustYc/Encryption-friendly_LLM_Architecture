# Ciphertext 代码结构详解

## 📁 文件夹组织

```
ciphertext/
│
├── 📄 CMakeLists.txt                   # 编译配置
│
├── �� build/                          # 编译输出目录
│   ├── bin/                           # 可执行文件
│   │   ├── keygen                     # 密钥生成
│   │   ├── convert                    # 权重转换
│   │   ├── eval                       # 推理评估
│   │   └── train                      # LoRA训练
│   └── CMakeFiles/                    # CMake临时文件
│
├── 📁 cmake/                          # CMake脚本和配置
│   ├── BuildType.cmake                # 构建类型设置
│   ├── CCache.cmake                   # 编译缓存
│   ├── CompilerWarnings.cmake         # 警告设置
│   └── Config.cmake.in                # 配置模板
│
├── 📁 include/HELLM/                  # 头文件 (核心)
│   ├── 🔐 HEMMer.hpp                 # ★ HE加密管理器 (19KB)
│   ├── 🔐 LoRA.hpp                   # ★ LoRA模块接口 (5KB)
│   ├── 📊 ModelArgs.hpp               # ★ 模型配置 (2.2MB)
│   ├── 🧮 TransformerBlock.hpp        # Transformer实现
│   ├── 🧮 TorchTransformerBlock.hpp   # PyTorch桥接
│   ├── 📐 MatrixUtils.hpp             # 矩阵操作工具
│   ├── 📊 LoRARemezCoeff.hpp          # LoRA多项式系数
│   ├── SoftmaxRemezCoeff.hpp          # Softmax多项式系数
│   ├── ReLU.hpp                       # ReLU激活函数
│   ├── Softmax.hpp                    # Softmax激活函数
│   ├── LayerNorm.hpp                  # 层归一化
│   ├── Loss.hpp                       # 损失函数
│   └── utils/                         # 工具函数
│       └── check_macros.hpp           # 检查宏定义
│
├── 📁 src/                            # 核心实现代码
│   ├── 🔐 HEMMer.cpp                 # ★ 加密操作实现 (~500行)
│   ├── 🔐 LoRA.cpp                   # ★ LoRA微调 (~700行)
│   ├── 🧮 TransformerBlock.cpp       # Transformer前向/反向
│   ├── TorchTransformerBlock.cpp     # PyTorch集成
│   ├── MatrixUtils.cpp               # 矩阵工具函数
│   ├── ReLU.cpp                      # ReLU实现
│   ├── Softmax.cpp                   # Softmax实现
│   ├── LayerNorm.cpp                 # LayerNorm实现
│   ├── Loss.cpp                      # 损失函数实现
│   ├── Exp.cpp                       # 指数函数
│   └── Tanh.cpp                      # Tanh函数
│
├── 📁 examples/                      # 可执行程序入口 ★★★
│   ├── CMakeLists.txt                # 编译配置
│   ├── ⭐ keygen.cpp                # 密钥生成入口 (20行)
│   ├── ⭐ convert2.cpp              # 权重转换入口 (100+行)
│   ├── ⭐ bert-test.cpp             # 推理评估入口 (200+行)
│   ├── ⭐ backward-bert-multi.cpp   # LoRA训练入口 (400+行)
│   ├── backward-time.cpp             # 性能基准测试
│   ├── bert-matmul.cpp               # 矩阵乘法验证
│   └── bert-matmul-backward.cpp      # 反向传播验证
│
├── 📁 test/                          # 测试代码
│   ├── CMakeLists.txt
│   ├── HELLMTestBase.hpp             # 测试基类
│   └── HEMMerTest_tmp.cpp            # 单元测试
│
├── 📁 external/                      # 外部依赖库
│   ├── include/                      # HEaaN库头文件
│   └── lib/                          # HEaaN库二进制
│
├── 📁 conda/                         # Conda环境配置
│   └── hellm-bert-env.yml            # 依赖环境定义
│
└── 📁 docs/                          # 文档
    ├── ANALYSIS_README.md
    ├── CIPHERTEXT_ANALYSIS.md
    └── CIPHERTEXT_CODE_REFERENCE.md

```

---

## 🎯 核心类和主要函数

### 1. **HEMMer** (src/HEMMer.cpp, include/HELLM/HEMMer.hpp)

**职责**: 全同态加密的集中管理

```cpp
class HEMMer {
public:
    // 构造和初始化
    static HEMMer genHEMMer();                    // 单GPU模式
    static HEMMer genHEMMerMultiGPU();           // 多GPU (MPI)
    void save(const std::string& key_path);      // 保存密钥
    void load(const std::string& key_path);      // 加载密钥
    
    // 核心加密操作 ★★★
    CtxtTensor encrypt2(torch::Tensor a, torch::Tensor b);  // 加密
    std::vector<torch::Tensor> decrypt2(const CtxtTensor& ct); // 解密
    void bootstrap(CtxtTensor& ct);              // 恢复噪声等级
    void bootstrap2(CtxtTensor& ct1, CtxtTensor& ct2);
    
    // 同态评估器
    HomEvaluator& getEval();                     // 获取评估器
    
    // MPI相关
    int getRank();                               // 进程号
    int getMaxRank();                            // 总进程数
    
    // 路径管理
    std::string getHEPath();                     // HE结果路径
    std::string getWeightPath();                 // 权重路径
    std::string getTorchPath();                  // PyTorch数据路径
};
```

**关键操作**:
```cpp
// 加密张量 (使用 CKKS 方案)
auto ct = hemmer->encrypt2(plaintext_a, plaintext_b);  // ct = (a, b) 加密

// 同态运算 (在密文上直接计算)
hemmer->getEval().add(ct1.get(), ct2.get(), result.get());    // ct1 + ct2
hemmer->getEval().mult(ct.get(), scalar, result.get());       // ct * scalar
hemmer->getEval().square(ct.get(), result.get());             // ct^2
hemmer->getEval().sub(ct1.get(), ct2.get(), result.get());    // ct1 - ct2

// Bootstrap (恢复计算能力)
hemmer->bootstrap(ct);  // 使用私钥部分恢复
```

---

### 2. **LoRA 模块** (src/LoRA.cpp, include/HELLM/LoRA.hpp)

**职责**: 在加密域中实现 LoRA 微调

```cpp
class LoraModule {
private:
    HEMMer* hemmer_;                            // HE管理器
    u64 layer_n_;                               // 当前层号
    
public:
    // 权重初始化 ★
    void generateInitialLoraWeight(const std::string &lora_type);
    void zeroGrad(const std::string &lora_type);
    void zeroAggGrad(const std::string &lora_type);
    
    // 优化器 ★★★
    void AdamW(const std::string &lora_t, const char *task, int step);
    void AdamW_head(const char *task, int step);
    void AdamW_head2(const char *task, int step);
    
    // 优化步骤
    void optimizerStep(const std::string &lora_type);
    void optimizerStep_bert(const char *task, int step);
    void optimizerStep_head_bert(const char *task, int step);
    
    // 数据管理
    CtxtTensor getCtxtTensor_lora(const std::string &name, u64 i, u64 j, u64 k);
    void saveCtxtTensor_lora(const CtxtTensor &tensor, const std::string &name, ...);
};
```

**关键函数详解**:

#### `generateInitialLoraWeight()`
```cpp
// 初始化 LoRA 权重 (加密)
void LoraModule::generateInitialLoraWeight(const std::string &lora_type) {
    for (const char t : lora_type) {  // 'q', 'k', 'v'
        // 1. 生成下投影权重 (128 x 8)
        auto lora_wa = torch::empty({2, 128, 128});
        lora_wa = torch::nn::init::kaiming_uniform_(lora_wa, ...);
        
        // 2. 初始化上投影权重为零
        auto lora_wb = torch::zeros({2, 128, 128});
        
        // 3. 加密并保存
        saveCtxtTensor_lora(hemmer_->encrypt2(lora_wa[0], lora_wa[1]),
                           "lora_wa_" + lora_t, 0, 0, 0);
        
        // 4. 初始化 Adam 优化器状态 (momentum, variance)
        saveCtxtTensor_lora(hemmer_->encrypt2(zeros, zeros),
                           "momentum_ma_" + lora_t, 0, 0, 0);
        // ... 初始化其他状态
    }
}
```

#### `AdamW()` - 加密 AdamW 优化器
```cpp
void LoraModule::AdamW(const std::string &lora_t, const char *task, int step) {
    auto lr = ModelArgs::RTE_LRS[step];  // 获取学习率
    
    for (const char a : "ab") {  // 分别处理 wa 和 wb
        auto weight = getCtxtTensor_lora("lora_w" + case_, ...);
        auto agg_grad = getCtxtTensor_lora("agg_grad_w" + case_, ...);
        auto momentum_m = getCtxtTensor_lora("momentum_m" + case_, ...);
        auto momentum_v = getCtxtTensor_lora("momentum_v" + case_, ...);
        
        // ★ 在加密域中执行 AdamW 步骤
        
        // 1. 权重衰减
        // θ = θ * (1 - lr * weight_decay)
        hemmer_->getEval().mult(theta.get(), 1 - ModelArgs::WEIGHT_DECAY * lr,
                               theta.get());
        
        // 2. 检查是否需要 Bootstrap
        if (momentum_m.get().getLevel() < 4 + 3 ||
            momentum_v.get().getLevel() < 4 + 4) {
            hemmer_->bootstrap2(momentum_m, momentum_v);  // 恢复
        }
        
        // 3. 一阶矩 m = β₁*m + (1-β₁)*∇
        hemmer_->getEval().mult(momentum_m.get(), ModelArgs::BETA_1, ...);
        CtxtTensor grad_tmp{agg_grad};
        hemmer_->getEval().mult(grad_tmp.get(), 1 - ModelArgs::BETA_1, ...);
        hemmer_->addInplace(momentum_m, grad_tmp);
        
        // 4. 二阶矩 v = β₂*v + (1-β₂)*∇²
        hemmer_->getEval().mult(momentum_v.get(), ModelArgs::BETA_2, ...);
        CtxtTensor grad_square{agg_grad};
        hemmer_->getEval().square(grad_square.get(), grad_square.get());
        hemmer_->getEval().mult(grad_square.get(), 1 - ModelArgs::BETA_2, ...);
        hemmer_->addInplace(momentum_v, grad_square);
        
        // 5. Bias 修正
        CtxtTensor m_hat{momentum_m};
        CtxtTensor v_hat{momentum_v};
        hemmer_->getEval().mult(m_hat.get(), 
                               lr * (1.0/(1 - pow(BETA_1, step))), ...);
        hemmer_->getEval().mult(v_hat.get(),
                               1.0 / (1 - pow(BETA_2, step)), ...);
        
        // 6. 加 epsilon 并计算倒数平方根 (最复杂的部分!)
        hemmer_->getEval().add(v_hat.get(), ModelArgs::OPTI_EPS, ...);
        
        if (task == "C") {
            approxInverseSqrt_COLA(hemmer_->getEval(), ..., v_hat, ..., 3);
        } else if (task == "M") {
            approxInverseSqrt_MRPC(hemmer_->getEval(), ..., v_hat, ..., 3);
        }
        // ...其他任务
        
        // 7. 更新权重 θ = θ - lr * m̂/(√v̂+ε)
        hemmer_->hadamardMultInplace(m_hat, v_hat);  // m_hat * v_hat (元素乘)
        hemmer_->getEval().sub(theta.get(), m_hat.get(), theta.get());
        
        // 8. Bootstrap 恢复
        hemmer_->bootstrap(theta);
        
        // 9. 保存更新的权重
        saveCtxtTensor_lora(theta, "lora_w" + case_ + "_" + lora_t, ...);
        saveCtxtTensor_lora(momentum_m, "momentum_m" + case_ + "_" + lora_t, ...);
        saveCtxtTensor_lora(momentum_v, "momentum_v" + case_ + "_" + lora_t, ...);
    }
}
```

#### `approxInverseSqrt_*()` - 近似求倒数平方根

```cpp
// 使用 Chebyshev 多项式近似 1/√x (加密域中)
void approxInverseSqrt_COLA(const HomEvaluator &eval, const Bootstrapper &btp,
                            const Ciphertext &op, Ciphertext &res,
                            const u64 num_iter) {
    // 1. 线性变换: 输入映射到 [0, 1] 区间
    Ciphertext ctxt_x = linearTransform(eval, btp, op, InputInterval(0, 1));
    
    // 2. Bootstrap 确保足够的计算能力
    bootstrapIfNecessary(btp, ctxt_x, 1 + COLA_INVERSE_SQRT_127.level_cost);
    
    // 3. Chebyshev 多项式展开 (预计算的系数)
    //    1/√x ≈ Σ cᵢ·Tᵢ(x)  其中 Tᵢ 是 Chebyshev 多项式
    Ciphertext ctxt_y = evaluateChebyshevExpansion(eval, btp, ctxt_x,
                                                    COLA_INVERSE_SQRT_127, 1.0);
    
    // 4. 牛顿迭代精化 (如果需要)
    if (num_iter > 0) {
        approxInverseSqrtNewton_TS(eval, btp, op, ctxt_y, ctxt_y, num_iter);
    }
    
    res = ctxt_y;
}
```

---

### 3. **TransformerBlock** (src/TransformerBlock.cpp)

**职责**: Transformer 层的加密实现

```cpp
class TransformerBlock {
public:
    // 前向传播
    CtxtTensor forward(const CtxtTensor& input, ...);
    
    // 反向传播
    void backward(const CtxtTensor& grad_output, ...);
    
    // LoRA 相关
    CtxtTensor applyLora(const CtxtTensor& x, const std::string& lora_type);
};
```

---

### 4. **主程序入口**

#### (1) keygen.cpp - 密钥生成
```cpp
int main() {
    std::string key_path{std::getenv("HELLM_KEY_PATH")};
    std::filesystem::create_directory(key_path);
    
    // 生成密钥对 (单GPU)
    HELLM::HEMMer hemmer{HELLM::HEMMer::genHEMMer()};
    
    // 保存到文件系统
    hemmer.save(key_path);
}
```

#### (2) convert2.cpp - 权重转换
```cpp
int main(int argc, char *argv[]) {
    bool convert_generation = (argc == 2 && std::strcmp(argv[1], "-g") == 0);
    
    // 加载 HE 环境
    HELLM::HEMMer hemmer = HELLM::HEMMer::genHEMMer();
    
    // 加载 PyTorch 权重
    auto container = torch::jit::load("./data_2ly_mrpc/converted_weights_mrpc.pth");
    
    // 转换每个权重
    // • 遍历容器中的每个权重
    // • 用公钥加密
    // • 保存为 .bin 文件
}
```

#### (3) bert-test.cpp - 推理评估 ★★
```cpp
int main() {
    static const std::string LORA_TYPE = "qkv";
    
    // 多GPU 模式
    auto *hemmer = new HELLM::HEMMer{HELLM::HEMMer::genHEMMerMultiGPU()};
    MPI_Barrier(MPI_COMM_WORLD);
    
    HELLM::TransformerBlock block{hemmer, data_path, data_path, 0};
    int rank = hemmer->getRank();
    int size = hemmer->getMaxRank();
    
    // 加载加密权重
    auto container = torch::jit::load(data_path + "converted_weights_mrpc_eval.pth");
    
    // 对每个评估样本
    for (int j = 0; j < 51; ++j) {  // MRPC: 51个样本
        // 1. 加载输入 (明文)
        auto inp = container.attr("input_" + std::to_string(...)).toTensor();
        
        // 2. 加密输入
        for (long i = 0; i < N_HEAD/2; ++i) {
            auto input_ctxt = hemmer->encrypt2(inp[i*2], inp[i*2+1]);
            ctxt_cur.push_back(input_ctxt);
        }
        
        // 3. 前向传播 (加密域)
        auto output = block.forward(ctxt_cur, ...);
        
        // 4. [可选] 解密计算准确度
        auto dec_output = hemmer->decrypt2(output);
        
        // 5. 评估指标
    }
}
```

#### (4) backward-bert-multi.cpp - LoRA 训练 ★★★
```cpp
int main() {
    static const std::string LORA_TYPE = "qkv";
    auto task = "M";  // MRPC
    const HELLM::u64 one_epo_step = 458;
    const int num_gpu = 8;
    
    // 多GPU 初始化
    auto *hemmer = new HELLM::HEMMer{HELLM::HEMMer::genHEMMerMultiGPU()};
    MPI_Barrier(MPI_COMM_WORLD);
    
    int rank = hemmer->getRank();
    auto container = torch::jit::load("./data_2ly_mrpc/converted_weights_mrpc.pth");
    
    // ===== Phase 1: 初始化 LoRA 权重 =====
    if (rank == 0) {
        for (int i = 0; i < 2; ++i) {  // 2 层
            auto torch_block = std::make_shared<HELLM::TorchTransformerBlock>(
                hemmer, container, i);
            torch_block->generateInitialLoraWeight(LORA_TYPE);  // 生成 + 加密
        }
    }
    HEaaN::CudaTools::cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    
    // ===== Phase 2: 梯度累积初始化 =====
    if (rank == 0) {
        for (HEaaN::u64 layer_ = 0; layer_ < 2; ++layer_) {
            auto lora_module_ = std::make_shared<HELLM::LoRA::LoraModule>(hemmer, layer_);
            lora_module_->zeroAggGrad(LORA_TYPE);
            if (layer_ == 0) {
                lora_module_->zeroAggGrad_head();
                lora_module_->zeroAggGrad_head2();
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    // ===== Phase 3: 训练循环 =====
    for (int epo = 0; epo < num_epochs; ++epo) {
        for (HEaaN::u64 step = 0; step < one_epo_step; ++step) {
            // -------- 数据加载 --------
            auto [input_idx, label_idx] = loadBatch(...);
            auto inp = container.attr("input_" + ...).toTensor();
            auto label = container.attr("label_" + ...).toTensor();
            
            // -------- 前向传播 --------
            auto encrypted_input = hemmer->encrypt2(inp[...], inp[...]);
            auto output = block.forward(encrypted_input, ...);
            
            // -------- 损失计算 (加密) --------
            auto loss = computeLoss(output, encrypted_label);
            
            // -------- 反向传播 (加密) --------
            block.backward(loss, ...);
            
            // -------- 梯度聚合 (MPI AllReduce) --------
            MPI_Allreduce(...);
            
            // -------- 优化器更新 (AdamW, 加密) --------
            for (layer = 0; layer < 2; ++layer) {
                auto lora_module = std::make_shared<HELLM::LoRA::LoraModule>(hemmer, layer);
                
                if (rank == 0 || rank == 3) {
                    lora_module->AdamW("q", task, step);
                } else if (rank == 1 || rank == 4) {
                    lora_module->AdamW("k", task, step);
                } else if (rank == 2 || rank == 5) {
                    lora_module->AdamW("v", task, step);
                }
            }
            
            // 处理分类头
            if (rank >= 6) {
                auto lora_module = std::make_shared<HELLM::LoRA::LoraModule>(hemmer, 0);
                lora_module->AdamW_head(task, step);
                lora_module->AdamW_head2(task, step);
            }
        }
        
        // -------- Epoch 结束: 评估 --------
        auto accuracy = evaluate(hemmer, block, eval_data);
        std::cout << "Epoch " << epo << " Accuracy: " << accuracy << std::endl;
    }
}
```

---

## 📊 ModelArgs.hpp 中的关键配置

```cpp
// ===== 模型结构参数 =====
static const u64 DIM = 768;           // 隐藏层维度
static const u64 N_HEAD = 12;         // 注意力头数
static const u64 HEAD_DIM = 64;       // 每头维度 (768/12)
static const u64 LOW_DIM = 8;         // LoRA 秩
static const u64 NUM_LAYER = 2;       // 层数

// ===== 数据相关参数 =====
static const u64 MAX_SEQ_LEN = 128;   // 最大序列长度
static const u64 VOCAB_SIZE = 30522;  // 词表大小

// ===== 训练超参数 =====
static const double LEARNING_RATE = 2e-4;
static const double WEIGHT_DECAY = 0.01;
static const double BETA_1 = 0.9;     // Adam β₁
static const double BETA_2 = 0.999;   // Adam β₂
static const double OPTI_EPS = 1e-8;  // 优化器 epsilon

// ===== 任务特定的学习率表 =====
static const double RTE_LRS[310] = {...};   // RTE 学习率表
static const double COLA_LRS[1068] = {...};
static const double MRPC_LRS[458] = {...};
// ...

// ===== LoRA 初始化 =====
static const double LORA_MEAN = 0.0;
static const double LORA_SQUARE_STD = 0.02;
```

---

## 🔄 数据流追踪例子

### 例1: 单个权重的更新流程

```
明文权重 (PyTorch)
    ↓
convert 程序
    ↓ [公钥加密]
密文权重 (*.bin)
    ↓ [梯度计算期间保存]
梯度 (密文, *.bin)
    ↓ [AdamW 优化]
更新的密文权重
    ↓ [保存]
(*.bin) 用于下一步
```

### 例2: 单个训练步骤

```
加密输入 (2×128×768)
    ↓
TransformerBlock::forward()
    ├─ Self-Attention (Q·K^T / √d_k + LoRA)
    ├─ LayerNorm (Remez 多项式近似)
    ├─ FeedForward (Linear + ReLU + Linear)
    └─ LoRA 适配器 (x → Wa·x + Wb·y)
    ↓
加密输出 (2×128×768)
    ↓
计算 Logits (分类头)
    ↓
加密 Loss (标量)
    ↓
TransformerBlock::backward()
    ├─ ∂L/∂output
    ├─ ∂L/∂Wa, ∂L/∂Wb (LoRA 梯度)
    └─ ∂L/∂input (下一层输入)
    ↓
加密梯度
    ↓ [MPI AllReduce]
聚合梯度
    ↓
LoRA::AdamW()
    ├─ 更新 m (一阶矩)
    ├─ 更新 v (二阶矩)
    ├─ approxInverseSqrt(v)
    ├─ θ ← θ - lr·m/(√v+ε)
    └─ Bootstrap
    ↓
更新的加密权重 (保存到 *.bin)
```

---

## 🎯 代码阅读建议路径

### 初级: 理解基本流程
1. **keygen.cpp** → 理解密钥生成
2. **convert2.cpp** → 理解权重转换
3. **HEMMer.hpp** → 理解 API

### 中级: 推理流程
1. **bert-test.cpp** → 理解推理主循环
2. **TransformerBlock.hpp** → 理解前向传播
3. **HEMMer.cpp** → 理解加密操作

### 高级: 训练流程
1. **backward-bert-multi.cpp** → 理解训练主循环
2. **LoRA.cpp::AdamW()** → 理解优化器
3. **LoRA.cpp::approxInverseSqrt_*()** → 理解多项式近似

---

## 📌 关键代码片段速查

| 功能 | 文件 | 函数 | 行数 |
|------|------|------|------|
| 密钥生成 | examples/keygen.cpp | main() | 15-23 |
| 加权转换 | examples/convert2.cpp | main() | 25-80 |
| 推理循环 | examples/bert-test.cpp | main() | 30-120 |
| 训练循环 | examples/backward-bert-multi.cpp | main() | 50-300 |
| LoRA初始化 | src/LoRA.cpp | generateInitialLoraWeight() | 25-80 |
| AdamW优化 | src/LoRA.cpp | AdamW() | 250-400 |
| 倒数平方根 | src/LoRA.cpp | approxInverseSqrt_COLA() | 550-570 |
| 加密操作 | src/HEMMer.cpp | encrypt2() | 50-100 |
| 同态运算 | src/HEMMer.cpp | getEval() | 150-200 |
| Bootstrap | src/HEMMer.cpp | bootstrap() | 300-350 |

