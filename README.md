# Mini Transformer Lab：从零实现 Decoder-Only Transformer

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-From%20Scratch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-TinyStories-FFD21E)](https://huggingface.co/datasets/roneneldan/TinyStories)
[![Status](https://img.shields.io/badge/Status-Project%20Complete-success)](#核心结果)

这是一个使用 PyTorch 从零实现的小型 Decoder-Only Transformer 语言模型项目。项目不调用 `nn.Transformer`，而是从张量运算出发实现注意力、归一化、前馈网络、位置编码、训练和自回归生成，并在 TinyStories 小规模数据集上跑通完整流程。

当前项目已经完成从“基础原理练习”到“可训练语言模型”的闭环，并进一步加入了 BPE Tokenizer、现代 LLM 组件、KV Cache 推理、学习率调度、断点续训和实验日志。在此基础上，项目完成了前馈网络与模型深度对比实验。目前主体功能和实验均已收尾，代码进入稳定维护状态，不再继续扩展模型功能。

> 项目重点不是追求大模型级别的生成质量，而是理解并验证语言模型从数据、分词、建模、训练到推理的完整工作流程。

---

## 项目亮点

- **从零实现 Decoder-Only Transformer**：手写因果自注意力、多头注意力、Transformer Block 和语言模型输出层。
- **采用现代 LLM 组件**：实现 RoPE、RMSNorm、SwiGLU 和 Pre-Norm 残差结构。
- **实现教学版 BPE Tokenizer**：支持合并规则学习、文本编码解码、词表构建以及 JSON 保存与加载。
- **接入 TinyStories**：使用 Hugging Face 流式加载小规模真实英文故事数据。
- **完整训练流程**：包含随机批量采样、训练集/验证集评估、梯度裁剪、Warmup + Cosine Decay、最佳模型保存和断点续训。
- **KV Cache 推理加速**：区分 Prefill 与逐 Token Decode，复用历史 K/V；在当前 80-Token CPU 生成实验中实现 **1.75×** 端到端加速。
- **归一化消融实验**：在相同配置下比较 RMSNorm 与 LayerNorm，并通过两次重复实验分析结果波动。
- **前馈网络对比实验**：在严格匹配参数量的条件下比较 SwiGLU 与 GELU，两个随机种子下 SwiGLU 均取得更低的验证 Loss。
- **模型深度对比实验**：比较 2 层与 4 层 Transformer，量化模型容量增加带来的效果提升与参数成本。
- **可复现实验记录**：按实验名称分别保存 Checkpoint、学习率、训练 Loss 和验证 Loss，便于绘图与公平比较。
- **保留渐进式学习记录**：通过每日代码展示模型从线性回归、Bigram 到完整 Transformer 的演进过程。

---

## 核心结果

### TinyStories 小规模训练

下表记录早期 TinyStories 小规模基线实验结果。当前实验代码已经进一步支持独立实验目录、学习率调度和断点续训，后续对比均以相同数据、Tokenizer、随机种子与训练预算为控制条件。

| 指标 | 当前结果 |
| --- | ---: |
| BPE 词表大小 | 124 |
| 训练 Token 数量 | 253,147 |
| 验证 Token 数量 | 257,108 |
| 模型参数量 | 360,064 |
| Transformer 层数 | 2 |
| 隐藏维度 | 128 |
| 注意力头数 | 4 |
| 训练上下文长度 | 128 |
| 训练步数 | 2,000 |
| 初始验证 Loss | 5.0060 |
| 最佳验证 Loss | **1.9093** |
| 对应 Perplexity | 约 **6.75** |
| 无 KV Cache 生成速度 | 47.00 tokens/s |
| KV Cache 生成速度 | **82.39 tokens/s** |
| KV Cache 加速倍数 | **1.75×** |

验证 Loss 从约 `5.01` 稳定下降到 `1.91`，训练 Loss 最终约为 `1.70`。训练集与验证集曲线走势接近，说明模型已经学习到 TinyStories 中常见的局部语法和叙事模式，暂未出现严重过拟合。

### 文本生成示例

Prompt：

```text
once upon a time
```

模型能够生成包含常见童话表达、人物和基本句式的英文文本，但仍会出现拼写错误、重复和长程逻辑不连贯：

```text
once upon a time there was a mirror. One day, ...
the little girl ... saw the sky ...
```

这是当前小数据、低参数量和小词表设定下的预期现象。本项目关注完整训练流程与组件实验，不以追求大模型级生成质量为目标。

### KV Cache

项目已实现两种生成路径：

- `generate_without_cache()`：每一步重新计算完整上下文。
- `generate_with_cache()`：Prefill 后缓存每一层、每个注意力头的 K/V，后续只输入最新 Token。

在相同模型、Prompt 和生成参数下进行 5 次重复测试，得到以下结果：

| 生成方式 | 平均耗时 | 生成速度 |
| --- | ---: | ---: |
| 无 KV Cache | 1.7021 秒 | 47.00 tokens/s |
| 使用 KV Cache | **0.9710 秒** | **82.39 tokens/s** |

实验配置：

- Prompt：`Once upon a time`
- Prompt Token 数量：11
- 生成 Token 数量：80
- Temperature：0.5
- Top-k：1
- 重复次数：5
- 计时方式：`time.perf_counter()`

两种方法生成的 Token 序列完全一致，说明缓存没有改变模型输出。在当前 36 万参数、两层 Transformer 的 CPU 实验中，KV Cache 将生成吞吐量从 47.00 tokens/s 提升到 82.39 tokens/s，获得约 **1.75×** 的端到端加速。

### RMSNorm 与 LayerNorm 消融实验

为验证归一化方式对当前小模型的影响，项目在数据、Tokenizer、模型规模、训练步数和学习率调度保持一致的条件下，分别训练 RMSNorm 与 LayerNorm 版本。每次实验训练 1,500 步，并以训练期间记录到的最低验证 Loss 作为比较指标。

| 重复实验 | LayerNorm 最佳验证 Loss | RMSNorm 最佳验证 Loss | 本次较优方案 |
| --- | ---: | ---: | --- |
| 实验 1 | 2.0319 | **2.0227** | RMSNorm |
| 实验 2 | **1.9688** | 1.9833 | LayerNorm |
| 两次平均 | **2.0003** | 2.0030 | 基本持平 |

第一次实验中 RMSNorm 略优约 `0.0092`，第二次实验中 LayerNorm 略优约 `0.0145`；两次实验的最佳验证 Loss 均值只相差约 `0.0027`。因此，在当前两层、小规模 TinyStories 设定下，尚无证据表明其中一种归一化方式具有稳定优势，实验波动大于两种结构之间的差异。

曲线在个别评估点出现同步下降或回升，主要与验证阶段随机抽取批次带来的噪声有关。因此，这组结果作为探索性实验记录，结论限定为：在当前规模下，两种归一化方式没有表现出稳定的显著差异。

### SwiGLU 与 GELU 对比实验

为比较门控前馈网络与传统 Transformer 前馈网络，实验保持数据、Tokenizer、模型深度、归一化方式、学习率调度和训练步数一致，仅替换前馈网络。SwiGLU 使用隐藏维度 256；GELU 使用隐藏维度 384，使两组两层模型的参数量均为 394,368，从而避免参数规模差异干扰结果。

| 随机种子 | SwiGLU 最佳验证 Loss | GELU 最佳验证 Loss | 本次较优方案 |
| --- | ---: | ---: | --- |
| 42 | **2.0245** | 2.0918 | SwiGLU |
| 123 | **2.0192** | 2.0910 | SwiGLU |
| 两次平均 | **2.0218** | 2.0914 | SwiGLU |

两个随机种子下，SwiGLU 均取得更低的验证 Loss。两次实验的平均最佳验证 Loss 比 GELU 低约 `0.0696`，对应平均 Perplexity 约从 `8.10` 降至 `7.55`。结果表明，在当前等参数量的小型 Transformer 上，SwiGLU 相比 GELU 具有更好的学习效果和跨随机种子一致性。

### 2 层与 4 层模型对比

在随机种子 123 下，将 Transformer 从 2 层增加到 4 层，并保持隐藏维度、注意力头数、数据和训练预算不变。该实验用于观察增加模型容量后验证性能的变化。

| 前馈网络 | 2 层最佳验证 Loss | 4 层最佳验证 Loss | 2 层参数量 | 4 层参数量 |
| --- | ---: | ---: | ---: | ---: |
| SwiGLU | 2.0192 | **1.9060** | 394,368 | 755,328 |
| GELU | 2.0910 | **2.0142** | 394,368 | 755,328 |

4 层模型在两种前馈网络下都获得了更低的最佳验证 Loss，其中 SwiGLU 的改善约为 `0.1132`，GELU 的改善约为 `0.0768`。这说明增加模型深度能够提升当前任务上的建模能力，但参数量也由约 39.4 万增加至约 75.5 万。综合所有已完成实验，4 层 SwiGLU 模型取得最低验证 Loss `1.9060`。

---

## 模型架构

```text
文本
  ↓
BPE Tokenizer
  ↓
Token IDs                         [B, T]
  ↓
Token Embedding                   [B, T, d_model]
  ↓
N × Transformer Block
  ├─ RMSNorm
  ├─ Causal Multi-Head Attention
  │    ├─ Q / K / V Projection
  │    ├─ RoPE
  │    ├─ Scaled Dot-Product Attention
  │    └─ KV Cache（生成阶段）
  ├─ Residual Connection
  ├─ RMSNorm
  ├─ SwiGLU Feed-Forward Network
  └─ Residual Connection
  ↓
Final RMSNorm
  ↓
Vocabulary Projection             [B, T, vocab_size]
  ↓
Cross-Entropy Loss / Token Sampling
```

### 核心模块

#### RMSNorm

使用均方根对隐藏状态进行归一化，并通过可学习缩放参数恢复表达能力：

```python
mean_square = x.pow(2).mean(dim=-1, keepdim=True)
inverse_rms = torch.rsqrt(mean_square + eps)
output = x * inverse_rms * weight
```

#### RoPE

对 Q 和 K 的成对维度施加与位置相关的旋转，使注意力分数包含相对位置信息。KV Cache 解码时通过 `position_offset` 保证新 Token 使用正确的绝对位置。

#### Causal Multi-Head Attention

每个注意力头独立计算 Q、K、V 和缩放点积注意力，使用因果遮罩阻止当前位置看到未来 Token。各个头的输出拼接后再经过线性投影。

#### SwiGLU

前馈网络使用门控结构：

```python
output = output_proj(
    silu(gate_linear(x)) * value_linear(x)
)
```

#### KV Cache

生成第一个 Token 前进行 Prefill，得到所有历史 Token 的 K/V；后续 Decode 阶段只计算最新 Token 的 Q/K/V，并将新的 K/V 追加到缓存的序列维。

---

## 项目结构

```text
mini-transformer/
├── model.py                 # Transformer、RoPE、RMSNorm、SwiGLU、KV Cache
├── bpe_tokenizer.py         # 教学版 BPE 的训练、编码、解码与持久化
├── prepare_data.py          # 流式获取 TinyStories 小规模数据
├── train.py                 # 训练、验证、梯度裁剪与 Checkpoint
├── generate.py              # 加载模型并使用 KV Cache 生成文本
├── config.py                # 模型、训练和生成超参数
├── tokenizer.py             # 早期字符级 Tokenizer 练习
├── data/
│   ├── tinystories_train.txt
│   └── tinystories_val.txt
├── checkpoints/
│   ├── swiglu_42/           # 2 层 SwiGLU，随机种子 42
│   ├── gelu_42/             # 2 层 GELU，随机种子 42
│   ├── swiglu_123/          # 2 层 SwiGLU，随机种子 123
│   ├── gelu_123/            # 2 层 GELU，随机种子 123
│   ├── swiglu_2_123/        # 4 层 SwiGLU，随机种子 123
│   └── gelu_2_123/          # 4 层 GELU，随机种子 123
├── experiments/
│   └── <experiment_name>/   # 各组训练 Loss、验证 Loss 与学习率日志
├── training/                # 从基础模型到现代 Transformer 的每日练习
├── requirements.txt
└── README.md
```

---

## 快速开始

### 1. 创建并激活虚拟环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. 准备 TinyStories 小规模数据

```powershell
python prepare_data.py
```

该脚本通过流式读取只保存指定数量的故事，不需要下载完整 TinyStories 数据集。

### 3. 训练模型

```powershell
python train.py
```

训练过程会：

1. 在训练集上学习 BPE 合并规则。
2. 使用同一套 Tokenizer 编码训练集和验证集。
3. 定期计算训练 Loss 和验证 Loss。
4. 对梯度进行裁剪。
5. 使用 Warmup + Cosine Decay 调整学习率。
6. 根据验证 Loss 保存最佳模型，同时保存最近训练状态以支持断点续训。
7. 将 Step、学习率、训练 Loss 和验证 Loss 写入 CSV 实验日志。

### 4. 生成文本

```powershell
python generate.py
```

可以在 `generate.py` 中调整：

```python
prompt = "Once upon a time"
temperature = 0.5
top_k = 3
max_new_tokens = 80
```

建议保证“提示词 Token 数 + 生成 Token 数”不超过训练时的上下文长度。

---

## 每日实现路线

| 阶段 | 学习与实现内容 | 状态 |
| --- | --- | --- |
| Day 1 | 线性回归、Loss、反向传播和优化器 | 已完成 |
| Day 2 | 字符级 Bigram 语言模型 | 已完成 |
| Day 3 | 固定上下文窗口与批量采样 | 已完成 |
| Day 4 | Q/K/V、缩放点积、Softmax 与因果遮罩 | 已完成 |
| Day 5 | 多头因果自注意力 | 已完成 |
| Day 6 | 前馈网络、残差连接、归一化与多层 Block | 已完成 |
| Day 7 | RMSNorm、SwiGLU 与 RoPE | 已完成 |
| Day 8 | KV Cache 与模块拆分 | 已完成 |
| Day 9 | 教学版 BPE Tokenizer | 已完成 |
| Day 10 | TinyStories 数据、训练、验证和生成闭环 | 已完成 |

---

## 当前局限

- 模型只有约 36 万参数，难以学习稳定的长程叙事逻辑。
- 当前训练数据只是 TinyStories 的小规模子集。
- 教学版 BPE 基于 `text.split()`，会丢失原始空白信息，不是完整的 Byte-Level BPE。
- BPE 合并使用重复扫描实现，大数据上的训练效率较低。
- 尚未实现 EOS 终止逻辑，模型只能按照最大 Token 数停止生成。
- 验证阶段仍采用随机批次估计，单个评估点可能存在一定噪声。
- 部分实验只使用一个或两个随机种子，结论仅适用于当前小模型与数据配置。
- 项目定位为教学与实验验证，不包含分布式训练、混合精度训练和大规模性能优化。

---

## 后续计划

- [x] 记录训练与验证 Loss，并绘制训练曲线
- [x] 比较有无 KV Cache 的生成速度和吞吐量（80 Token，1.75×）
- [x] 加入 Warmup + Cosine Learning Rate Schedule
- [x] 支持断点续训和训练状态恢复
- [x] 完成 RMSNorm 与 LayerNorm 消融实验（两次重复实验，结果基本持平）
- [x] 完成 SwiGLU 与 GELU 的等参数量对比实验
- [x] 使用两个随机种子验证前馈网络实验的一致性
- [x] 完成 2 层与 4 层 Transformer 深度对比
- [x] 汇总参数量、最佳验证 Loss、Perplexity 和训练曲线


---

## 项目定位

本项目是一个面向大模型基础原理、训练工程和推理优化的学习型项目。它展示了如何从最小语言模型逐步构建完整的 Decoder-Only Transformer，并通过真实数据完成训练、验证、保存、加载和生成。

当前项目已经完成预定的模型实现、训练闭环、推理优化和组件对比实验。最终成果展示了从原理实现到可复现实验的完整过程，可用于课程展示、课题组申请和个人简历中的小型 LLM Research Engineering 项目经历。

## 致谢

- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)：用于小型语言模型训练的英文故事数据集。
- [PyTorch](https://pytorch.org/)：提供基础张量运算、自动求导和模块系统。
