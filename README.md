# Mini Transformer Lab：从零实现 Decoder-Only Transformer

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-From%20Scratch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-TinyStories-FFD21E)](https://huggingface.co/datasets/roneneldan/TinyStories)
[![Status](https://img.shields.io/badge/Status-Training%20Pipeline%20Complete-success)](#核心结果)

这是一个使用 PyTorch 从零实现的小型 Decoder-Only Transformer 语言模型项目。项目不调用 `nn.Transformer`，而是从张量运算出发实现注意力、归一化、前馈网络、位置编码、训练和自回归生成，并在 TinyStories 小规模数据集上跑通完整流程。

当前项目已经完成从“基础原理练习”到“可训练语言模型”的闭环，并进一步加入了 BPE Tokenizer、现代 LLM 组件和 KV Cache 推理支持。

> 项目重点不是追求大模型级别的生成质量，而是理解并验证语言模型从数据、分词、建模、训练到推理的完整工作流程。

---

## 项目亮点

- **从零实现 Decoder-Only Transformer**：手写因果自注意力、多头注意力、Transformer Block 和语言模型输出层。
- **采用现代 LLM 组件**：实现 RoPE、RMSNorm、SwiGLU 和 Pre-Norm 残差结构。
- **实现教学版 BPE Tokenizer**：支持合并规则学习、文本编码解码、词表构建以及 JSON 保存与加载。
- **接入 TinyStories**：使用 Hugging Face 流式加载小规模真实英文故事数据。
- **完整训练流程**：包含随机批量采样、训练集/验证集评估、梯度裁剪和最佳模型保存。
- **KV Cache 推理加速**：区分 Prefill 与逐 Token Decode，复用历史 K/V；在当前 80-Token CPU 生成实验中实现 **1.75×** 端到端加速。
- **保留渐进式学习记录**：通过每日代码展示模型从线性回归、Bigram 到完整 Transformer 的演进过程。

---

## 核心结果

### TinyStories 小规模训练

当前实验使用约 500 篇训练故事和 500 篇验证故事，模型成功完成训练并保存最佳检查点。

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

这是当前小数据、低参数量和小词表设定下的预期现象。项目后续将通过更大的 BPE 词表、更多训练数据和更完整的训练策略改善生成质量。

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
│   ├── best_bpe_model.pth
│   └── bpe_tokenizer.json
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
5. 根据验证 Loss 保存最佳模型。

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
- 尚未加入学习率 Warmup、Cosine Decay 和断点续训。
- 尚未完成单元测试、不同生成长度的 KV Cache 曲线和训练 Loss 可视化。

---

## 后续计划

- [ ] 记录训练与验证 Loss，并绘制训练曲线
- [x] 比较有无 KV Cache 的生成速度和吞吐量（80 Token，1.75×）
- [ ] 比较不同生成长度下的 KV Cache 加速趋势
- [ ] 为 RoPE、RMSNorm、Tokenizer 和 KV Cache 编写单元测试
- [ ] 加入 Warmup + Cosine Learning Rate Schedule
- [ ] 支持断点续训和完整训练状态恢复
- [ ] 加入 EOS 特殊 Token 与提前停止生成
- [ ] 将教学版 BPE 升级为 Byte-Level BPE
- [ ] 扩大 TinyStories 数据规模和模型参数量
- [ ] 进行模型深度、上下文长度和词表大小的消融实验

---

## 项目定位

本项目是一个面向大模型基础原理、训练工程和推理优化的学习型项目。它展示了如何从最小语言模型逐步构建完整的 Decoder-Only Transformer，并通过真实数据完成训练、验证、保存、加载和生成。

后续目标是将项目从“功能正确”继续推进到“实验可复现、结果可量化、代码可测试”，最终形成适合课程展示、课题组申请和个人简历陈述的小型 LLM Research Engineering 项目。

## 致谢

- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)：用于小型语言模型训练的英文故事数据集。
- [PyTorch](https://pytorch.org/)：提供基础张量运算、自动求导和模块系统。
