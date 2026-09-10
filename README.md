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
- **实现 BPE Tokenizer**：支持合并规则学习、文本编码解码、词表构建以及 JSON 保存与加载。
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

