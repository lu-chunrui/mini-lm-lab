# Mini Transformer Lab

从零实现一个小型 Decoder-Only Transformer。项目按每日练习逐步推进，重点不是调用现成大模型，而是理解语言模型的数据处理、训练流程、因果自注意力和文本生成。

## 当前状态

目前已完成从线性回归到多头因果注意力语言模型的学习版本。`day_5train.py` 可以在小规模重复文本上训练并生成字符级文本。

> 当前生成结果主要反映模型对少量重复训练语料的拟合能力，不代表模型已经具备通用语言理解或生成能力。

## 每日练习

| 文件 | 学习内容 | 完成情况 |
| --- | --- | --- |
| `day_1train.py` | 使用 `nn.Linear`、MSE 和 SGD 学习线性关系 `y = 2x + 1`，掌握基本训练循环 | 已完成 |
| `day_2train.py` | 字符级 Bigram 语言模型，学习词表、编码解码、Embedding、交叉熵和自回归生成 | 已完成 |
| `day_3train.py` | 固定上下文窗口语言模型，学习 `batch_size`、`block_size`、批量采样与 MLP 预测 | 已完成 |
| `day_4train.py` | 手动计算 Q、K、V、缩放点积、因果遮罩、Softmax 和 `weights @ value` | 已完成 |
| `day_4.2train.py` | 将单头因果自注意力封装成模块，加入 Token/Position Embedding，并训练单头注意力语言模型 | 已完成 |
| `day_5train.py` | 使用 `ModuleList` 实现多头因果注意力，拼接多个头并进行输出投影 | 已完成 |

辅助练习：

- `python/autograd_demo.py`：PyTorch 自动求导练习。
- `cpp/`：C++ 与算法基础练习，与语言模型主线相互独立。
- `notes/daily-log.md`：预留的每日学习记录。

## 已实现的数据流

以 `day_5train.py` 为例：

```text
字符文本
  → 字符ID                         [B, T]
  → Token Embedding + Position Embedding
                                   [B, T, embedding_dim]
  → 多头因果自注意力                [B, T, embedding_dim]
  → 词表输出层                      [B, T, vocab_size]
  → Cross Entropy Loss
```

多头注意力内部：

```text
同一份输入 [B, T, 32]
  ├─ Head 1 → [B, T, 8]
  ├─ Head 2 → [B, T, 8]
  ├─ Head 3 → [B, T, 8]
  └─ Head 4 → [B, T, 8]
       ↓ 沿特征维拼接
     [B, T, 32]
       ↓ 输出投影
     [B, T, 32]
```

## 当前实验配置

`day_5train.py` 当前使用：

- 字符级词表：43 个字符（由当前训练文本生成）
- 上下文长度：8
- Batch Size：32
- Embedding Dimension：32
- Attention Heads：4
- Head Size：8
- 优化器：AdamW
- 学习率：0.001
- 训练步数：1500

一次实际运行中，训练 loss 从约 `3.80` 降至 `0.01～0.06`。生成结果能够复现大部分训练句式，但会出现“机器学习机器学习”等重复片段。这符合当前数据规模小、上下文短、模型层数少的特点。

## 环境安装

推荐使用 Python 虚拟环境。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

如果出现 `No module named 'numpy'` 警告，可补充安装：

```powershell
python -m pip install numpy
```

## 运行

运行当前最新练习：

```powershell
python day_5train.py
```

也可以按顺序运行每日练习：

```powershell
python day_1train.py
python day_2train.py
python day_3train.py
python day_4train.py
python day_4.2train.py
python day_5train.py
```

## 当前局限

- 训练文本很少并被重复多次，训练 loss 不能反映泛化能力。
- 当前主要使用字符级 Tokenizer，尚未实现 BPE 等子词分词。
- 尚未加入前馈网络、残差连接和归一化，因此还不是完整的 Transformer Block。
- 尚未实现训练集/验证集的可靠对比、Checkpoint 和速度基准。
- 尚未实现 RoPE、RMSNorm、SwiGLU 与 KV Cache。

## 后续计划

- [ ] 加入前馈网络和残差连接
- [ ] 加入 LayerNorm，组成基础 Transformer Block
- [ ] 堆叠多个 Transformer Block
- [ ] 加入训练集/验证集评估、Checkpoint 和 loss 曲线
- [ ] 用 RMSNorm 替换 LayerNorm
- [ ] 用 SwiGLU 替换普通前馈网络
- [ ] 用 RoPE 替换可学习的位置向量
- [ ] 使用 TinyStories 小规模子集训练
- [ ] 实现 KV Cache
- [ ] 对比有无 KV Cache 的生成速度和内存占用

## 项目定位

这是一个面向学习和实验的实现项目。最终目标是形成一套代码可运行、模块可解释、实验可复现的小型语言模型，并记录不同结构与推理优化方案的对比结果。

