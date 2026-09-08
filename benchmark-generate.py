from pathlib import Path
import time

import torch

from config import device
from bpe_tokenizer import BPETokenizer
from model import TransformerLanguageModel

project_dir = Path(__file__).resolve().parent
checkpoint_dir = project_dir / "checkpoints"
checkpoint_path = checkpoint_dir / "best_bpe_model.pth"
tokenizer_path = checkpoint_dir / "bpe_tokenizer.json"
if not checkpoint_path.exists():
    raise FileNotFoundError(
        f"找不到模型：{checkpoint_path}"
    )

if not tokenizer_path.exists():
    raise FileNotFoundError(
        f"找不到Tokenizer：{tokenizer_path}"
    )
tokenizer = BPETokenizer.load(tokenizer_path)
checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)
model = TransformerLanguageModel(
    vocab_size=checkpoint["vocab_size"],
    embedding_dim=checkpoint["embedding_dim"],
    num_heads=checkpoint["num_heads"],
    num_layers=checkpoint["num_layers"],
    feedforward_dim=checkpoint["feedforward_dim"],
    max_seq_len=checkpoint["max_seq_len"]
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)
model.eval()
print("模型加载成功")
print("运行设备：", device)
print("词表大小：", tokenizer.vocab_size)
print("验证Loss：", checkpoint["val_loss"])

prompt = "Once upon a time"
prompt_ids = tokenizer.encode(prompt)
unknown_id = tokenizer.token_to_id[
    tokenizer.unknown_token
]
if unknown_id in prompt_ids:
    print("警告：提示词中包含未知Token")
input_idx = torch.tensor(
    [prompt_ids],
    dtype=torch.long,
    device=device
)
max_new_tokens = 80
total_length = (
    input_idx.shape[1]
    + max_new_tokens
)

if total_length > checkpoint["max_seq_len"]:
    raise ValueError(
        f"总长度{total_length}超过模型最大长度"
        f"{checkpoint['max_seq_len']}"
    )
print("\n提示词：", prompt)
print("Token ID：", prompt_ids)
print("输入形状：", input_idx.shape)
print("生成Token数：", max_new_tokens)
print("生成后总长度：", total_length)

def synchronize_device():
    if device.type == "cuda":
        torch.cuda.synchronize()
print("\n开始预热无Cache生成……")
torch.manual_seed(42)
_ = model.generate_without_cache(
    input_idx.clone(),
    max_new_tokens=10,
    temperature=0.5,
    top_k=1
)

synchronize_device()
print("无Cache预热完成")
repeat_times = 5
without_cache_times = []
without_cache_output = None

print("\n开始测试无Cache生成速度……")

for run_index in range(repeat_times):
    torch.manual_seed(42)
    synchronize_device()
    start_time = time.perf_counter()
    without_cache_output = model.generate_without_cache(
        input_idx.clone(),
        max_new_tokens=max_new_tokens,
        temperature=0.5,
        top_k=1
    )
    synchronize_device()
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    without_cache_times.append(elapsed_time)
    print(
        f"第{run_index + 1}次："
        f"{elapsed_time:.4f}秒"
    )
    average_without_cache_time = (
    sum(without_cache_times)
    / len(without_cache_times)
)

without_cache_tokens_per_second = (
    max_new_tokens
    / average_without_cache_time
)
print("\n无Cache测试结果：")
print(
    "平均耗时：",
    f"{average_without_cache_time:.4f}秒"
)
print(
    "生成速度：",
    f"{without_cache_tokens_per_second:.2f} tokens/s"
)
print("\n开始预热KV Cache生成……")

torch.manual_seed(42)

_ = model.generate_with_cache(
    input_idx.clone(),
    max_new_tokens=10,
    temperature=0.5,
    top_k=1
)
synchronize_device()
print("KV Cache预热完成")
with_cache_times = []
with_cache_output = None
print("\n开始测试KV Cache生成速度……")
for run_index in range(repeat_times):
    torch.manual_seed(42)
    synchronize_device()
    start_time = time.perf_counter()
    with_cache_output = model.generate_with_cache(
        input_idx.clone(),
        max_new_tokens=max_new_tokens,
        temperature=0.5,
        top_k=1
    )
    synchronize_device()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    with_cache_times.append(elapsed_time)
    print(
        f"第{run_index + 1}次："
        f"{elapsed_time:.4f}秒"
    )
    average_with_cache_time = (
    sum(with_cache_times)
    / len(with_cache_times)
)
with_cache_tokens_per_second = (
    max_new_tokens
    / average_with_cache_time
)
print("\nKV Cache测试结果：")
print(
    "平均耗时：",
    f"{average_with_cache_time:.4f}秒"
)
print(
    "生成速度：",
    f"{with_cache_tokens_per_second:.2f} tokens/s"
)
outputs_are_equal = torch.equal(
    without_cache_output,
    with_cache_output
)

speedup = (
    average_without_cache_time
    / average_with_cache_time
)

print("\n==============================")
print("KV Cache最终对比")
print("==============================")

print(
    "无Cache平均耗时：",
    f"{average_without_cache_time:.4f}秒"
)

print(
    "有Cache平均耗时：",
    f"{average_with_cache_time:.4f}秒"
)

print(
    "无Cache生成速度：",
    f"{without_cache_tokens_per_second:.2f} tokens/s"
)

print(
    "有Cache生成速度：",
    f"{with_cache_tokens_per_second:.2f} tokens/s"
)
print("两种方法输出是否一致：", outputs_are_equal)
print("KV Cache加速倍数：", f"{speedup:.2f}x")
without_cache_text = tokenizer.decode(
    without_cache_output[0].tolist()
)

with_cache_text = tokenizer.decode(
    with_cache_output[0].tolist()
)

print("\n无Cache生成结果：")
print(without_cache_text)

print("\n有Cache生成结果：")
print(with_cache_text)