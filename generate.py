from pathlib import Path
import torch
from config import device
from bpe_tokenizer import BPETokenizer
from model import TransformerLanguageModel

project_dir = Path(__file__).resolve().parent
checkpoint_path = project_dir / "checkpoints" / "best_bpe_model.pth"

if not checkpoint_path.exists():raise FileNotFoundError(f"没有找到模型文件：{checkpoint_path}\n"
    "请先运行python train.py")
checkpoint=torch.load(checkpoint_path,map_location=device)
print("加载模型：", checkpoint_path)
print("训练step：", checkpoint["step"])
print("验证loss：", checkpoint["val_loss"])

tokenizer_path = project_dir / "checkpoints" / "bpe_tokenizer.json"
tokenizer = BPETokenizer.load(tokenizer_path)
print("词表大小：", tokenizer.vocab_size)
model = TransformerLanguageModel(
    vocab_size=checkpoint["vocab_size"],
    embedding_dim=checkpoint["embedding_dim"],
    num_heads=checkpoint["num_heads"],
    num_layers=checkpoint["num_layers"],
    feedforward_dim=checkpoint[
        "feedforward_dim"
    ],
    max_seq_len=checkpoint["max_seq_len"],
    norm_type=checkpoint.get("norm_type","rmsnorm"),
    feedforward_type=checkpoint.get("feedforward_type","swiglu"),
).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

prompt="once upon a time"

prompt_ids = tokenizer.encode(prompt)
unknown_id = tokenizer.token_to_id[tokenizer.unknown_token]
if unknown_id in prompt_ids:
    print("警告：提示词中包含词表无法识别的内容")
input_idx = torch.tensor(
    [prompt_ids],
    dtype=torch.long,
    device=device
)
print("提示词：", prompt)
print("Token ID：", prompt_ids)
print("输入形状：", input_idx.shape)
with torch.no_grad():
    generated_ids = model.generate_with_cache(
        input_idx,
        max_new_tokens=80,
        temperature=0.8,
        top_k=3
    )

    generated_text = tokenizer.decode(
    generated_ids[0].tolist()
)
print("\n生成结果：")
print(generated_text)
