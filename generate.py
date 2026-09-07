from pathlib import Path
import torch
from config import device
from tokenizer import CharTokenizer
from model import TransformerLanguageModel

project_dir = Path(__file__).resolve().parent
checkpoint_path = project_dir / "checkpoints" / "best_kv_cache_model.pth"

if not checkpoint_path.exists():raise FileNotFoundError(f"没有找到模型文件：{checkpoint_path}\n"
    "请先运行python train.py")
checkpoint=torch.load(checkpoint_path)
print("加载模型：", checkpoint_path)
print("训练step：", checkpoint["step"])
print("验证loss：", checkpoint["val_loss"])

tokenizer = CharTokenizer(chars=checkpoint["chars"])
print("词表大小：", tokenizer.vocab_size)

model = TransformerLanguageModel(
    vocab_size=checkpoint["vocab_size"],
    embedding_dim=checkpoint["embedding_dim"],
    num_heads=checkpoint["num_heads"],
    num_layers=checkpoint["num_layers"],
    feedforward_dim=checkpoint[
        "feedforward_dim"
    ],
    max_seq_len=checkpoint["max_seq_len"]
).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

prompt="人工智能"
unknown_chars=[char for char in prompt if char not in tokenizer.stoi]
if unknown_chars:
    print(f"警告：输入中包含未知字符：{', '.join(unknown_chars)}")
input_idx=torch.tensor(tokenizer.encode(prompt),dtype=torch.long).to(device)
print("提示词：", prompt)
print("输入形状：", input_idx.shape)

with torch.no_grad():
    generated_ids = model.generate_with_cache(
        input_idx,
        max_new_tokens=100,
        temperature=0.8,
        top_k=5
    )

    generated_text = tokenizer.decode(
    generated_ids[0].tolist()
)

print("\n生成结果：")
print(generated_text)
