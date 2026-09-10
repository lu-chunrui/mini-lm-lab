import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

torch.manual_seed(42)
train_text = (
    "人工智能正在改变世界。"
    "大语言模型可以根据上下文预测下一个字符。"
    "学习机器学习需要多写代码多做实验。\n"
) * 100
val_text = ("人工智能需要学习上下文。\n"
    "大语言模型正在学习预测字符。\n")*20
all_text = train_text + val_text
chars = sorted(list(set(all_text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
def encode(s):
    return [stoi[c] for c in s]
def decode(idx):
    return "".join(itos[i] for i in idx)
train_data=torch.tensor(encode(train_text), dtype=torch.long)
val_data=torch.tensor(encode(val_text), dtype=torch.long)
batch_size = 32
block_size = 8
embedding_dim = 32
num_heads = 4
learning_rate = 0.001
max_steps = 1500
num_layers=2
def get_batch(data_source):
    positions = torch.randint(0, len(data_source) - block_size, (batch_size,))
    x = torch.stack([data_source[p : p + block_size] for p in positions])
    y = torch.stack([data_source[p + 1 : p + block_size + 1] for p in positions])
    return x, y
class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_dim, head_size, block_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("mask", mask)
    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = q @ k.transpose(-2, -1) / (self.head_size**0.5)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        outputs = weights @ v
        return outputs
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, block_size):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim必须是num_heads的整数倍数")
        self.num_heads = num_heads
        self.head_size = embedding_dim // num_heads
        self.heads = nn.ModuleList(
            [
                CausalSelfAttention(
                    embedding_dim, head_size=self.head_size, block_size=block_size
                )
                for _ in range(num_heads)
            ]
        )
        self.projection = nn.Linear(embedding_dim, embedding_dim)
    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]
        combined = torch.cat(head_outputs, dim=-1)
        output = self.projection(combined)
        return output
class FeedForward(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )
    def forward(self, x):
        return self.net(x)
class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, block_size):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.attention = MultiHeadAttention(embedding_dim, num_heads, block_size)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim)
    def forward(self, x):
        x = x + self.attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x
class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, block_size):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(embedding_dim=embedding_dim, num_heads=num_heads, block_size=block_size) for _ in range(num_layers)]
        )
        self.final_layer_norm = nn.LayerNorm(embedding_dim)
        self.output_linear = nn.Linear(embedding_dim, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"输入序列长度 {T} 超过了模型的最大块大小 {self.block_size}")
        token_embeddings = self.token_embedding(idx)
        positions = torch.arange(T, device=idx.device)
        position_embeddings = self.position_embedding(positions)
        x = token_embeddings + position_embeddings
        x = self.transformer_blocks(x)
        x=self.final_layer_norm(x)
        logits = self.output_linear(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(B * T, -1), targets.reshape(B * T))
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_context = idx[:, -self.block_size :]
            logits, _ = self(idx_context)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token_idx), dim=1)
        return idx
model = TransformerLanguageModel(
    vocab_size=vocab_size, embedding_dim=embedding_dim, num_heads=num_heads, num_layers=num_layers, block_size=block_size
)
print("词表大小：", vocab_size)
print("注意力头数：", num_heads)
print( "每个头的维度：",embedding_dim // num_heads)
print("前馈网络隐藏维度：",4 * embedding_dim)
x,y = get_batch(train_data)
with torch.no_grad():
    logits, loss = model(x, y)
print("输入形状：", x.shape)
print("标签形状：", y.shape)
print("logits形状：", logits.shape)
print("初始loss：", loss.item())
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
checkpoint_dir=Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)
checkpoint_path=checkpoint_dir/"best_model.pth"
best_val_loss = float("inf")
config = {
    "vocab_size": vocab_size,
    "embedding_dim": embedding_dim,
    "num_heads": num_heads,
    "block_size": block_size,
    "num_layers": num_layers
}
@torch.no_grad()
def estimate_loss(model,data_source,eval_batches=50):
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(eval_batches):
        x, y = get_batch(data_source)
        _, loss = model(x, y)
        losses.append(loss.item())
    if was_training:
        model.train()
    return sum(losses) / len(losses)
model.train()
for step in range(max_steps):
    x, y = get_batch(train_data)
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        train_loss = estimate_loss(model, train_data)
        val_loss = estimate_loss(model, val_data)
        print(f"Step {step}, Loss: {loss.item()}. Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"config": config, "model_state_dict": model.state_dict()}, checkpoint_path)
            print(f"保存模型到 {checkpoint_path}")
checkpoint =torch.load(checkpoint_path,map_location="cpu")
config = checkpoint["config"]
loaded_model =TransformerLanguageModel(**config)
loaded_model.load_state_dict(checkpoint["model_state_dict"])
loaded_model.eval()
print("加载模型完成")
print("模型参数数量：", sum(p.numel() for p in loaded_model.parameters()))
start_idx = torch.tensor([[stoi["\n"]]], dtype=torch.long)
generated = loaded_model.generate(start_idx, max_new_tokens=100)
print("加载模型后的生成结果：",decode(generated[0].tolist()))
