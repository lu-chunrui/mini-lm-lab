from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
train_text = (
    "人工智能正在改变世界。"
    "大语言模型可以根据上下文预测下一个字符。"
    "学习机器学习需要多写代码多做实验。\n"
) * 100
chars = sorted(list(set(train_text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
def encode(s):
    return [stoi[c] for c in s]
def decode(idx):
    return "".join(itos[i] for i in idx)
data=torch.tensor(encode(train_text), dtype=torch.long)
split_idx=int(0.8*len(data))
train_data=data[:split_idx]
val_data=data[split_idx:]
batch_size = 32
block_size = 16
embedding_dim = 32
num_heads = 4
num_layers=2
feed_forward_dim = 128
learning_rate = 0.001
max_steps = 1500
eval_interval = 100
eval_iterations = 30

def get_batch(data_source):
    positions = torch.randint(0, len(data_source) - block_size, (batch_size,))
    x = torch.stack([data_source[p : p + block_size] for p in positions])
    y = torch.stack([data_source[p + 1 : p + block_size + 1] for p in positions])
    return x, y
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        mean_square=x.pow(2).mean(dim=-1,keepdim=True)
        inverse_rms=1.0/(mean_square+self.eps).sqrt()
        return x*inverse_rms*self.weight
class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_dim,num_heads,head_size, block_size):
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
        self.head_size = embedding_dim // num_heads
        self.heads=nn.ModuleList([
            CausalSelfAttention(embedding_dim=embedding_dim, num_heads=num_heads, head_size=self.head_size, block_size=block_size)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
    def forward(self, x):
        B, T, C = x.shape
        outputs = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.proj(outputs)
class SwiGLUFeedForward(nn.Module):
    def __init__(self, embedding_dim, feed_forward_dim):
        super().__init__()
        self.gate_linear = nn.Linear(embedding_dim, feed_forward_dim, bias=False)
        self.value_linear = nn.Linear(embedding_dim, feed_forward_dim, bias=False)
        self.out_linear = nn.Linear(feed_forward_dim, embedding_dim, bias=False)
    def forward(self, x):
        gate=F.silu(self.gate_linear(x))
        value=self.value_linear(x)
        outputs=self.out_linear(gate*value)
        return outputs
class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, block_size, feed_forward_dim):
        super().__init__()
        self.norm1=RMSNorm(embedding_dim)
        self.norm2=RMSNorm(embedding_dim)
        self.attention=MultiHeadAttention(embedding_dim=embedding_dim, num_heads=num_heads, block_size=block_size)
        self.feed_forward=SwiGLUFeedForward(embedding_dim=embedding_dim, feed_forward_dim=feed_forward_dim)
    def forward(self, x):
        x=x+self.attention(self.norm1(x))
        x=x+self.feed_forward(self.norm2(x))
        return x
class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_dim, embedding_dim, num_heads, block_size, feed_forward_dim, num_layers):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_dim, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.blocks=nn.Sequential(
            *[TransformerBlock(embedding_dim=embedding_dim, num_heads=num_heads, block_size=block_size, feed_forward_dim=feed_forward_dim) for _ in range(num_layers)]
        )
        self.final_norm=RMSNorm(embedding_dim)
        self.output_embedding = nn.Linear(embedding_dim, vocab_dim)
    def forward(self, x, targets=None):
        B, T = x.shape
        if T > self.block_size:
            raise ValueError(f"输入序列长度{T}大于模型最大长度{self.block_size}")
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(torch.arange(T, device=x.device))
        x = token_embeddings + position_embeddings
        x=self.blocks(x)
        x=self.final_norm(x)
        logits=self.output_embedding(x)
        loss=None
        if targets is not None:
            loss=F.cross_entropy(logits.reshape(B*T,logits.shape[-1]), targets.reshape(B*T), reduction="mean")
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100,temperature=1.0, top_k=5):
        if temperature<=0.0:
            raise ValueError("temperature必须大于0.0")
        for _ in range(max_new_tokens):
            idx_context=idx[:,-self.block_size:]
            logits, loss=self(idx_context)
            logits=logits[:, -1, :]
            logits=logits/temperature
            if top_k is not None:
                k=min(top_k, logits.shape[-1])
                top_values, top_indices=torch.topk(logits, k, dim=-1, sorted=True)
                minimum_value=(top_values[:, -1].unsqueeze(-1))
                logits=logits.masked_fill(logits< minimum_value, float("-inf"))
            probabilities=F.softmax(logits, dim=-1)
            next_idx=torch.multinomial(probabilities, num_samples=1)
            idx=torch.cat([idx, next_idx], dim=-1)
        return idx
@torch.no_grad()
def estimate_loss(model):
    model.eval()
    results={}
    data_source={"train": train_data, "val": val_data}
    for name,data in data_source.items():
        losses=torch.zeros(eval_iterations)
        for i in range(eval_iterations):
            x, y=get_batch(data)
            logits, loss=model(x, targets=y)
            losses[i]=loss.item()
        results[name]=losses.mean().item()
    model.train()
    return results
model=TransformerLanguageModel(vocab_dim=vocab_size, embedding_dim=embedding_dim, num_heads=num_heads, block_size=block_size, feed_forward_dim=feed_forward_dim, num_layers=num_layers)
paremeter_count=sum(p.numel() for p in model.parameters())
print("模型参数数量：", paremeter_count)
x, y=get_batch(train_data)
logits, loss=model(x, targets=y)
print("logits.shape:", logits.shape)
print("loss:", loss.item())
optimizer=torch.optim.AdamW(model.parameters(), lr=learning_rate)
checkpoint_directory=Path("checkpoints")
checkpoint_directory.mkdir(parents=True, exist_ok=True)
checkpoint_path=checkpoint_directory/"best_rmsnorm_swiglu_model.pth"
best_val_loss = float("inf")

model.train()
for step in range(max_steps):
    x, y=get_batch(train_data)
    logits, loss=model(x, targets=y)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if step % eval_interval == 0:
        results=estimate_loss(model)
        train_loss=results["train"]
        val_loss=results["val"]
        print(f"step: {step}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_data={
                "step": step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(checkpoint_data, checkpoint_path)
            print(f"保存最佳模型到：{checkpoint_path}")
        else:
            print("val_loss未改进，不保存检查点")
checkpoint_data=torch.load(checkpoint_path)
model.load_state_dict(checkpoint_data["model_state_dict"])
optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
print("\n加载最佳模型完成")
print("保存时的step：",checkpoint_data["step"])
print("最佳验证loss：",checkpoint_data["val_loss"])

model.eval()
start=torch.tensor([[stoi["\n"]]],dtype=torch.long)
generated=model.generate(start, max_new_tokens=100,temperature=1.0,top_k=5)
result=decode(generated[0].tolist())
print("生成结果：",result)
