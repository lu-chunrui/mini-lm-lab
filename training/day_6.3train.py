from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备：", device)

text=("人工智能正在改变世界。"
    "大语言模型可以根据上下文预测下一个字符。"
    "学习机器学习需要多写代码多做实验。\n")*100
chars=sorted(list(set(text)))
vocab_size=len(chars)
stoi={ch:i for i,ch in enumerate(chars)}
itos={i:ch for ch,i in stoi.items()}
def encode(s):
    return [stoi[c] for c in s]
def decode(idx):
    return "".join(itos[i] for i in idx)
data=torch.tensor(encode(text), dtype=torch.long)
split_idx=0.8*len(data)
train_data=data[:int(split_idx)]
val_data=data[int(split_idx):]
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
    def __init__(self, embedding_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embedding_dim))
    def forward(self, x):
        mean_squared = x.pow(2).mean(dim=-1, keepdim=True)
        inverse_rms=torch.rsqrt(mean_squared + self.eps)
        output = x * inverse_rms * self.weight
        return output
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
            [CausalSelfAttention(embedding_dim, self.head_size, block_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(embedding_dim, embedding_dim)
    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]
        output = torch.cat(head_outputs, dim=-1)
        output = self.proj(output)
        return output
class FeedForward(nn.Module):
    def __init__(self, embedding_dim, feed_forward_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, embedding_dim)
        )
    def forward(self, x):
        return self.net(x)
class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, block_size, feed_forward_dim):
        super().__init__()
        self.attention = MultiHeadAttention(embedding_dim=embedding_dim, num_heads=num_heads, block_size=block_size)
        self.feed_forward = FeedForward(embedding_dim, feed_forward_dim)
        self.norm1 = RMSNorm(embedding_dim)
        self.norm2 = RMSNorm(embedding_dim)
    def forward(self, x):
        x =x + self.attention(self.norm1(x))
        x =x + self.feed_forward(self.norm2(x))
        return x
class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, block_size, feed_forward_dim, num_layers):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.blocks=nn.Sequential(
            *[TransformerBlock(embedding_dim=embedding_dim, num_heads=num_heads, block_size=block_size, feed_forward_dim=feed_forward_dim) for _ in range(num_layers)]
        )
        self.final_norm = RMSNorm(embedding_dim)
        self.output_linear = nn.Linear(embedding_dim, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"输入序列长度{T}大于模型最大长度{self.block_size}")
        token_embeddings = self.token_embedding(idx)
        positions=torch.arange(T, dtype=torch.long, device=idx.device)
        position_embeddings = self.position_embedding(positions)
        x = token_embeddings + position_embeddings
        x=self.blocks(x)
        x=self.final_norm(x)
        logits=self.output_linear(x)
        loss=None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(B * T, vocab_size),
        targets.reshape(B * T), reduction="mean")
        return logits, loss
    @torch.no_grad()
    def generate( self,idx,max_new_tokens,temperature=1, top_k=5,):
        for _ in range(max_new_tokens):
            idx_context = idx[:, -self.block_size :]
            logits, _ = self(idx_context)
            logits = logits[:, -1, :]
            logits = logits / temperature
            if top_k is not None:
                top_values, _ = torch.topk(logits,min(top_k, logits.shape[-1]),
                )
                minimum_value = top_values[:, -1].unsqueeze(-1)
                logits = logits.masked_fill( logits < minimum_value,float("-inf"))
            probabilities = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probabilities, num_samples=1,)
            idx = torch.cat( (idx, next_idx), dim=1, )
        return idx
@torch.no_grad()
def estimate_loss(model):
    model.eval()
    results=[]
    for name,data_source in [("train",train_data),("val",val_data)]:
        losses=torch.zeros(eval_iterations)
        for i in range(eval_iterations):
            x, y=get_batch(data_source)
            logits, loss=model(x, y)
            losses[i]=loss.item()
        results.append(losses.mean())
    model.train()
    return results
model=TransformerLanguageModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    block_size=block_size,
    feed_forward_dim=feed_forward_dim,
    num_layers=num_layers
).to(device)
parameter_count = sum(parameter.numel()for parameter in model.parameters())
print("词表大小：", vocab_size)
print("注意力头数：", num_heads)
print("每个头的维度：", embedding_dim // num_heads)
print("Transformer层数：", num_layers)
print("模型参数数量：", parameter_count)
x,y=get_batch(train_data)
logits, loss=model(x, y)
print(logits.shape)
print(loss.item())
optimizer=torch.optim.AdamW(model.parameters(), lr=learning_rate)
checkpoint_path=Path("checkpoints")
checkpoint_path.mkdir(exist_ok=True)
checkpoint_path /= "transformer_model.pth"
best_loss=float("inf")
model.train()
for step in range(max_steps):
    x,y=get_batch(train_data)
    logits, loss=model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        results=estimate_loss(model)
        print(f"step: {step}, train_loss: {results[0]:.4f}, val_loss: {results[1]:.4f}")
        if results[1] < best_loss:
            best_loss=results[1]
            checkpoint = {
                "step": step,
                "val_loss": best_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"保存模型到 {checkpoint_path}")
        
checkpoint=torch.load(checkpoint_path)
model.load_state_dict(checkpoint["model_state_dict"])
print("\n加载最佳模型完成")
print("保存时的step：", checkpoint["step"])
print("最佳验证loss：", checkpoint["val_loss"])
model.eval()
start=torch.tensor([[stoi["\n"]]],dtype=torch.long)
generated=model.generate(start,max_new_tokens=100,temperature=1,top_k=5)
result=(decode(generated[0].tolist()))
print("生成结果：")
print(result)





    
    