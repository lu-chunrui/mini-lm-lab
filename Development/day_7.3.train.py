from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:",device)

text = (
    "人工智能正在改变世界。"
    "大语言模型可以根据上下文预测下一个字符。"
    "学习机器学习需要多写代码多做实验。\n"
) * 100

chars=sorted(list(set(text)))
vocab_size=len(chars)
stoi={ch:i for i,ch in enumerate(chars)}
itos={i:ch for i,ch in enumerate(chars)}
def encode(s):
    return [stoi[ch] for ch in s]
def decode(ids):
    return "".join([itos[i] for i in ids])
data=torch.tensor(encode(text))
split_index=int(0.8*len(data))
train_data, val_data = data[:split_index], data[split_index:]

batch_size=32
block_size=8
embedding_dim=128
num_heads=4
num_layers=2
feedforward_dim=128
lr=0.001
max_steps=1500
eval_interval = 100
eval_iterations = 30

def get_batch(data_source):
    start=torch.randint(0,len(data_source)-block_size,(batch_size,))
    x=torch.stack([data_source[i:i+block_size] for i in start])
    y=torch.stack([data_source[i+1:i+block_size+1] for i in start])
    return x.to(device),y.to(device)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        mean_square=x.pow(2).mean(dim=-1,keepdim=True)
        inverse_rms=torch.rsqrt(mean_square+self.eps)
        return inverse_rms*x*self.weight
def precompute_rope(head_size,block_size,base=10000.0):
    if head_size%2!=0:
        raise ValueError("head_size must be even")
    dimension_indices=torch.arange(0,head_size,2)
    inverse_freqs=1.0/(base**(dimension_indices/head_size))
    positions=torch.arange(block_size)
    angles=torch.outer(positions,inverse_freqs)
    cos=torch.cos(angles)
    sin=torch.sin(angles)
    return cos,sin
def apply_rope(x,cos,sin):
    B,T,head_size=x.shape
    cos=cos[:T].unsqueeze(0)
    sin=sin[:T].unsqueeze(0)
    x_even=x[...,::2]
    x_odd=x[...,1::2]
    rotated_even=(x_even*cos-x_odd*sin)
    rotated_odd=(x_even*sin+x_odd*cos)
    rotated=torch.stack([rotated_even,rotated_odd],dim=-1)
    rotated=rotated.flatten(start_dim=-2)
    return rotated
class CausalselfAttention(nn.Module):
    def __init__(self,embedding_dim, head_size, block_size):
        super().__init__()
        self.head_size = head_size
        self.key=nn.Linear(embedding_dim,head_size,bias=False)
        self.query=nn.Linear(embedding_dim,head_size,bias=False)
        self.value=nn.Linear(embedding_dim,head_size,bias=False)
        cos, sin = precompute_rope(head_size=head_size, block_size=block_size)
        self.register_buffer("cos",cos)
        self.register_buffer("sin",sin)
        causal_mask=torch.tril(torch.ones(block_size,block_size,dtype=torch.bool))
        self.register_buffer("causal_mask",causal_mask)
    def forward(self,x):
        B,T,C=x.shape
        k=self.key(x)
        q=self.query(x)
        v=self.value(x)
        k=apply_rope(k,self.cos,self.sin)
        q=apply_rope(q,self.cos,self.sin)
        scores=q@k.transpose(-2,-1)
        scores = scores / (self.head_size ** 0.5)
        scores = scores.masked_fill(~self.causal_mask[:T,:T], float('-inf'))
        weights=F.softmax(scores,dim=-1)
        output=weights@v
        return output
class MultiHeadAttention(nn.Module):
    def __init__(self,embedding_dim, num_heads, block_size):
        super().__init__()
        if embedding_dim%num_heads!=0:
           raise ValueError("embedding_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_size = embedding_dim//num_heads
        self.heads=nn.ModuleList([CausalselfAttention(embedding_dim=embedding_dim,head_size=self.head_size,block_size=block_size) for _ in range(num_heads)])
        self.out=nn.Linear(embedding_dim,embedding_dim,bias=False)
    def forward(self,x):
        head_outputs=[head(x) for head in self.heads]
        output=torch.cat(head_outputs,dim=-1)
        output=self.out(output)
        return output
class SwiGLUFeedForward(nn.Module):
    def __init__(self,embedding_dim,feedforward_dim):
        super().__init__()
        self.gate_linear=nn.Linear(embedding_dim,feedforward_dim,bias=False)
        self.value_linear=nn.Linear(embedding_dim,feedforward_dim,bias=False)
        self.output_linear=nn.Linear(feedforward_dim,embedding_dim,bias=False)
    def forward(self,x):
        gate=F.silu(self.gate_linear(x))
        value=self.value_linear(x)
        output=gate*value
        output=self.output_linear(output)
        return output
class TransformerBlock(nn.Module):
    def __init__(self,embedding_dim, num_heads, block_size, feedforward_dim):
        super().__init__()
        self.attention=MultiHeadAttention(embedding_dim=embedding_dim, num_heads=num_heads, block_size=block_size)
        self.feed_forward=SwiGLUFeedForward(embedding_dim, feedforward_dim)
        self.ln1=RMSNorm(embedding_dim)
        self.ln2=RMSNorm(embedding_dim)
    def forward(self,x):
        x=x+self.attention(self.ln1(x))
        x=x+self.feed_forward(self.ln2(x))
        return x
class TransformerLanguageModel(nn.Module):
    def __init__(self,embedding_dim, num_heads, block_size, feedforward_dim, vocab_size):
        super().__init__()
        self.block_size=block_size
        self.token_embedding=nn.Embedding(vocab_size,embedding_dim)
        self.blocks=nn.Sequential(*[TransformerBlock(embedding_dim, num_heads, block_size, feedforward_dim) for _ in range(num_layers)])
        self.final_norm=RMSNorm(embedding_dim)
        self.out=nn.Linear(embedding_dim,vocab_size,bias=False)
    def forward(self,x,target=None):
        B,T=x.shape
        if T>block_size:
            raise ValueError("T must be less than or equal to block_size")
        x=self.token_embedding(x)
        x=self.blocks(x)
        x=self.final_norm(x)
        logits=self.out(x)
        loss=None
        if target is not None:
            loss=F.cross_entropy(logits.reshape(B*T,logits.shape[-1]),target.reshape(B*T))
        return logits,loss
    @torch.no_grad()
    def generate(self,idx,max_new_tokens=100,temperature=1.0,top_k=5):
        if temperature<=0.0:
            raise ValueError("temperature must be greater than 0")
        for _ in range(max_new_tokens):
            idx_context=idx[:,-block_size:]
            logits,loss=self(idx_context)
            logits=logits[:,-1,:]
            logits=logits/temperature
            if top_k is not None:
                k=min(top_k,logits.shape[-1])
                top_values,top_indices=logits.topk(k,dim=-1)
                threshold=(top_values[:,-1].unsqueeze(-1))
                logits=logits.masked_fill(logits<threshold,-float("inf"))
            probabilities=F.softmax(logits,dim=-1)
            next_token=torch.multinomial(probabilities,num_samples=1)
            idx=torch.cat([idx,next_token],dim=-1)
        return idx
@torch.no_grad()
def estimate_loss(model):
    model.eval()
    results={}
    data_sources={"train":train_data,"val":val_data}
    for name,data in data_sources.items():
        losses=torch.zeros(eval_iterations)
        for iteration in range(eval_iterations):
           x,y=get_batch(data)
           logits,loss=model(x,y)
           losses[iteration]=loss.item()
        results[name]=losses.mean().item()
    model.train()
    return results
model=TransformerLanguageModel(embedding_dim=embedding_dim, num_heads=num_heads, block_size=block_size, feedforward_dim=feedforward_dim, vocab_size=vocab_size).to(device)
parameter_count=sum(p.numel() for p in model.parameters())
print("模型参数数量:",parameter_count)

x, y = get_batch(train_data)
logits, initial_loss = model(x,y)
print("输入形状：", x.shape)
print("标签形状：", y.shape)
print("logits形状：", logits.shape)
print("初始loss：", initial_loss.item())

optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
checkpoint_directory = Path("checkpoints")
checkpoint_directory.mkdir(parents=True,exist_ok=True)
checkpoint_path = (checkpoint_directory/ "best_rope_rmsnorm_swiglu_model.pth"
)
best_loss=float("inf")

model.train()
for epoch in range(max_steps):
    x,y=get_batch(train_data)
    logits,loss=model(x,y)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    optimizer.step()
    if epoch%eval_interval==0:
        results=estimate_loss(model)
        train_loss=results["train"]
        val_loss=results["val"]
        print(f"epoch:{epoch},train_loss:{train_loss:.4f},val_loss:{val_loss:.4f}")
        if val_loss<best_loss:
            best_loss=val_loss
            chackpoint_data={
                "step":epoch,
                "val_loss":val_loss,
                "model_state_dict":model.state_dict(),
                "optimizer_state_dict":optimizer.state_dict(),
            }
            torch.save(chackpoint_data,checkpoint_path)
            print(f"保存模型到{checkpoint_path}")
        else:
            print("未改进最佳模型")
checkpoint_data=torch.load(checkpoint_path)
model.load_state_dict(checkpoint_data["model_state_dict"])
optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
print("加载最佳模型完成")
print("保存时的step:",checkpoint_data["step"])
print("最佳模型的loss:",checkpoint_data["val_loss"])

model.eval()
start=torch.tensor([[stoi["\n"]]],device=device)
generated=model.generate(start,max_new_tokens=100,temperature=1.0,top_k=5)
result=decode(generated[0].tolist())
print("生成结果：")
print(result)