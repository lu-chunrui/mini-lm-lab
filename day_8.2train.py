from pathlib import Path
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备", device)

text=("人工智能正在改变世界。"
    "大语言模型可以根据上下文预测下一个字符。"
    "学习机器学习需要多写代码多做实验。\n")*100
chars=sorted(list(set(text)))
vocab_size=len(chars)
stoi={ch:i for i,ch in enumerate(chars)}
itos={i:ch for i,ch in enumerate(chars)}
def encode(s):
    return [stoi[ch] for ch in s]
def decode(ids):
    return "".join([itos[i] for i in ids])
data=torch.tensor(encode(text), dtype=torch.long)
split_index=int(0.8*len(data))
train_data=data[:split_index]
val_data=data[split_index:]

batch_size=32
max_seq_len=256
embedding_dim=128
block_size=8
num_heads=4
num_layers=2
feedforward_dim=256
learning_rate=0.001
max_steps=1500
eval_interval=100
eval_iterations=30
generation_length=100

def get_batch(data_source):
    positions=torch.randint(0, len(data_source)-block_size, (batch_size,))
    x=torch.stack([data_source[i:i+block_size] for i in positions])
    y=torch.stack([data_source[i+1:i+block_size+1] for i in positions])
    return x.to(device), y.to(device)
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(dim))
    def forward(self, x):
        mean_square=x.pow(2).mean(dim=-1, keepdim=True)
        inverse_rms=torch.rsqrt(mean_square+self.eps)
        return x*inverse_rms*self.weight
def precompute_rope(head_size,max_length,base=10000.0):
    if head_size%2!=0:
        raise ValueError("head_size must be even")
    dimension_indices=torch.arange(0,head_size,2)
    inverse_freq=1.0/(base**(dimension_indices/head_size))
    positions=torch.arange(max_length, dtype=torch.float32)
    angles=torch.outer(positions, inverse_freq)
    cos_angles=torch.cos(angles)
    sin_angles=torch.sin(angles)
    return cos_angles, sin_angles
def apply_rope(x, cos_angles, sin_angles, position_offset=0):
    B,T,head_size=x.shape
    position_end=position_offset+T
    if position_end>cos_angles.shape[0]:
        raise ValueError("position_end must be less than or equal to cos_angles.shape[0]")
    selected_cos_angles=cos_angles[position_offset:position_end]
    selected_sin_angles=sin_angles[position_offset:position_end]
    selected_sin_angles=selected_sin_angles.unsqueeze(0)
    selected_cos_angles=selected_cos_angles.unsqueeze(0)
    x_even=x[...,::2]
    x_odd=x[...,1::2]
    rotated_even=x_even*selected_cos_angles-x_odd*selected_sin_angles
    rotated_odd=x_even*selected_sin_angles+x_odd*selected_cos_angles
    rotated=torch.stack([rotated_even, rotated_odd], dim=-1)
    return rotated.flatten(start_dim=-2)
class CausalSelfAttention(nn.Module):
    def __init__(self,embedding_dim,head_size,max_seq_len):
        super().__init__()
        self.head_size=head_size
        self.k=nn.Linear(embedding_dim, head_size, bias=False)
        self.q=nn.Linear(embedding_dim, head_size, bias=False)
        self.v=nn.Linear(embedding_dim, head_size, bias=False)
        cos,sin=precompute_rope(head_size=head_size, max_length=max_seq_len,)
        self.register_buffer("cos_angles", cos)
        self.register_buffer("sin_angles", sin)
    def forward(self,x,past_kv=None,use_cache=False):
        B,T,C=x.shape
        if past_kv is None:
            past_k = None
            past_v = None
            past_length = 0
        else:
            past_k, past_v = past_kv
            past_length = past_k.shape[1]
        q=self.q(x)
        new_k=self.k(x)
        new_v=self.v(x)
        q=apply_rope(q, self.cos_angles, self.sin_angles, position_offset=past_length)
        new_k=apply_rope(new_k, self.cos_angles, self.sin_angles, position_offset=past_length)
        if past_k is None:
            k=new_k
            v=new_v
        else:
            k=torch.cat([past_k, new_k], dim=1)
            v=torch.cat([past_v, new_v], dim=1)
        total_length=k.shape[1]

        scores=q@k.transpose(-2,-1)
        scores=(scores/(self.head_size**0.5))
        query_positions=(past_length+torch.arange(T, dtype=torch.long, device=x.device))
        key_positions=torch.arange(total_length, dtype=torch.long, device=x.device)
        causal_mask=(key_positions.unsqueeze(0)<=query_positions.unsqueeze(1))
        scores=scores.masked_fill(causal_mask==False, float("-inf"))
        attention_weights=F.softmax(scores, dim=-1)
        output=attention_weights@v
        if use_cache:
            new_cache=(k, v)
        else:
            new_cache=None
        return output, new_cache
class MultiHeadAttention(nn.Module):
    def __init__(self,embedding_dim,num_heads,max_seq_len):
        super().__init__()
        self.num_heads=num_heads
        if embedding_dim%num_heads!=0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.num_size=(embedding_dim//num_heads)
        self.heads=nn.ModuleList(
            [CausalSelfAttention(embedding_dim=embedding_dim, head_size=self.num_size, max_seq_len=max_seq_len) for _ in range(num_heads)]
        )
        self.output_proj=nn.Linear(embedding_dim, embedding_dim, bias=False)
    def forward(self,x,past_kv=None,use_cache=False):
        if past_kv is None:
            past_kv=[None for _ in range(self.num_heads)]
        head_outputs=[]
        new_head_cache=[]
        for head, head_past_kv in zip(self.heads, past_kv):
            output, new_cache=head(x, past_kv=head_past_kv, use_cache=use_cache)
            head_outputs.append(output)
            if use_cache:
               new_head_cache.append(new_cache)
        combined=torch.cat(head_outputs, dim=-1)
        output=self.output_proj(combined)
        if use_cache:
            return(output,new_head_cache)
        return output, None
class SwiGLUFeedForward(nn.Module):
    def __init__(self,embedding_dim,feedforward_dim):
        super().__init__()
        self.embedding_dim=embedding_dim
        self.feedforward_dim=feedforward_dim
        self.gate_linear=nn.Linear(embedding_dim, feedforward_dim, bias=False)
        self.value_linear=nn.Linear(embedding_dim, feedforward_dim, bias=False)
        self.output_proj=nn.Linear(feedforward_dim, embedding_dim, bias=False)
    def forward(self,x):
        gate=F.silu(self.gate_linear(x))
        value=self.value_linear(x)
        output=gate*value
        output=self.output_proj(output)
        return output
class TransformerBlock(nn.Module):
    def __init__(self,embedding_dim,num_heads,feedforward_dim,max_seq_len):
        super().__init__()
        self.attention=MultiHeadAttention(embedding_dim=embedding_dim, num_heads=num_heads, max_seq_len=max_seq_len)
        self.feedforward=SwiGLUFeedForward(embedding_dim=embedding_dim, feedforward_dim=feedforward_dim)
        self.norm1=RMSNorm(embedding_dim)
        self.norm2=RMSNorm(embedding_dim)
    def forward(self,x,past_kv=None,use_cache=False):
        attn_output, attn_cache=self.attention(self.norm1(x), past_kv, use_cache)
        x=x+attn_output
        x=x+self.feedforward(self.norm2(x))
        return x, attn_cache
class TransformerLanguageModel(nn.Module):
    def __init__(self,embedding_dim,num_heads,num_layers,feedforward_dim,max_seq_len):
        super().__init__()
        self.embedding_dim=embedding_dim
        self.num_heads=num_heads
        self.feedforward_dim=feedforward_dim
        self.max_seq_len=max_seq_len
        self.token_embedding=nn.Embedding(vocab_size, embedding_dim)
        self.blocks=nn.ModuleList(
            [TransformerBlock(embedding_dim=embedding_dim, num_heads=num_heads, feedforward_dim=feedforward_dim, max_seq_len=max_seq_len) for _ in range(num_layers)]
        )
        self.final_norm=RMSNorm(embedding_dim)
        self.output_linear=nn.Linear(embedding_dim, vocab_size, bias=False)
    def forward(self,x,target=None,past_kv=None,use_cache=False):
        B,T=x.shape
        if past_kv is None:
            past_length=0
            past_kv=[None for _ in range(len(self.blocks))]
        else:
            first_k=past_kv[0][0][0]
            past_length = first_k.shape[1]
        if(past_length+T>self.max_seq_len):
            raise ValueError("past_length+T must be less than or equal to max_seq_len")
        x=self.token_embedding(x)
        new_past_kv=[]
        for block,layer_kv in zip(self.blocks, past_kv):
            x, new_cache=block(x, past_kv=layer_kv, use_cache=use_cache)
            if use_cache:
               new_past_kv.append(new_cache)
        x=self.final_norm(x)
        logits=self.output_linear(x)
        loss=None
        if target is not None:
            loss=F.cross_entropy(logits.reshape(B * T, -1),
                target.reshape(B * T), reduction="mean")
        if use_cache:
            return logits,loss, new_past_kv
        return logits, loss
    def sample_next_token(self,logits,temperature=1.0,top_k=5):
        if temperature<=0.0:
            raise ValueError("temperature must be greater than or equal to 0.0")
        logits=logits/temperature
        if top_k is not None:
            k=min(top_k, logits.shape[-1])
            top_values,_=torch.topk(logits, k, dim=-1)
            threshold=(top_values[:,-1].unsqueeze(-1))
            logits=logits.masked_fill(logits<threshold, float("-inf"))
        probs=F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    @torch.no_grad()
    def generate_without_cache(self,x,max_new_tokens=100,temperature=1.0,top_k=5):
        if (x.shape[1]+max_new_tokens>self.max_seq_len):
            raise ValueError("x.shape[1]+max_new_tokens must be less than or equal to max_seq_len")
        for _ in range(max_new_tokens):
            logits, loss=self(x)
            last_logits=logits[:, -1, :]
            next_token=self.sample_next_token(last_logits, temperature, top_k)
            x=torch.cat([x, next_token], dim=1)
        return x
    @torch.no_grad()
    def generate_with_cache(self,x,max_new_tokens=100,temperature=1.0,top_k=5):
        if (x.shape[1]+max_new_tokens>self.max_seq_len):
            raise ValueError("x.shape[0]+max_new_tokens must be less than or equal to max_seq_len")
        logits,loss,cache=self(x, use_cache=True)
        for _ in range(max_new_tokens):
            last_logits=logits[:, -1, :]
            next_token=self.sample_next_token(last_logits, temperature, top_k)
            x=torch.cat([x, next_token], dim=1)
            if _ == max_new_tokens - 1:
                break
            logits,loss,cache=self(next_token, past_kv=cache, use_cache=True)
        return x
@torch.no_grad()
def estimate_loss(model):
    model.eval()
    result={}
    data_source={"train": train_data,
        "val": val_data}
    for name,data_source in data_source.items():
        losses=torch.zeros(eval_iterations)
        for iteration in range(eval_iterations):
            x,y=get_batch(data_source)
            logits,loss=model(x, y)
            losses[iteration]=loss.item()
        result[name]=losses.mean().item()
    model.train()
    return result

model=TransformerLanguageModel(embedding_dim=embedding_dim, num_heads=num_heads, num_layers=num_layers, feedforward_dim=feedforward_dim, max_seq_len=max_seq_len).to(device)
parameter_count = sum(parameter.numel() for parameter in model.parameters())
print("模型参数数量：", parameter_count)

x, y = get_batch(train_data)
logits, initial_loss = model(x, y)
print("输入形状：", x.shape)
print("标签形状：", y.shape)
print("logits形状：", logits.shape)
print("初始loss：", initial_loss.item())

optimizer=torch.optim.AdamW(model.parameters(), lr=learning_rate)
checkpoint_directory=Path("checkpoints")
checkpoint_directory.mkdir(parents=True,exist_ok=True)
checkpoint_path = (checkpoint_directory/ "best_kv_cache_model.pth")
best_val_loss=float("inf")
model.train()
for step in range(max_steps):
    x,y=get_batch(train_data)
    logits,loss=model(x, y)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
    optimizer.step()
    if step % eval_interval == 0 or step == max_steps - 1:
        results=estimate_loss(model)
        train_loss = results["train"]
        val_loss = results["val"]
        print(f"step:{step}, train_loss:{train_loss:.4f}, val_loss:{val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss=val_loss
            checkpoint_data = {"step": step,"val_loss": val_loss,"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict()}
            torch.save( checkpoint_data,checkpoint_path)
            print("保存最佳模型到：",checkpoint_path )
checkpoint_data=torch.load(checkpoint_path,map_location=device)
model.load_state_dict(checkpoint_data["model_state_dict"])
optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
print("\n加载最佳模型完成")
print("保存时的step：",checkpoint_data["step"])
print("最佳验证loss：",checkpoint_data["val_loss"])

model.eval()
start=torch.tensor([[stoi["\n"]]],dtype=torch.long,device=device)
normal_start_time = time.perf_counter()
normal_generated = (model.generate_without_cache( start.clone(),max_new_tokens=generation_length,temperature=1.0,top_k=1))
normal_time = (time.perf_counter() - normal_start_time)
cache_start_time = time.perf_counter()
cached_generated = ( model.generate_with_cache(start.clone(),max_new_tokens=generation_length,temperature=1.0,top_k=1))
cache_time = (time.perf_counter()- cache_start_time)

same_result = torch.equal(normal_generated,cached_generated)
print("\n===== KV Cache测试 =====")
print( "两种生成结果是否相同：",same_result)
print(f"普通生成时间：{normal_time:.4f}秒")
print("普通生成速度:",f"{generation_length / normal_time:.2f} ""tokens/s")
print( f"KV Cache生成时间：{cache_time:.4f}秒")
print("KV Cache生成速度:",f"{generation_length / cache_time:.2f} ""tokens/s")
print("速度倍率:",f"{normal_time / cache_time:.2f}x")
result = decode(cached_generated[0].tolist())
print("\nKV Cache生成结果：")
print(result)
