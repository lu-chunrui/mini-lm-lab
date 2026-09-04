import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
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
    return "".join([itos[i] for i in idx])
data=torch.tensor(encode(text),dtype=torch.long)
batch_size=32
block_size=8
embedding_dim=32
num_heads=4
learning_rate=0.001
max_steps=1500
def get_batch():
    positions=torch.randint(0,len(data)-block_size,(batch_size,))
    x=torch.stack([data[p:p+block_size] for p in positions])
    y=torch.stack([data[p+1:p+block_size+1] for p in positions])
    return x,y
class CausalSelfAttention(nn.Module):
    def __init__(self,embedding_dim,head_size,block_size):
        super().__init__()
        self.head_size=head_size
        self.query=nn.Linear(embedding_dim,head_size,bias=False)
        self.key=nn.Linear(embedding_dim,head_size,bias=False)
        self.value=nn.Linear(embedding_dim,head_size,bias=False)
        mask=torch.tril(torch.ones(block_size,block_size,dtype=torch.bool))
        self.register_buffer("mask",mask)
    def forward(self,x):
        B,T,C=x.shape
        q=self.query(x)
        k=self.key(x)
        v=self.value(x)
        scores=q@k.transpose(-2,-1)/(self.head_size**0.5)
        scores=scores.masked_fill(self.mask[:T,:T]==0, float('-inf'))
        weights=F.softmax(scores,dim=-1)
        outputs=weights@v
        return outputs
class MultiHeadAttention(nn.Module):
    def __init__(self,embedding_dim,num_heads,block_size):
        super().__init__()
        if embedding_dim%num_heads!=0:
            raise ValueError("embedding_dim必须是num_heads的整数倍数")
        self.num_heads=num_heads
        self.head_size=embedding_dim//num_heads
        self.heads=nn.ModuleList([CausalSelfAttention(embedding_dim,head_size=self.head_size,block_size=block_size) for _ in range(num_heads)])
        self.projection=nn.Linear(embedding_dim,embedding_dim)
    def forward(self,x):
        head_outputs=[head(x) for head in self.heads]
        combined=torch.cat(head_outputs,dim=-1)
        output=self.projection(combined)
        return output
class AttentionLanguageModel(nn.Module):
    def __init__(self,vocab_size,embedding_dim,num_heads,block_size):
        super().__init__()
        self.block_size=block_size
        self.token_embedding=nn.Embedding(vocab_size,embedding_dim)
        self.position_embedding=nn.Embedding(block_size,embedding_dim)
        self.attention=MultiHeadAttention(embedding_dim,num_heads,block_size)
        self.output_linear=nn.Linear(embedding_dim,vocab_size)
    def forward(self,idx,targets=None):
        B,T=idx.shape
        token_embeddings=self.token_embedding(idx)
        positions=torch.arange(T,device=idx.device)
        position_embeddings=self.position_embedding(positions)
        x=token_embeddings+position_embeddings
        x=self.attention(x)
        logits=self.output_linear(x)
        loss=None
        if targets is not None:
            loss=F.cross_entropy(logits.reshape(B*T,-1),targets.reshape(B*T))
        return logits,loss
    @torch.no_grad()
    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            idx_context=idx[:,-block_size:]
            logits,_=self(idx_context)
            logits=logits[:,-1,:]
            probs=F.softmax(logits,dim=-1)
            next_idx=torch.multinomial(probs,num_samples=1)
            idx=torch.cat((idx,next_idx),dim=-1)
        return idx
model=AttentionLanguageModel(vocab_size=vocab_size,embedding_dim=embedding_dim,num_heads=num_heads,block_size=block_size)
print("词表大小:",vocab_size)
print("注意力头数:",num_heads)
print("每个头的维度",embedding_dim//num_heads)
x,y=get_batch()
logits,loss=model(x,y)        
print("损失:",loss.item())
optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate)
model.train()
for step in range(max_steps):
    x,y=get_batch()
    logits,loss=model(x,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step%100==0:
        print(f"step:{step},loss:{loss.item():.4f}")
model.eval()
start=torch.tensor([[stoi["\n"]]],dtype=torch.long)
generated=model.generate(start,max_new_tokens=200)
result=decode(generated[0].tolist())
print(result)