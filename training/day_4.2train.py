import torch
import torch.nn as nn
import torch.nn.functional as F

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
        scores=q@k.transpose(-2,-1)
        scores = scores / (self.head_size ** 0.5)
        scores = scores.masked_fill(
            ~self.mask[:T, :T],
            float("-inf")
)
        weights=F.softmax(scores,dim=-1)
        output=weights@v
        return output
class AttentionHead(nn.Module):
    def __init__(self,vocab_size,embedding_dim,head_size,block_size):
        super().__init__()
        self.block_size=block_size
        self.position_embedding=nn.Embedding(block_size,embedding_dim)
        self.token_embedding=nn.Embedding(vocab_size,embedding_dim)
        self.attention=CausalSelfAttention(embedding_dim,head_size,block_size)
        self.output_linear=nn.Linear(head_size,vocab_size)
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
        return logits, loss
    @torch.no_grad()
    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            idx_content=idx[:,-self.block_size:]
            logits,_=self(idx_content)
            logits=logits[:,-1,:]
            probs=F.softmax(logits,dim=-1)
            idx_next=torch.multinomial(probs,num_samples=1)
            idx=torch.cat((idx,idx_next),dim=1)
        return idx
torch.manual_seed(42)
text=("人工智能正在改变世界。"
    "大语言模型可以根据上下文预测下一个字符。"
    "学习机器学习需要多写代码多做实验。\n")*100
chars=sorted(list(set(text)))
vocab_size=len(chars)
stoi={ch:i for i,ch in enumerate(chars)}
itos={i:ch for i,ch in enumerate(chars)}
def encode(s):
    return [stoi[c] for c in s]
def decode(l):
    return ''.join([itos[i] for i in l])
data=torch.tensor(encode(text),dtype=torch.long)
block_size=8
batch_size=32
def get_batch():
    positions=torch.randint(0,len(data)-block_size,(batch_size,))
    x = torch.stack([data[p:p + block_size] for p in positions])
    y = torch.stack([ data[p + 1:p + block_size + 1] for p in positions ])
    return x,y
model=AttentionHead(vocab_size,embedding_dim=32,head_size=16,block_size=block_size)
x_batch,y_batch=get_batch()
logits,loss=model(x_batch,y_batch)
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
model.train()
for step in range(1000):
    x_batch,y_batch=get_batch()
    logits,loss=model(x_batch,y_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step%100==0:
        print(f"step {step}: loss {loss.item()}")
model.eval()
start=torch.tensor([[stoi["\n"]]],dtype=torch.long)
generated=model.generate(start,max_new_tokens=100)
print("生成结果：",decode(generated[0].tolist()))