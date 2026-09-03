import torch
import torch.nn as nn
import torch.nn.functional as F

text = (
    "hello world. "
    "hello pytorch. "
    "hello transformer. "
) * 200
chars=sorted(list(set(text)))
bocab_size=len(chars)
print("全部字符",chars)
print("字符数量",bocab_size)

itos={index:char for index,char in enumerate(chars)}
stoi={char:index for index,char in enumerate(chars)}
def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return [itos[i] for i in l]
data=torch.tensor(encode(text),dtype=torch.long)
x = data[:-1].unsqueeze(0) 
y = data[1:].unsqueeze(0)
print("输入示例：", decode(x[0][:20].tolist()))
print("目标示例：", decode(y[0][:20].tolist()))
print("x形状：", x.shape)
print("y形状：", y.shape)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding=nn.Embedding(num_embeddings=vocab_size,
            embedding_dim=vocab_size)
    def forward(self,idx,targets=None):
        logits=self.embedding(idx)
        loss=None
        if targets is not None:
            B,T,C=logits.shape
            logits_flat = logits.reshape(B * T, C)
            targets_flat = targets.reshape(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        return logits,loss
    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            logits,_=self(idx)
            logits=logits[:,-1,:]
            probs=F.softmax(logits,dim=-1)
            idx_next=torch.multinomial(probs,num_samples=1)
            idx=torch.cat((idx,idx_next),dim=1)
        return idx
model=BigramLanguageModel(bocab_size)
optimizer=torch.optim.Adam(model.parameters(),lr=0.05)
for epoch in range(1000):
    logits,loss=model(x,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch%100==0:
        print(f"epoch:{epoch},loss:{loss.item()}")
    