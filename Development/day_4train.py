import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
batch_size=2
block_size=4
embedding_dim=8
head_size=4

x=torch.rand(batch_size,block_size,embedding_dim)
query=nn.Linear(embedding_dim,head_size,bias=False)
key=nn.Linear(embedding_dim,head_size,bias=False)
value=nn.Linear(embedding_dim,head_size,bias=False)

q=query(x)
k=key(x)
v=value(x)

print("q:",q.shape)
print("k:",k.shape) 
print("v:",v.shape)

scores=q@k.transpose(-2,-1)
scores = scores / (head_size ** 0.5)
print("scores:",scores.shape)
mask=torch.tril(torch.ones(block_size,block_size,dtype=torch.bool))
scores = scores.masked_fill(~mask, float('-inf'))
weights=F.softmax(scores,dim=-1)
output=weights@v
print("第一段文本的注意力权重:",weights[0])
print("每一行的注意力权重和:",weights[0].sum(dim=-1))
print("output:",output.shape)
