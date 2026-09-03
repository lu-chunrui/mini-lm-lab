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
torch.manual_seed(42)
attention=CausalSelfAttention(embedding_dim=8,head_size=4,block_size=8)
x=torch.rand(2,4,8)
out=attention(x)
print("输入x:",x.shape)
print("输出out:",out.shape)
