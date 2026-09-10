import torch
import torch.nn as nn
import torch.nn.functional as F
from config import norm_type,feedforward_type
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(dim))
    def forward(self, x):
        mean_square=x.pow(2).mean(dim=-1, keepdim=True)
        inverse_rms=torch.rsqrt(mean_square+self.eps)
        return x*inverse_rms*self.weight
def create_norm(embedding_dim, norm_type):
    if norm_type == "rmsnorm":
        return RMSNorm(embedding_dim)
    if norm_type == "layer_norm":
        return nn.LayerNorm(embedding_dim,elementwise_affine=True,bias=False)
    raise ValueError(f"未知的归一化类型：{norm_type}")
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
class GELUFeedForward(nn.Module):
    def __init__(self,embedding_dim,feedforward_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim,feedforward_dim,bias=False),
            nn.GELU(),
            nn.Linear(feedforward_dim,embedding_dim,bias=False)
        )
    def forward(self, x):
        return self.net(x)
def create_feedforward(embedding_dim,feedforward_dim,feedforward_type):
    if feedforward_type == "swiglu":
        return SwiGLUFeedForward(embedding_dim,feedforward_dim)
    if feedforward_type == "gelu":
        gelu_hidden_dim = int( feedforward_dim * 3 / 2)
        return GELUFeedForward(embedding_dim,gelu_hidden_dim )
    raise ValueError(
        f"未知的前馈网络类型：{feedforward_type}"
    )
class TransformerBlock(nn.Module):
    def __init__(self,embedding_dim,num_heads,feedforward_dim,max_seq_len,norm_type=norm_type,feedforward_type=feedforward_type):
        super().__init__()
        self.attention=MultiHeadAttention(embedding_dim=embedding_dim, num_heads=num_heads, max_seq_len=max_seq_len)
        self.feedforward=create_feedforward(embedding_dim=embedding_dim, feedforward_dim=feedforward_dim, feedforward_type=feedforward_type)
        self.norm1=create_norm(embedding_dim, norm_type)
        self.norm2=create_norm(embedding_dim, norm_type)
    def forward(self,x,past_kv=None,use_cache=False):
        attn_output, attn_cache=self.attention(self.norm1(x), past_kv, use_cache)
        x=x+attn_output
        x=x+self.feedforward(self.norm2(x))
        return x, attn_cache
    
class TransformerLanguageModel(nn.Module):
    def __init__(self,vocab_size,embedding_dim,num_heads,num_layers,feedforward_dim,max_seq_len,norm_type=norm_type):
        super().__init__()
        self.embedding_dim=embedding_dim
        self.num_heads=num_heads
        self.feedforward_dim=feedforward_dim
        self.max_seq_len=max_seq_len
        self.token_embedding=nn.Embedding(vocab_size, embedding_dim)
        self.blocks=nn.ModuleList(
            [TransformerBlock(embedding_dim=embedding_dim, num_heads=num_heads, feedforward_dim=feedforward_dim, max_seq_len=max_seq_len, norm_type=norm_type,feedforward_type=feedforward_type) for _ in range(num_layers)]
        )
        self.final_norm=create_norm(embedding_dim, norm_type)
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
    def sample_next_token(self,logits,temperature=0.3,top_k=1):
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