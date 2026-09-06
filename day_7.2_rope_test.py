import torch

torch.manual_seed(42)

def precomputer_rope(head_size,block_size,base=10000.0):
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
    cos=cos[:T,:head_size].unsqueeze(0)
    sin=sin[:T,:head_size].unsqueeze(0)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated_even=(x_even*cos-x_odd*sin)
    rotated_odd=(x_even*sin+x_odd*cos)
    rotated=torch.stack([rotated_even,rotated_odd],dim=-1)
    rotated=rotated.flatten(start_dim=-2)
    return rotated
batch_size=2
block_size=8
head_size=16
k=torch.randn(batch_size,block_size,head_size)
q=torch.randn(batch_size,block_size,head_size)
print("原始k形状:",k.shape)
print("原始q形状:",q.shape)

cos, sin = precomputer_rope( head_size=head_size,block_size=block_size)
print("cos形状：", cos.shape)
print("sin形状：", sin.shape)

k_rotated=apply_rope(k,cos,sin)
q_rotated=apply_rope(q,cos,sin)
print("旋转后Q形状：", q_rotated.shape)
print("旋转后K形状：", k_rotated.shape)

scores_before = (
    q @ k.transpose(-2, -1)
) / (head_size ** 0.5)
scores_after = (
    q_rotated @ k_rotated.transpose(-2, -1)
) / (head_size ** 0.5)
print("\n使用RoPE前的注意力分数：")
print(scores_before[0])
print("\n使用RoPE后的注意力分数：")
print(scores_after[0])