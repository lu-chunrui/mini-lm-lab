import torch

torch.manual_seed(42)

batch_size = 2
past_length = 3
head_size = 4

past_k = torch.randn(batch_size, past_length, head_size)
past_v = torch.randn(batch_size, past_length, head_size)

new_k = torch.randn(batch_size, 1, head_size)
new_v = torch.randn(batch_size, 1, head_size)

cached_k = torch.cat([past_k, new_k], dim=1)
cached_v = torch.cat([past_v, new_v], dim=1)

print("原K缓存形状：", past_k.shape)
print("新K形状：", new_k.shape)
print("追加后K缓存形状：", cached_k.shape)

print("原V缓存形状：", past_v.shape)
print("新V形状：", new_v.shape)
print("追加后V缓存形状：", cached_v.shape)