import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 8
block_size = 256
max_seq_len = 512

embedding_dim = 256
num_heads = 8
num_layers = 4
feedforward_dim = 1024

learning_rate = 0.0003
max_steps = 2000
eval_interval = 100
eval_iterations = 20

generation_length = 100

