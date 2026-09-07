import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32
block_size = 8
max_seq_len = 256

embedding_dim = 128
num_heads = 4
num_layers = 2
feedforward_dim = 256

learning_rate = 0.001
max_steps = 1500
eval_interval = 100
eval_iterations = 30

generation_length = 100

