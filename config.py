import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 8
block_size = 64
max_seq_len = 128

embedding_dim = 128
num_heads = 8
num_layers = 4
feedforward_dim = 256

learning_rate = 0.0005
max_steps = 1500
eval_interval = 50
eval_iterations = 50

generation_length = 50

experiment_name = "swiglu_2_123"
norm_type = "rmsnorm"
feedforward_type = "swiglu"



