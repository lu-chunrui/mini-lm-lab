import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
text=("人工智能正在改变世界。"
    "大语言模型可以根据上下文预测下一个字符。"
    "学习机器学习需要多写代码多做实验。\n")*100
chars=sorted(list(set(text)))
vocab_size=len(chars)
stoi={char:index for index,char in enumerate(chars)}
itos={index:char for index,char in enumerate(chars)}
def encode(s):
    return [stoi[c] for c in s]
def decode(l):
    return [itos[i] for i in l]
data=torch.tensor(encode(text),dtype=torch.long)
split=int( 0.9*len(data))
block_size = 8
batch_size = 32
train_data=data[:split]
val_data=data[split:]
def get_batch(data_source):
    position=torch.randint(0,len(data_source)-block_size-1,(batch_size,))
    x=torch.stack([data_source[p:p+block_size] for p in position])
    y=torch.stack([data_source[p+1:p+block_size+1] for p in position])
    return x,y
class ContextLanguageModel(nn.Module):
    def __init__(self, vocab_size,embedding_dim,block_size):
        super().__init__()
        self.block_size=block_size
        self.embedding=nn.Embedding(vocab_size,embedding_dim)
        self.network=nn.Sequential(
            nn.Linear(block_size*embedding_dim,128),
            nn.ReLU(),
            nn.Linear(128,vocab_size)
        )
    def forward(self,idx,targets=None):
        embeddings=self.embedding(idx)
        batch_size=embeddings.shape[0]
        embeddings = embeddings.reshape(batch_size, -1)
        logits=self.network(embeddings)
        if targets is None:
            return logits, None
        next_character = targets[:, -1]

        loss = F.cross_entropy(logits, next_character)

        return logits, loss
    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            idx_context = idx[:, -self.block_size:]
            logits, _ = self(idx_context)
            probabilities = F.softmax(logits, dim=-1)
            next_character = torch.multinomial(
                probabilities,
                num_samples=1
            )
            idx = torch.cat(
                [idx, next_character],
                dim=1
            )
        return idx
model = ContextLanguageModel(
    vocab_size=vocab_size,
    embedding_dim=16,
    block_size=block_size
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
for step in range(3000):
    x_batch, y_batch = get_batch(train_data)

    logits, loss = model(x_batch, y_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 300 == 0:
        print(f"step: {step}, loss: {loss.item():.4f}")
start = torch.full(
    (1, block_size),
    stoi["\n"],
    dtype=torch.long
)

generated = model.generate(
    start,
    max_new_tokens=100
)

print(decode(generated[0].tolist()))