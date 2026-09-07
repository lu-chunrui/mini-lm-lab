from pathlib import Path
from networkx import bethe_hessian_matrix
import torch

from config import(device,batch_size,block_size,max_seq_len,embedding_dim,num_heads,num_layers,feedforward_dim,learning_rate,max_steps,eval_interval,eval_iterations)
from tokenizer import CharTokenizer
from model import TransformerLanguageModel
torch.manual_seed(123)
project_dir = Path(__file__).resolve().parent
checkpoint_dir = project_dir / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
checkpoint_path = checkpoint_dir / "best_kv_cache_model.pth"

text = (
    "人工智能正在改变世界。"
    "大语言模型可以根据上下文预测下一个字符。"
    "学习机器学习需要多写代码多做实验。\n"
) * 100
tokenizer = CharTokenizer(text)
data = torch.tensor(tokenizer.encode(text),dtype=torch.long)
split_index = int(0.8 * len(data))
train_data = data[:split_index]
val_data = data[split_index:]
print("词表大小：", tokenizer.vocab_size)
print("训练token数量：", len(train_data))
print("验证token数量：", len(val_data))

def get_batch(data_source):
    positions=torch.randint(0,len(data_source)-block_size,(batch_size,))
    x=torch.stack([data_source[i:i+block_size] for i in positions])
    y=torch.stack([data_source[i+1:i+block_size+1] for i in positions])
    return x.to(device),y.to(device)

@torch.no_grad
def estimate_loss(model):
    model.eval()
    results={}
    datasets={"train":train_data,"val":val_data}
    for name,data_source in datasets.items():
        losses=[]
        for _ in range(eval_iterations):
            x,y=get_batch(data_source)
            logits,loss=model(x)
            losses.append(loss.item())
        results[name]=torch.mean(torch.tensor(losses))
        model.train()
        return results

model=TransformerLanguageModel(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    feedforward_dim=feedforward_dim,
    learning_rate=learning_rate,
).to(device)
parameter_count=model.count_parameters(parameter.numel()for parameter in model.parameters())
print("模型参数数量：", parameter_count)

optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate)

beat_val_loss=float("inf")
model.train()
for step in range(max_steps):
    x,y=get_batch(train_data)
    logits,loss=model(x)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
    optimizer.step()
    if (step%eval_interval or step==max_steps-1)==0:
        results=estimate_loss(model)
        train_loss=results["train"]
        val_loss=results["val"]
        print(f"step={step}, "f"train_loss={train_loss:.4f}, "f"val_loss={val_loss:.4f}")
        if val_loss<best_val_loss:
            best_val_loss=val_loss
            checkpoint={
                "model_state_dict":model.state_dict(),
                "optimizer_state_dict":optimizer.state_dict(),
                "step":step,
                "val_loss":val_loss,
                "chars":tokenizer.chars,
                "vocab_size":tokenizer.vocab_size,
                "embedding_dim":embedding_dim,
                "num_heads":num_heads,
                "num_layers":num_layers,
                "feedforward_dim":feedforward_dim,
                "max_seq_len":max_seq_len
            }
            torch.save(checkpoint,checkpoint_path)
            print("保存最佳模型：",checkpoint_path)
print("训练完成")
print("最佳验证损失：",best_val_loss)
print("最佳模型已保存至：",checkpoint_path)