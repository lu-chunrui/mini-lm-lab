from pathlib import Path
import csv
import torch
import math

from config import(device,batch_size,block_size,max_seq_len,embedding_dim,num_heads,num_layers,feedforward_dim,learning_rate,max_steps,eval_interval,eval_iterations,experiment_name,norm_type,feedforward_type)
from bpe_tokenizer import BPETokenizer
from model import TransformerLanguageModel
torch.manual_seed(123)

project_dir = Path(__file__).resolve().parent
checkpoint_dir = project_dir / "checkpoints" / experiment_name
checkpoint_dir.mkdir(parents=True, exist_ok=True)
best_checkpoint_path = checkpoint_dir / "best_bpe_model.pth"
latest_checkpoint_path = checkpoint_dir / "latest_checkpoint.pth"
tokenizer_path = checkpoint_dir / "bpe_tokenizer.json"
resume_training = latest_checkpoint_path.exists()
experiment_dir = project_dir / "experiments" / experiment_name
experiment_dir.mkdir(parents=True, exist_ok=True)
loss_log_path = experiment_dir / "training_with_lr_loss.csv"
if not resume_training:
    with open(loss_log_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "learning_rate", "train_loss", "val_loss"])
elif not loss_log_path.exists():
    with open(loss_log_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "learning_rate", "train_loss", "val_loss"])

train_text_path = project_dir / "data" / "tinystories_train.txt"
val_text_path = project_dir / "data" / "tinystories_val.txt"
if not train_text_path.exists():
    raise FileNotFoundError(
        "没有找到训练数据，请先运行 python prepare_data.py"
    )
if not val_text_path.exists():
    raise FileNotFoundError(
        "没有找到验证数据，请先运行 python prepare_data.py"
    )
with open(train_text_path, "r", encoding="utf-8") as file:
    train_text = file.read()
with open(val_text_path, "r", encoding="utf-8") as file:
    val_text = file.read()
print("训练文本字符数：", len(train_text))
print("验证文本字符数：", len(val_text))
if resume_training:
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            "没有找到分词器，请先运行 python train.py"
        )
    tokenizer = BPETokenizer.load(tokenizer_path)
    print("加载已有Tokenizer：", tokenizer_path)
else:
    tokenizer = BPETokenizer()
    tokenizer.train(train_text,num_steps=50)
    tokenizer.save(tokenizer_path)
    print("训练新Tokenizer：", tokenizer_path)
train_token_ids = tokenizer.encode(train_text)
val_token_ids = tokenizer.encode(val_text)
train_data = torch.tensor(
    train_token_ids,
    dtype=torch.long
)
val_data = torch.tensor(
    val_token_ids,
    dtype=torch.long
)
print("词表大小：", tokenizer.vocab_size)
print("训练token数量：", len(train_data))
print("验证token数量：", len(val_data))

def get_batch(data_source):
    positions=torch.randint(0,len(data_source)-block_size,(batch_size,))
    x=torch.stack([data_source[i:i+block_size] for i in positions])
    y=torch.stack([data_source[i+1:i+block_size+1] for i in positions])
    return x.to(device),y.to(device)

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    results={}
    datasets={"train":train_data,"val":val_data}
    for name,data_source in datasets.items():
        losses=[]
        for _ in range(eval_iterations):
            x,y=get_batch(data_source)
            logits,loss=model(x,target=y)
            losses.append(loss.item())
        results[name]=(sum(losses) / len(losses))
    model.train()
    return results

model=TransformerLanguageModel(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    feedforward_dim=feedforward_dim,
    max_seq_len=max_seq_len,
    norm_type=norm_type,
).to(device)
parameter_count=sum(parameter.numel()for parameter in model.parameters())
print("模型参数数量：", parameter_count)

optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate)
start_step = 0
best_val_loss=float("inf")
if resume_training:
    checkpoint=torch.load(latest_checkpoint_path,map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_step=checkpoint["step"]+1
    best_val_loss=checkpoint.get("best_val_loss",
    checkpoint.get("val_loss", float("inf")))
    print("成功加载训练断点：", latest_checkpoint_path)
    print("上次训练到 step：", checkpoint["step"])
    print("本次从 step 开始：", start_step)
    print("历史最佳验证loss：", best_val_loss)
else:
    print("没有发现训练断点，从头开始训练")
def get_learning_rate(step):
    warmup_steps=200
    minimum_learning_rate=3e-5
    if step<warmup_steps:
        return learning_rate * (step + 1) / warmup_steps
    decay_ratio=((step-warmup_steps)/(max_steps-warmup_steps))
    decay_ratio=min(max(decay_ratio,0),1)
    coefficient=0.5*(1.0+math.cos(math.pi*decay_ratio))
    
    return minimum_learning_rate + coefficient*(learning_rate-minimum_learning_rate)
    
model.train()
for step in range(start_step,max_steps):
    current_learning_rate=get_learning_rate(step)
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = current_learning_rate
    x,y=get_batch(train_data)
    logits,loss=model(x,target=y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
    optimizer.step()
    if (step%eval_interval==0 or step==max_steps-1):
        results=estimate_loss(model)
        train_loss=results["train"]
        val_loss=results["val"]
        print(f"step={step}, learning_rate={current_learning_rate:.6f}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        with open(loss_log_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([step, current_learning_rate, train_loss, val_loss])
        if val_loss<best_val_loss:
            best_val_loss=val_loss
            checkpoint={
                "model_state_dict":model.state_dict(),
                "optimizer_state_dict":optimizer.state_dict(),
                "step":step,
                "val_loss":val_loss,
                "vocab_size":tokenizer.vocab_size,
                "embedding_dim":embedding_dim,
                "num_heads":num_heads,
                "num_layers":num_layers,
                "feedforward_dim":feedforward_dim,
                "max_seq_len":max_seq_len,
                "norm_type":norm_type,
                "experiment_name": experiment_name,
                "feedforward_type": feedforward_type,
            }
            torch.save(checkpoint,best_checkpoint_path)
            print("保存最佳模型：", best_checkpoint_path)
        latest_checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "best_val_loss": best_val_loss,
            "vocab_size": tokenizer.vocab_size,
            "embedding_dim": embedding_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "feedforward_dim": feedforward_dim,
            "max_seq_len": max_seq_len,
            "norm_type": norm_type,
            "experiment_name": experiment_name,
            "feedforward_type": feedforward_type,
        }

        torch.save(
            latest_checkpoint,
            latest_checkpoint_path
        )
        print(
            "保存最近训练状态：",
            latest_checkpoint_path
        )
    
print("训练完成")
print("最佳验证损失：", best_val_loss)
print("最佳模型已保存至：", best_checkpoint_path)
print("最近训练状态保存至：", latest_checkpoint_path)