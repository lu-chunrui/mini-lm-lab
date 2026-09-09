from pathlib import Path
import csv
import matplotlib.pyplot as plt

project_dir = Path(__file__).resolve().parent
experiment_dir = project_dir / "experiments"
swiglu_log_path = (experiment_dir/ "swiglu_2_123"/ "training_with_lr_loss.csv")
gelu_log_path = (experiment_dir/ "gelu_2_123"/"training_with_lr_loss.csv")
figure_path = (experiment_dir/ "feedforward_comparison.png")

def load_loss_log(log_path):
    steps=[]
    train_losses=[]
    val_losses=[]
    with open(log_path, "r",encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            steps.append(int(row["step"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))
    return steps, train_losses, val_losses
swiglu_steps, swiglu_train, swiglu_val = (load_loss_log(swiglu_log_path))

gelu_steps, gelu_train, gelu_val = (load_loss_log(gelu_log_path))
plt.figure(figsize=(10, 6))

plt.plot(swiglu_steps, swiglu_train,linestyle="--",alpha=0.6, label="swiglu training loss")

plt.plot(gelu_steps, gelu_train,linestyle="--",alpha=0.6, label="gelu training loss")

plt.plot(swiglu_steps, swiglu_val,linewidth=2.0, label="swiglu validation loss")

plt.plot(gelu_steps, gelu_val,linewidth=2.0, label="gelu validation loss")
plt.xlabel("training step")
plt.ylabel("loss")
plt.legend()
plt.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig(figure_path, dpi=200)
print(f"loss curve saved to {figure_path}")
plt.show()
