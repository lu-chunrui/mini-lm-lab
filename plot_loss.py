from pathlib import Path
import csv
import matplotlib.pyplot as plt

project_dir = Path(__file__).resolve().parent
experiment_dir = project_dir / "experiments"
loss_log_path = experiment_dir / "training_with_lr_loss.csv"
figure_path = experiment_dir / "loss_curve.png"

steps=[]
train_losses=[]
val_losses=[]
with open(loss_log_path, "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        steps.append(int(row["step"]))
        train_losses.append(float(row["train_loss"]))
        val_losses.append(float(row["val_loss"]))

plt.figure(figsize=(10, 6))
plt.plot(steps, train_losses, label="training loss")
plt.plot(steps, val_losses, label="validation loss")
plt.xlabel("step")
plt.ylabel("loss")
plt.legend()
plt.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig(figure_path, dpi=200)
print(f"loss curve saved to {figure_path}")
plt.show()
