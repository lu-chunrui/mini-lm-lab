from pathlib import Path
from datasets import load_dataset

project_dir = Path(__file__).resolve().parent
data_dir = project_dir / "data"
data_dir.mkdir(parents=True, exist_ok=True)

def save_stories(split,count,output_path):
    print(f"正在读取 TinyStories 的 {split} 数据……")
    dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    save_count=0
    with open(output_path, "w",encoding="utf-8") as f:
        for sample in dataset:
            story=sample["text"].strip()
            if not story:
                continue
            f.write(story + "\n")
            save_count+=1
            if save_count >= count:
                break
    print(f"已经保存 {save_count} 篇故事")
    print(f"保存位置：{output_path}")

train_path = data_dir / "tinystories_train.txt"
val_path = data_dir / "tinystories_val.txt"

save_stories("train",2000,train_path)
save_stories("validation",200,val_path)
print("TinyStories 小规模数据集准备完成")