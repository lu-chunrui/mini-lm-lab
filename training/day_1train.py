import torch
import torch.nn as nn

x=torch.tensor([[1.0],[2.0],[3.0],[4.0]])
y=torch.tensor([[3.0],[5.0],[7.0],[9.0]])
model=nn.Linear(1,1)
loss_fn=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
for epoch in range(1000):
    prediction=model(x)
    loss=loss_fn(prediction,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch%100==0:
        print(f"epoch:{epoch},loss:{loss.item()}")
print("输入5的预测值为：",model(torch.tensor([[5.0]])).item())        