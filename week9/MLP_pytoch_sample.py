import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

# 모델
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784,62)
        self.fc2 = nn.Linear(64,20) # if CNN? -> c..2d.. (next time do!) 

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x
    

# MNist 데이터셋 학습
def train_mnist():
    import datasets

    mnist = datasets.load_dataset("mnist")
    xtrain, ytrain = np.array(mnist["train"]["image"]).reshape(-1, 784) / 255.0, mnist["train"]["label"]
    xtest, ytest = np.array(mnist["test"]["image"]).reshape(-1, 784) / 255.0, mnist["test"]["label"]

    def compute_val_acc(model):
        val_correct = 0
        for x, y in zip(xtest, ytest):
            x = torch.from_numpy(x).view(-1, 784).float().to(device)
            z = model(x)
            val_correct += np.argmax(z.detach().numpy()) == y
        return val_correct / len(xtest)

    lr = 1e-3
    bs = 64
    n_epochs = 10
    log_every_n_steps = 100

    # 모델 등 선언
    device = torch.device("cpu")

    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(n_epochs):
        for step, idx in enumerate(range(0, len(xtrain), bs)):
            # get batch of training examples
            x, y = xtrain[idx:idx+bs], ytrain[idx:idx+bs]

            # numpy to tensor
            x = torch.from_numpy(np.array(x)).float().to(device)
            y = torch.from_numpy(np.array(y)).long().to(device)

            # 학습
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output,y)
            #loss = (output-y)**2
            loss.backward()
            optimizer.step()

            # log
            if step % log_every_n_steps == 0:
                print(f"epoch: {epoch} | step: {step} | acc: {compute_val_acc(model):.4f}")


train_mnist()