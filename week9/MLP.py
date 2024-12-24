import numpy as np
import copy

# 소프트맥스 함수
def softmax(x):
    x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return x / np.sum(x, axis=-1, keepdims=True)

# Loss 함수의 미분(소프트맥스 함수 사용)
def d_loss_fn(logits, y):
    p = softmax(logits)
    p = p - np.eye(logits.shape[-1])[y]
    return p


# ReLU 함수의 미분
def d_relu(x):
    return np.where(x > 0, 1, 0)

# ReLU 함수
def relu(x):
    return np.maximum(0, x)

# 파라미터 초기화
def init_params(in_size: int, out_size: int, hidden_sizes: list[int]):
    dims = [in_size] + hidden_sizes + [out_size]
    params = []
    for nin, nout in zip(dims[:-1], dims[1:]):
        # xavier uniform initialization for weight matrixes
        a = (6 / (nin + nout)) ** 0.5
        w = np.random.uniform(-a, a, size=[nin, nout])
        b = np.zeros(nout)
        params.append([w, b])
    return params

# forward
def forward(x, params):
    tape = [(None, x)]
    for w, b in params:
        z = x @ w + b
        x = relu(z)
        tape.append((z, x))
    return z, tape[:-1]

# 역전파
def backprop(x, y, params):
    logits, tape = forward(x, params)

    grad = []
    error = d_loss_fn(logits, y)
    for (z, a), (w, _) in zip(reversed(tape), reversed(params)):
        print(  (error[..., np.newaxis, :] * a[..., :, np.newaxis]).shape)

        grad_w = np.sum(error[..., np.newaxis, :] * a[..., :, np.newaxis], axis=0) / x.shape[0]
        grad_b = np.sum(error, axis=0) / x.shape[0]
        grad.append((grad_w, grad_b))
        if z is not None:
            error = error @ w.T
            error = error * d_relu(z)
    grad = list(reversed(grad))
    
    return grad


# MNist 데이터셋 학습
def train_mnist():
    import datasets

    mnist = datasets.load_dataset("mnist")
    xtrain, ytrain = np.array(mnist["train"]["image"]).reshape(-1, 784) / 255.0, mnist["train"]["label"]
    xtest, ytest = np.array(mnist["test"]["image"]).reshape(-1, 784) / 255.0, mnist["test"]["label"]

    def compute_val_acc(params):
        val_correct = 0
        for x, y in zip(xtest, ytest):
            z, _ = forward(x, params)
            val_correct += np.argmax(z) == y
        return val_correct / len(xtest)

    lr = 1e-3
    bs = 64
    n_epochs = 10
    log_every_n_steps = 100

    params = init_params(784, 10, [64])
    for epoch in range(n_epochs):
        for step, idx in enumerate(range(0, len(xtrain), bs)):
            # get batch of training examples
            x, y = xtrain[idx:idx+bs], ytrain[idx:idx+bs]

            # compute gradient
            grad = backprop(x, y, params)

            # update the parameters
            for k in range(len(params)):
                params[k][0] -= lr * grad[k][0] 
                params[k][1] -= lr * grad[k][1]

            # log
            if step % log_every_n_steps == 0:
                print(f"epoch: {epoch} | step: {step} | acc: {compute_val_acc(params):.4f}")


train_mnist()