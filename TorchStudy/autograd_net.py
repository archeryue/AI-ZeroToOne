import torch

# cpu or cuda
device = torch.device('cpu')

N, D_in, D_h, D_out = 100, 1000, 100, 10

# generate training data
x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)

# init the weights
w1 = torch.randn(D_in, D_h, device=device, requires_grad=True)
w2 = torch.randn(D_h, D_out, device=device, requires_grad=True)

# training
learning_rate = 1e-6
for t in range(500):
    # forward inference
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    # calculate the loss
    loss = (y_pred - y).pow(2).sum()
    print(t, loss)
    # use autograd to do the back-propagation
    loss.backward()
    # update the weights, prevent torch do autograd for this part
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        # zero grads
        w1.grad.zero_()
        w2.grad.zero_()
