import torch

# cpu or cuda
device = torch.device('cpu')

N, D_in, D_h, D_out = 100, 1000, 100, 10

# generate the training data
x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)

# init the weights
w1 = torch.randn(D_in, D_h, device=device)
w2 = torch.randn(D_h, D_out, device=device)

learning_rate = 1e-6
for t in range(500):
    # forward inference
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)
    # calculate the loss
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())
    # back-propagation
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)
    # update the weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
