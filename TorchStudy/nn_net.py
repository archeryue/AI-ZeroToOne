# nn version
import torch

# cpu or cuda
device = torch.device('cpu')

N, D_in, D_h, D_out = 100, 1000, 100, 10

# generate training data
x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)

# define the model
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, D_h),
    torch.nn.ReLU(),
    torch.nn.Linear(D_h, D_out),
).to(device)

# define the loss function
loss_fn = torch.nn.MSELoss(reduction='sum')

# training
learning_rate = 1e-6
for t in range(500):
    # forward inference
    y_pred = model(x)
    # calculate the loss
    loss = loss_fn(y_pred, y)
    print(t, loss.item())
    # back-propagation
    model.zero_grad()
    loss.backward()
    # update the model parameters(weights)
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad