import numpy as np

N, D_in, D_h, D_out = 100, 1000, 100, 1

# generate the training data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# init the weights
w1 = np.random.randn(D_in, D_h)
w2 = np.random.randn(D_h, D_out)

learning_rate = 1e-6

for t in range(500):
    # forward inference
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)
    # calculate the loss
    loss = np.square(y_pred - y).sum()
    print(t, loss)
    # back-propagation
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)
    # update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
