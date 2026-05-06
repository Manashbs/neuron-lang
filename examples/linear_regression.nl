## Linear Regression in Neuron
## Demonstrates AI training with Neuron's unique syntax

grab tensor

show("Neuron -- Linear Regression Demo")
show("=" * 40)

## Generate training data: y = 3x + 2 + noise
x_data = tensor.random(100, 1)
noise = tensor.random(100, 1)
y_data = x_data * 3.0 + 2.0 + noise * 0.1

## Initialize weights
w = tensor.random(1, 1)
b = tensor.zeros(1, 1)
lr = 0.01

## Training loop
show("\nTraining...")
each epoch within upto(100):
    ## Forward pass
    y_pred = x_data.matmul(w) + b

    ## Compute MSE loss
    diff = y_pred - y_data
    loss = diff * diff
    loss_val = loss.mean()

    ## Compute gradients manually
    grad_w = x_data.transpose().matmul(diff) * (2.0 / 100)
    grad_b = diff.mean() * 2.0

    ## Update weights
    w = w - grad_w * lr
    b = b - grad_b * lr

    when epoch % 20 == 0:
        show("  Epoch {epoch}: loss = {loss_val.value}")

show("\nTraining complete!")
show("Learned: w = {w.value}, b = {b.value}")
show("Expected: w ~ 3.0, b ~ 2.0")
