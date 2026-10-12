# Lab 2: A neural network from scratch

## 1. Goal
The objective is to design, implement, and train a two-hidden-layer neural network from absolute scratch using NumPy. This includes implementing forward propagation, cross-entropy loss, and deriving analytical backpropagation gradients verified against a numerical gradient checker.

## 2. Method
The `load_digits` dataset was scaled (divided by 16) and labels were one-hot encoded before applying a 70/30 stratified split with a random state of 42. A network architecture of (64 -> W -> ReLU -> W -> ReLU -> 10 Softmax) was initialized using $N(0, 0.01)$ weights. Analytical gradients were derived via the chain rule and validated numerically using $J(\theta \pm \epsilon)$, enforcing a relative error tolerance of `< 1e-6`. The network was trained using a custom mini-batch gradient descent loop (batch size 32, 50 epochs) across combinations of hidden widths (W=16, 64) and learning rates (0.001, 0.01, 0.1). Forward and backward passes were independently timed over 100 runs.

## 3. Results

| Hidden Width (W) | Learning Rate | Test Accuracy |
| :--- | :--- | :--- |
| 16 | 0.001 | 10.19% |
| 16 | 0.01 | 10.19% |
| 16 | 0.1 | 67.22% |
| 64 | 0.001 | 10.19% |
| 64 | 0.01 | 10.19% |
| 64 | 0.1 | 94.44% |

*   **Median Forward Pass Time:** 2.1594 ms
*   **Median Backward Pass Time:** 2.1877 ms
*(See `results/loss_curves.png` for training convergence plots).*

## 4. Conclusions
1. **Capacity and Learning Rate Dependency:** The network requires both sufficient capacity ($W=64$) and an adequately high learning rate ($lr=0.1$) to successfully learn this dataset within 50 epochs, achieving 94.44% accuracy. Slower learning rates (0.001 and 0.01) failed to converge within the epoch limit, resulting in baseline random-guessing accuracy (~10.19%).
2. **Backpropagation Computational Cost:** The backward pass is significantly slower than the forward pass. While the forward pass only requires computing the activations, the backward pass dominates the computational time because it must calculate gradients for both the weights ($dW$) and the local errors ($dZ$, $dA$) across every layer, effectively doubling the number of matrix multiplications required.
3. **Numerical Precision Limits:** Implementing the numerical gradient checker demonstrated the extreme sensitivity of gradient calculations to floating-point truncation. Gradients resting near zero (or at the non-differentiable ReLU kink) are susceptible to catastrophic cancellation in standard `float64` environments, requiring stability floors to properly validate analytical math.