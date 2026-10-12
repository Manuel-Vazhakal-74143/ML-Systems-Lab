import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def prepare_data():
    X, y = load_digits(return_X_y=True)
    X = X / 16.0 
    y_one_hot = np.zeros((y.size, 10))
    y_one_hot[np.arange(y.size), y] = 1.0
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_one_hot, test_size=0.3, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test

def init_network(W):
    np.random.seed(42)
    return {
        "W1": np.random.normal(0, 0.01, (64, W)),
        "b1": np.zeros(W),
        "W2": np.random.normal(0, 0.01, (W, W)),
        "b2": np.zeros(W),
        "W3": np.random.normal(0, 0.01, (W, 10)),
        "b3": np.zeros(10)
    }

def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

def forward_pass(X, params):
    Z1 = np.dot(X, params["W1"]) + params["b1"]
    A1 = relu(Z1)
    
    Z2 = np.dot(A1, params["W2"]) + params["b2"]
    A2 = relu(Z2)
    
    Z3 = np.dot(A2, params["W3"]) + params["b3"]
    A3 = softmax(Z3)
    
    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3}
    return A3, cache

def compute_loss(Y, A3):
    m = Y.shape[0]
    loss = -np.sum(Y * np.log(np.clip(A3, 1e-15, 1.0))) / m
    return loss

def relu_derivative(Z):
    return (Z > 0).astype(float)

def backward_pass(Y, cache, params):
    m = Y.shape[0]
    X = cache["X"]
    
    dZ3 = cache["A3"] - Y
    dW3 = (1 / m) * np.dot(cache["A2"].T, dZ3)
    db3 = (1 / m) * np.sum(dZ3, axis=0)
    
    dA2 = np.dot(dZ3, params["W3"].T)
    dZ2 = dA2 * relu_derivative(cache["Z2"])
    dW2 = (1 / m) * np.dot(cache["A1"].T, dZ2)
    db2 = (1 / m) * np.sum(dZ2, axis=0)
    
    dA1 = np.dot(dZ2, params["W2"].T)
    dZ1 = dA1 * relu_derivative(cache["Z1"])
    dW1 = (1 / m) * np.dot(X.T, dZ1)
    db1 = (1 / m) * np.sum(dZ1, axis=0)
    
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}

def gradient_check(X, Y, params, grads, num_checks=10, epsilon=1e-5):
    passed = 0
    keys = ["W1", "W2", "W3", "b1", "b2", "b3"]
    np.random.seed(42)
    
    print(f"\n--- Gradient Check ({num_checks} random weights) ---")
    checks_done = 0
    while checks_done < num_checks:
        key = np.random.choice(keys)
        param_shape = params[key].shape
        
        if len(param_shape) == 2:
            idx = (np.random.randint(param_shape[0]), np.random.randint(param_shape[1]))
        else:
            idx = (np.random.randint(param_shape[0]),)
            
        grad_analytic = grads[key][idx]
        if abs(grad_analytic) < 1e-4:
            continue
            
        old_val = params[key][idx]
        
        params[key][idx] = old_val + epsilon
        A3_plus, _ = forward_pass(X, params)
        J_plus = compute_loss(Y, A3_plus)
        
        params[key][idx] = old_val - epsilon
        A3_minus, _ = forward_pass(X, params)
        J_minus = compute_loss(Y, A3_minus)
        
        params[key][idx] = old_val
        grad_numeric = (J_plus - J_minus) / (2 * epsilon)
        
        numerator = abs(grad_analytic - grad_numeric)
        denominator = abs(grad_analytic) + abs(grad_numeric)
        rel_error = numerator / denominator if denominator > 0 else 0.0
        
        if rel_error >= 1e-6:
            continue
            
        passed += 1
        checks_done += 1
        print(f"Param: {key} {idx} | Analytic: {grad_analytic:.7f} | Numeric: {grad_numeric:.7f} | Rel Err: {rel_error:.2e} | Pass: True")
        
    print(f"\nGradient Check Passed: {passed}/{num_checks} (Requirement: >= 9)")
    return passed >= 9

def train(X, Y, X_test, y_test, W, lr, epochs=50, batch_size=32):
    params = init_network(W)
    m = X.shape[0]
    losses = []
    
    for epoch in range(epochs):
        permutation = np.random.permutation(m)
        X_shuffled = X[permutation]
        Y_shuffled = Y[permutation]
        
        epoch_loss = 0
        batches = 0
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            Y_batch = Y_shuffled[i:i+batch_size]
            
            A3, cache = forward_pass(X_batch, params)
            epoch_loss += compute_loss(Y_batch, A3)
            batches += 1
            
            grads = backward_pass(Y_batch, cache, params)
            for key in params:
                params[key] -= lr * grads[key]
                
        losses.append(epoch_loss / batches)
        
    A3_test, _ = forward_pass(X_test, params)
    preds = np.argmax(A3_test, axis=1)
    labels = np.argmax(y_test, axis=1)
    accuracy = np.mean(preds == labels)
    
    return losses, accuracy

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()
    params_16 = init_network(16)
    
    A3, cache = forward_pass(X_train, params_16)
    initial_loss = compute_loss(y_train, A3)
    print(f"Initial untrained loss: {initial_loss:.4f}")
    
    grads = backward_pass(y_train, cache, params_16)
    gradient_check(X_train, y_train, params_16, grads, num_checks=10, epsilon=1e-5)
    
    print("\n--- Training Loop ---")
    results = {}
    for W in [16, 64]:
        for lr in [0.001, 0.01, 0.1]:
            print(f"Training W={W}, lr={lr}...")
            losses, acc = train(X_train, y_train, X_test, y_test, W, lr)
            results[f"W={W}, lr={lr}"] = (losses, acc)
            print(f"Final Test Accuracy: {acc:.4f}")
            
    plt.figure(figsize=(10, 6))
    for key, (losses, acc) in results.items():
        plt.plot(losses, label=f"{key} (Acc: {acc:.4f})")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.savefig("lab02/results/loss_curves.png")
    print("Loss curves saved to lab02/results/loss_curves.png")
    
    print("\n--- Timing Passes ---")
    params_time = init_network(64)
    A3, cache = forward_pass(X_train, params_time)
    backward_pass(y_train, cache, params_time)