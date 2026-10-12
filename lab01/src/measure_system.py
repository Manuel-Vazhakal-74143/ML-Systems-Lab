import os
import time
import numpy as np
import pandas as pd
import joblib
from memory_profiler import memory_usage
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_model(model, X, y):
    model.fit(X, y)

def infer_model(model, X):
    model.predict(X)

def main():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    # Single sample for inference testing
    single_sample = X_test[[0]]

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = []

    for name, model in models.items():
        print(f"Profiling {name}...")
        
        # 1. Peak Memory (Training)
        mem_train = memory_usage((train_model, (model, X_train, y_train)), max_usage=True)
        
        # 2. Training Time (1 warm-up, 5 runs)
        model.fit(X_train, y_train) # warm-up
        train_times = []
        for _ in range(5):
            start = time.perf_counter()
            model.fit(X_train, y_train)
            train_times.append(time.perf_counter() - start)
        median_train_time = np.median(train_times) * 1000 # Convert to ms
        
        # 3. Peak Memory (Inference)
        mem_infer = memory_usage((infer_model, (model, single_sample)), max_usage=True)
        
        # 4. Inference Time (100 runs)
        infer_times = []
        for _ in range(100):
            start = time.perf_counter()
            model.predict(single_sample)
            infer_times.append(time.perf_counter() - start)
        median_infer_time = np.median(infer_times) * 1000 # Convert to ms
        
        # 5. Model Size (Bytes and KB)
        model_path = f"lab01/results/model_{name}.joblib"
        joblib.dump(model, model_path)
        size_bytes = os.path.getsize(model_path)
        size_kb = size_bytes / 1024
        
        results.append({
            "Model": name,
            "Train Time (ms)": f"{median_train_time:.4f}",
            "Infer Time (ms)": f"{median_infer_time:.4f}",
            "Peak Train Mem (MB)": f"{mem_train:.2f}",
            "Peak Infer Mem (MB)": f"{mem_infer:.2f}",
            "Size (Bytes)": size_bytes,
            "Size (KB)": f"{size_kb:.2f}"
        })

    # Save and display results
    df = pd.DataFrame(results)
    df.to_csv("lab01/results/system_metrics.csv", index=False)
    print("\nSystem Measurements:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()