# Lab 1: Environment and first system measurements

## 1. Goal
The objective of this lab is to set up an isolated, reproducible Python environment, train two baseline classification models (Logistic Regression and Random Forest), and measure their system costs (training time, inference latency, memory usage, and storage size) to determine their deployment viability across different hardware budgets.

## 2. Method
I configured a Python 3.11.8 environment and pinned dependencies using `requirements.txt`. The Breast Cancer Wisconsin dataset was loaded and split into a 70/30 stratified training and testing set using a random seed of 42. I trained a `LogisticRegression` (max_iter=1000) and a `RandomForestClassifier` (n_estimators=100). Training time was measured using the median of 5 runs after a single warm-up run. Inference latency was measured using the median of 100 single-sample runs. Peak memory was tracked using the `memory_profiler` library, and model sizes were calculated from the serialized `.joblib` files. 

## 3. Results

| Model | Train Time (ms) | Infer Time (ms) | Peak Train Mem (MB) | Peak Infer Mem (MB) | Size (KB) | Deployment Budget Fit |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 185.4067 | 0.0684 | 210.46 | 210.62 | 1.08 | Mobile |
| Random Forest | 121.1603 | 3.1354 | 211.55 | 211.69 | 284.09 | Mobile |

## 4. Conclusions
1. **Deployment Suitability:** Both models fit the "Mobile" budget. Their inference memory utilization (~211 MB) falls within the 64-256 MB constraint, their latencies are well under the 20 ms limit, and their sizes are significantly below the 10 MB threshold. 
2. **TinyML Incompatibility:** Neither model can be deployed on TinyML devices. While the Logistic Regression model size (1.08 KB) fits the TinyML storage budget (≤ 100 KB), both models require over 210 MB of peak memory during inference, vastly exceeding the TinyML limit of 256 KB.
3. **Algorithm Efficiency:** Logistic Regression is far more lightweight for edge deployment. It is approximately 263 times smaller in storage size and evaluates single samples significantly faster than the tree-based ensemble, making it the better choice for highly constrained environments.