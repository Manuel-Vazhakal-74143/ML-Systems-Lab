# Lab 7: Model Compression (Pruning and Distillation)

## 1. Efficiency Summary Table (Ch. 9 Pillars)

| Efficiency Pillar | Baseline Teacher | Pruned Teacher (95% Global) | Distilled Student | Plain Student |
| :--- | :--- | :--- | :--- | :--- |
| **Data** (Precision/Format) | FP32 Dense | FP32 Sparse / Dense | FP32 Dense | FP32 Dense |
| **Model** (Size/Params) | 3.88 MB (~1M params) | 7.76 MB Dense / 21.73 MB Sparse | 0.41 MB (~100k params) | 0.41 MB (~100k params) |
| **Compute** (Latency) | 23.39 ms / batch | 33.48 ms / batch | 9.61 ms / batch | 9.40 ms / batch |
| **Energy** (Accuracy/Overhead)| 91.67% (High MACs) | 91.83% (Zero-multiply overhead) | 89.46% (Low MACs) | 89.13% (Low MACs) |

## 2. Honest Latency & Storage Analysis (Tasks 2 & 3)
**The Pruning Paradox:** Global unstructured pruning successfully removed 95% of the network's weights without dropping accuracy (91.83%). However, converting the model to PyTorch's sparse coordinate format heavily inflated the file size on disk to 21.73 MB due to the overhead of storing coordinate indices. Furthermore, unstructured sparsity did not improve CPU latency. Because standard CPUs process data in dense, fixed-size SIMD blocks, multiplying by zero takes the exact same number of clock cycles and energy as multiplying by a non-zero float.

## 3. Knowledge Distillation Conclusion (Task 4)
Training a compact 100k-parameter student model using a Teacher's soft-target probabilities (Distillation) improved its accuracy to 89.46%, outperforming the plain student (89.13%). The distilled student successfully retained near-Teacher accuracy while reducing the disk footprint by ~90% (0.41 MB) and more than halving the latency (9.61 ms), making it the superior deployment strategy for strict hardware constraints.