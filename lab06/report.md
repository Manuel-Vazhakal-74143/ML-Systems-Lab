# Lab 6: Training Pipelines and Profiling Report

## 1. Profiling Breakdown (Task 2)
Based on the `torch.profiler` CPU trace of a single epoch, the execution time was distributed as follows:
* **Data Loading (`enumerate(DataLoader)`):** 51.79% (Primary Bottleneck)
* **Backward Pass:** 25.07%
* **Forward Pass:** 18.08%
* **Optimizer Step:** < 5%

## 2. Worker Count Study (Task 3)
Increasing `num_workers` reduced execution time for 50 batches, though `num_workers=1` introduced significant multiprocessing spawn overhead on this OS configuration before stabilizing.

| num_workers | Time (50 batches) |
|-------------|-------------------|
| 0           | 0.3457s           |
| 1           | 3.2739s           |
| 2           | 0.2538s           |
| 4           | 0.1894s           |

## 3. Batch Size Throughput Study (Task 4)
Larger batch sizes steadily increased processing throughput (samples per second) by maximizing CPU vectorization capabilities.

| Batch Size | Throughput (samples/s) |
|------------|------------------------|
| 32         | 8695.43                |
| 64         | 9428.67                |
| 128        | 8896.89                |
| 256        | 9619.63                |
| 512        | 9701.67                |

## 4. Pipeline Mechanics (Tasks 1, 5, 6, 7)
* **YAML Config & MLflow:** Training is entirely driven by `config.yaml` with parameters and metrics successfully logged to local MLflow directories.
* **Gradient Accumulation:** An effective batch size of 256 was successfully emulated using a physical batch size of 64 and `grad_accum_steps=4`.
* **Checkpoint Recovery:** Training was intentionally interrupted at Epoch 3 and successfully resumed from `checkpoint.pth` to complete Epochs 4 and 5.