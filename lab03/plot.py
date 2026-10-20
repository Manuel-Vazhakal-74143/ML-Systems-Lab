import matplotlib.pyplot as plt

models = ['MLP', 'CNN', 'Attention']
accuracies = [88.23, 89.28, 84.11]
params = [235146, 20490, 70794] 
latencies = [0.306, 12.394, 1.566] # Using the batch 128 latencies

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(params, accuracies, color=['blue', 'orange', 'green'], s=100)
for i, txt in enumerate(models):
    ax1.annotate(txt, (params[i], accuracies[i]), xytext=(5, 5), textcoords='offset points')
ax1.set_xlabel('Parameters')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Accuracy vs Parameters')

ax2.scatter(latencies, accuracies, color=['blue', 'orange', 'green'], s=100)
for i, txt in enumerate(models):
    ax2.annotate(txt, (latencies[i], accuracies[i]), xytext=(5, 5), textcoords='offset points')
ax2.set_xlabel('Latency (ms) [Batch 128]')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Accuracy vs Latency')

plt.tight_layout()
plt.savefig('results/tradeoff_plots.png')
print("Plots saved to results/tradeoff_plots.png")