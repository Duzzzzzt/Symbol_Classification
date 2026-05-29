import numpy as np
import matplotlib.pyplot as plt
import os

filters = []
current = []

with open("../heatmap.txt", "r") as f:
    for line in f:
        line = line.strip()

        if line.startswith("FILTER"):
            if current:
                filters.append(np.array(current))
                current = []
            continue

        if line == "":
            continue

        row = [float(x) for x in line.split()]
        current.append(row)

if current:
    filters.append(np.array(current))


os.makedirs("images", exist_ok=True)

fig, axes = plt.subplots(2, 4, figsize=(10, 5))

for i, ax in enumerate(axes.flat):
    ax.imshow(filters[i], cmap="hot")
    ax.set_title(f"Filter {i}")
    ax.axis("off")

plt.tight_layout()


plt.savefig("images/heatmaps.png", dpi=300)

plt.show()