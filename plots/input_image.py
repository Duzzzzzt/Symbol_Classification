import numpy as np
import matplotlib.pyplot as plt
import os

image = np.loadtxt("../plots/input.txt")

os.makedirs("images", exist_ok=True)

plt.imshow(image, cmap="gray")
plt.title("Input Image")
plt.axis("off")

plt.savefig("images/input.png", dpi=300)

plt.show()