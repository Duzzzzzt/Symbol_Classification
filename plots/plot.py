import matplotlib.pyplot as plt
import os
losses = []
accs = []

with open("../metrics.txt") as f:
    for line in f:
        loss, acc = map(float, line.split())
        losses.append(loss)
        accs.append(acc)

os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(10,5))
plt.plot(losses)
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.savefig("plots/loss.png")

plt.figure(figsize=(10,5))
plt.plot(accs)
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid()
plt.savefig("plots/accuracy.png")

plt.show()