import matplotlib.pyplot as plt
losses = []
accs = []

with open("./heatmap.txt") as f:
    for line in f:
        loss, acc = map(float, line.split())
        losses.append(loss)
        accs.append(acc)
        print(loss)
        print(acc)

plt.figure(figsize=(10,5))
plt.plot(losses)
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.savefig("graphs/loss.png")

plt.figure(figsize=(10,5))
plt.plot(accs)
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid()
plt.savefig("graphs/accuracy.png")

plt.show()