import mnistocr
from torchvision import datasets, transforms

mnist = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

model = mnistocr.LogisticRegression(784)

for i in range(1000):
    image, label = mnist[i]

    x = image.numpy().flatten().astype("float32").tolist()
    y = 1.0 if label == 0 else 0.0

    model.train_step(x, y)

correct = 0
total = 100

for i in range(1000, 1100):  # test on new data
    image, label = mnist[i]

    x = image.numpy().flatten().astype("float32").tolist()
    y = 1.0 if label == 0 else 0.0

    pred = model.predict(x)

    # Convert probability → class
    pred_label = 1 if pred > 0.5 else 0

    if pred_label == y:
        correct += 1

    print(f"Pred: {pred}, True: {y}")

print(f"Accuracy: {correct}/{total}")