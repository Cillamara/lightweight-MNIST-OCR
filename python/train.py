import mnistocr

from torchvision import datasets
from torchvision import transforms

mnist = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

image, label = mnist[1001]

model = mnistocr.LogisticRegression(784)

for i in range(1000):
    image, label = mnist[i]

    x = image.numpy().flatten().astype("float32").tolist()
    y = 1.0 if label == 0 else 0.0

    model.train_step(x, y)
