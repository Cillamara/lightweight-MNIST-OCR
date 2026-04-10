from torchvision import datasets
from torchvision import transforms

from mnistocr import LogisticRegression

def load_data(path):
    mnist = datasets.MNIST(
        root=path,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    return mnist

def train(args):
    print(train)
    exit()
    n_features = 784
    model = LogisticRegression(n_features)

    data = load_data(args.data)
    for epoch in range(args.epochs):
        for x, y in data:
            x = x.view(-1).tolist()
            model.train_step(x, y, args.lr)

    model.save(args.model)
    print(f"Model saved to {args.model}")

def predict(args):
    pass

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    sub_parser = parser.add_subparsers(dest="command", required=True)

    # for the training commands
    train_p = sub_parser.add_parser("train")
    train_p.add_argument(
        "--data",
        required=True
    )
    train_p.add_argument(
        "--model",
        required=True
    )
    train_p.add_argument(
        "--lr",
        type=float,
        default=0.01
    )
    train_p.add_argument(
        "--epochs",
        type=int,
        default=1
    )
    train_p.add_argument(
        "--threads",
        type=int, 
        default=1
    )
    train_p.add_argument(
        "--device",
        type=str, 
        default="cpu"
    ) 

    # for the prediction commands
    predict_p = sub_parser.add_parser("predict") 
    predict_p.add_argument(
        "--model",
        required=True
    )
    predict_p.add_argument(
        "--input",
        required=True
    )

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "predict":
        predict(args)
