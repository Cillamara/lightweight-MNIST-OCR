import os
import sys
import gzip
import struct
import urllib.request
import numpy as np

# Add project root to path so Python finds mnistocr.so
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

try:
    from mnistocr import LogisticRegression
except ImportError:
    print("ERROR: mnistocr.so not found. Run ./compile.sh first.")
    sys.exit(1)

# MNIST loader

MNIST_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}

def download_file(url, dest):
    if os.path.exists(dest):
        return
    print(f"Downloading {url}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    urllib.request.urlretrieve(url, dest)

def load_images(path):
    """IDX3 image file -> float32 array [n, 784] in [0, 1]"""
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">4I", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype(np.float32) / 255.0

def load_labels(path):
    """IDX1 label file -> int array [n]"""
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">2I", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int64)

def load_mnist(root, train=True):
    prefix = "train" if train else "test"
    img_file = os.path.join(root, MNIST_FILES[f"{prefix}_images"])
    lbl_file = os.path.join(root, MNIST_FILES[f"{prefix}_labels"])
    download_file(f"{MNIST_MIRROR}/{MNIST_FILES[f'{prefix}_images']}", img_file)
    download_file(f"{MNIST_MIRROR}/{MNIST_FILES[f'{prefix}_labels']}", lbl_file)
    return load_images(img_file), load_labels(lbl_file)


# Class map helpers 

def parse_class_map(s):
    """Parse user string into a {digit: class_id} dict.

    Accepts:
      "10"                            -> identity map {0:0, ..., 9:9}
      "even_odd"                      -> {0:0,2:0,4:0,6:0,8:0, 1:1,3:1,5:1,7:1,9:1}
      "0,1,2,3,4 vs 5,6,7,8,9"       -> group 0 vs group 1
      "0,1 vs 2,3 vs 4,5 vs 6,7 vs 8,9" -> 5 groups
      "{7:1, 0:0, 1:0, ...}"         -> literal dict
      "[[0,1,2],[3,4,5],[6,7,8,9]]"  -> literal list of lists
    """
    s = s.strip()

    # shorthand: just a number
    if s.isdigit():
        n = int(s)
        return {d: d for d in range(n)}

    # shorthand: even_odd
    if s.lower() == "even_odd":
        return {0:0, 2:0, 4:0, 6:0, 8:0, 1:1, 3:1, 5:1, 7:1, 9:1}

    # "X,X vs X,X vs ..." syntax
    if " vs " in s:
        groups = s.split(" vs ")
        out = {}
        for cls_id, group in enumerate(groups):
            for d in group.split(","):
                d = int(d.strip())
                out[d] = cls_id
        return out

    # try as python literal (dict or list of lists)
    import ast
    obj = ast.literal_eval(s)
    if isinstance(obj, dict):
        return {int(k): int(v) for k, v in obj.items()}
    if isinstance(obj, list):
        out = {}
        for cls_id, digits in enumerate(obj):
            for d in digits:
                out[int(d)] = cls_id
        return out

    raise ValueError(f"Cannot parse class map: {s}")


def remap_labels(y, class_map, n_classes):
    """Remap digit labels -> class ids. Returns (remapped_y, keep_mask)."""
    mapped = np.full_like(y, -1)
    for digit, cls_id in class_map.items():
        mapped[y == digit] = cls_id
    keep = mapped >= 0
    return mapped, keep


# Command Line Interface

def interactive_setup():
    """Ask the user what they want to do and how to configure it."""
    print("=" * 50)
    print("  MNIST OCR — CUDA Logistic Regression")
    print("=" * 50)
    print()
    print("What would you like to do?")
    print("  1) Train a new model")
    print("  2) Train one-vs-rest (10 binary models, one per digit)")
    print("  3) Run inference with a trained model")
    print("  4) Run inference with one-vs-rest ensemble (all 10 models)")
    print()

    choice = input("Enter 1-4: ").strip()

    if choice == "1":
        interactive_train()
    elif choice == "2":
        interactive_train_ovr()
    elif choice == "3":
        interactive_predict()
    elif choice == "4":
        interactive_predict_ovr()
    else:
        print(f"Unknown choice: {choice}")


def interactive_train():
    print()
    print("--- Training Configuration ---")
    print()

    # Data directory
    data_dir = input("MNIST data directory [./data]: ").strip() or "./data"

    # Class map
    print()
    print("How should digits be classified?")
    print("  Examples:")
    print("    10                  -> standard 10-way (each digit is its own class)")
    print("    even_odd            -> even digits vs odd digits (2 classes)")
    print("    0,1,2,3,4 vs 5,6,7,8,9  -> low vs high (2 classes)")
    print("    0,1,2 vs 3,4,5 vs 6,7,8,9  -> 3 custom groups")
    print("    [[0,1],[2,3],[4,5],[6,7],[8,9]]  -> 5 paired groups")
    print()
    class_str = input("Class mapping [10]: ").strip() or "10"
    class_map = parse_class_map(class_str)
    n_classes = max(class_map.values()) + 1

    # Print what was parsed
    inverse = {}
    for digit, cls in sorted(class_map.items()):
        inverse.setdefault(cls, []).append(digit)
    print(f"\n  -> {n_classes} classes:")
    for cls_id in sorted(inverse):
        print(f"     class {cls_id}: digits {inverse[cls_id]}")

    # Hyperparameters
    print()
    lr = float(input("Learning rate [0.01]: ").strip() or "0.01")
    epochs = int(input("Epochs [5]: ").strip() or "5")
    save_path = input("Save model to [model.bin]: ").strip() or "model.bin"

    # Load data
    print(f"\nLoading MNIST from {data_dir}...")
    X_train, y_train = load_mnist(data_dir, train=True)
    X_test, y_test = load_mnist(data_dir, train=False)

    # Create model (or resume from existing)
    model = LogisticRegression(784, n_classes)
    if os.path.exists(save_path):
        resume_input = input(f"\n{save_path} already exists. Resume training? [y/N]: ").strip().lower()
        if resume_input in ("y", "yes"):
            model.load(save_path)
            print("Resumed from existing model.")
    print(f"Model: 784 features, {n_classes} classes\n")

    # Remap labels
    y_train_mapped, train_keep = remap_labels(y_train, class_map, n_classes)
    y_test_mapped, test_keep = remap_labels(y_test, class_map, n_classes)

    train_X = X_train[train_keep]
    train_y = y_train_mapped[train_keep]
    test_X = X_test[test_keep]
    test_y = y_test_mapped[test_keep]

    print(f"Training samples: {len(train_X)}  (dropped {len(X_train) - len(train_X)})")
    print(f"Test samples:     {len(test_X)}  (dropped {len(X_test) - len(test_X)})")
    print()

    # Train
    for epoch in range(epochs):
        # Shuffle
        perm = np.random.permutation(len(train_X))

        total_loss = 0.0
        for i in perm:
            x = train_X[i].tolist()
            y = int(train_y[i])
            model.train_step(x, y, lr)

        # Evaluate on test set
        correct = 0
        for i in range(len(test_X)):
            probs = model.predict(test_X[i].tolist())
            pred = probs.index(max(probs))
            if pred == test_y[i]:
                correct += 1
        acc = correct / len(test_X)

        print(f"  epoch {epoch+1}/{epochs}  test_acc={acc:.4f}")

    # Save
    model.save(save_path)
    print(f"\nModel saved to {save_path}")

    # Save class map alongside model so inference knows the mapping
    import json
    meta_path = save_path + ".meta.json"
    meta = {
        "n_features": 784,
        "n_classes": n_classes,
        "class_map": {str(k): v for k, v in class_map.items()},
        "inverse_map": {str(k): v for k, v in inverse.items()},
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Class map saved to {meta_path}")


def interactive_train_ovr():
    """Train 10 binary models (one-vs-rest), one per digit."""
    print()
    print("--- One-vs-Rest Training (10 binary models) ---")
    print()

    data_dir = input("MNIST data directory [./data]: ").strip() or "./data"
    lr = float(input("Learning rate [0.01]: ").strip() or "0.01")
    epochs = int(input("Epochs per model [5]: ").strip() or "5")
    save_dir = input("Save directory [./models_ovr]: ").strip() or "./models_ovr"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nLoading MNIST from {data_dir}...")
    X_train, y_train = load_mnist(data_dir, train=True)
    X_test, y_test = load_mnist(data_dir, train=False)

    # Check for existing models to resume from
    resume = False
    existing = [os.path.join(save_dir, f"digit_{d}.bin") for d in range(10)]
    if all(os.path.exists(p) for p in existing):
        resume_input = input("Existing models found. Resume training? [y/N]: ").strip().lower()
        resume = resume_input in ("y", "yes")

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples:     {len(X_test)}\n")

    for digit in range(10):
        print(f"=== Training model for digit {digit} (is it a {digit}?) ===")

        # Binary class map: target digit -> class 1, everything else -> class 0
        class_map = {}
        for d in range(10):
            class_map[d] = 1 if d == digit else 0

        y_train_mapped, train_keep = remap_labels(y_train, class_map, 2)
        y_test_mapped, test_keep = remap_labels(y_test, class_map, 2)
        train_X = X_train[train_keep]
        train_y = y_train_mapped[train_keep]
        test_X = X_test[test_keep]
        test_y = y_test_mapped[test_keep]

        model = LogisticRegression(784, 2)
        if resume:
            model.load(existing[digit])
            print(f"  Resumed from {existing[digit]}")

        for epoch in range(epochs):
            perm = np.random.permutation(len(train_X))
            for i in perm:
                model.train_step(train_X[i].tolist(), int(train_y[i]), lr)

            # Evaluate
            correct = 0
            for i in range(len(test_X)):
                probs = model.predict(test_X[i].tolist())
                pred = probs.index(max(probs))
                if pred == test_y[i]:
                    correct += 1
            acc = correct / len(test_X)
            print(f"  epoch {epoch+1}/{epochs}  test_acc={acc:.4f}")

        save_path = os.path.join(save_dir, f"digit_{digit}.bin")
        model.save(save_path)
        print(f"  Saved to {save_path}\n")

    # Save metadata
    import json
    meta_path = os.path.join(save_dir, "ovr_meta.json")
    meta = {
        "type": "one_vs_rest",
        "n_models": 10,
        "n_features": 784,
        "n_classes_per_model": 2,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"All 10 models saved to {save_dir}/")
    print(f"Use option 4 (one-vs-rest inference) to run them together.")


def interactive_predict_ovr():
    """Load all 10 binary models and ensemble their predictions."""
    print()
    print("--- One-vs-Rest Inference (10 binary models) ---")
    print()

    model_dir = input("Model directory [./models_ovr]: ").strip() or "./models_ovr"

    # Load all 10 models
    models = []
    for digit in range(10):
        path = os.path.join(model_dir, f"digit_{digit}.bin")
        if not os.path.exists(path):
            print(f"ERROR: {path} not found. Train with option 2 first.")
            return
        model = LogisticRegression(784, 2)
        model.load(path)
        models.append(model)
    print(f"Loaded 10 models from {model_dir}/\n")

    # Choose input source
    print("Input source:")
    print("  1) MNIST test set (pick random samples)")
    print("  2) Image file (28x28 grayscale PNG)")
    print()
    src = input("Enter 1 or 2: ").strip()

    if src == "1":
        data_dir = input("MNIST data directory [./data]: ").strip() or "./data"
        X_test, y_test = load_mnist(data_dir, train=False)
        n_samples = int(input("How many samples to test? [10]: ").strip() or "10")

        indices = np.random.choice(len(X_test), size=n_samples, replace=False)
        correct = 0
        for idx in indices:
            x = X_test[idx].tolist()
            true_digit = int(y_test[idx])

            # Ask each model "is it digit d?" and get P(yes)
            scores = []
            for d in range(10):
                probs = models[d].predict(x)
                scores.append(probs[1])  # P(class 1) = P(is this digit)

            pred_digit = scores.index(max(scores))
            conf = max(scores)
            status = "CORRECT" if pred_digit == true_digit else "WRONG"
            if pred_digit == true_digit:
                correct += 1

            print(f"  true={true_digit}  pred={pred_digit}  conf={conf:.3f}  {status}")

        print(f"\nAccuracy: {correct}/{n_samples} = {correct/n_samples:.4f}")

    elif src == "2":
        filepath = input("Path to image: ").strip()
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return

        from PIL import Image
        img = Image.open(filepath).convert("L").resize((28, 28))
        x = np.array(img, dtype=np.float32).flatten() / 255.0
        x = x.tolist()

        scores = []
        for d in range(10):
            probs = models[d].predict(x)
            scores.append(probs[1])

        pred_digit = scores.index(max(scores))
        conf = max(scores)

        print(f"\nPredicted digit: {pred_digit}  confidence: {conf:.4f}")
        print(f"All scores: {[f'{s:.4f}' for s in scores]}")

    else:
        print(f"Unknown choice: {src}")


def interactive_predict():
    print()
    print("--- Inference ---")
    print()

    model_path = input("Model file [model.bin]: ").strip() or "model.bin"
    meta_path = model_path + ".meta.json"

    # Load metadata
    import json
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        n_classes = meta["n_classes"]
        inverse_map = {int(k): v for k, v in meta["inverse_map"].items()}
        print(f"Loaded model: {n_classes} classes")
        for cls_id in sorted(inverse_map):
            print(f"  class {cls_id}: digits {inverse_map[cls_id]}")
    else:
        n_classes = int(input("Number of classes: ").strip())
        inverse_map = None

    model = LogisticRegression(784, n_classes)
    model.load(model_path)
    print()

    # Choose input source
    print("Input source:")
    print("  1) MNIST test set (pick random samples)")
    print("  2) Image file (28x28 grayscale PNG)")
    print()
    src = input("Enter 1 or 2: ").strip()

    if src == "1":
        data_dir = input("MNIST data directory [./data]: ").strip() or "./data"
        X_test, y_test = load_mnist(data_dir, train=False)
        n_samples = int(input("How many samples to test? [10]: ").strip() or "10")

        indices = np.random.choice(len(X_test), size=n_samples, replace=False)
        correct = 0
        for idx in indices:
            x = X_test[idx].tolist()
            true_digit = int(y_test[idx])
            probs = model.predict(x)
            pred_class = probs.index(max(probs))
            conf = max(probs)

            label = ""
            if inverse_map:
                label = f" (digits {inverse_map.get(pred_class, '?')})"

            status = "OK" if inverse_map is None else ""
            if inverse_map:
                # check if the true digit belongs to the predicted class
                if true_digit in inverse_map.get(pred_class, []):
                    status = "CORRECT"
                    correct += 1
                else:
                    status = "WRONG"

            print(f"  digit={true_digit}  -> class {pred_class}{label}  conf={conf:.3f}  {status}")

        if inverse_map:
            print(f"\nAccuracy: {correct}/{n_samples} = {correct/n_samples:.4f}")

    elif src == "2":
        filepath = input("Path to image: ").strip()
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return

        # Load image as grayscale 28x28, flatten, normalize to [0,1]
        from PIL import Image
        img = Image.open(filepath).convert("L").resize((28, 28))
        x = np.array(img, dtype=np.float32).flatten() / 255.0

        probs = model.predict(x.tolist())
        pred_class = probs.index(max(probs))
        conf = max(probs)

        print(f"\nPredicted class: {pred_class}  confidence: {conf:.4f}")
        if inverse_map:
            print(f"Digits in this class: {inverse_map.get(pred_class, '?')}")
        print(f"All probabilities: {[f'{p:.4f}' for p in probs]}")

    else:
        print(f"Unknown choice: {src}")


if __name__ == "__main__":
    interactive_setup()
