import torchvision.datasets as datasets

def get_dataset():
    train_ds = datasets.MNIST(root="/mnist/datasets",download=True,train=True)
    test_ds = datasets.MNIST(root="/mnist/datasets",download=True, train=False)
    print(f"Training data size: {len(train_ds)}")
    print(f"Test data size: {len(test_ds)}")

if __name__ == '__main__':
    print("===============================")
    print("Downloading dataset")
    get_dataset()
