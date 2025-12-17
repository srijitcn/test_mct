import torchvision.datasets as datasets

train_ds = datasets.MNIST('/mnist/datasets/train',download=True,train=True)
test_ds = datasets.MNIST('/mnist/datasets/test',download=True, train=False)