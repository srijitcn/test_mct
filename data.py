import os
import torch
import torch.distributed as dist
import composer.models
import torchvision.datasets as datasets

def setup():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    dist.destroy_process_group()

def get_dataset():
    train_ds = datasets.MNIST(root="/mnist/datasets",download=True,train=True)
    test_ds = datasets.MNIST(root="/mnist/datasets",download=True, train=False)
    print(f"Training data size: {len(train_ds)}")
    print(f"Test data size: {len(test_ds)}")

if __name__ == '__main__':
    print("===============================")
    print("Downloading dataset")
    get_dataset()
