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
    train_ds = datasets.MNIST('/mnist/datasets/train',download=True,train=True)
    test_ds = datasets.MNIST('/mnist/datasets/test',download=True, train=False)

# Your custom model
class SimpleModel(composer.models.ComposerClassifier):
    """Your custom model."""

    def __init__(self, num_hidden: int, num_classes: int):
        module = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(28 * 28, num_hidden),
            torch.nn.Linear(num_hidden, num_classes),
        )
        self.num_classes = num_classes
        super().__init__(module=module, num_classes=num_classes)

def train():
    
    """Example for training with an algorithm on a custom model."""
    import torch
    import torch.utils.data
    import torch.distributed as dist
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, DistributedSampler

    import composer.models
    from composer import Trainer
    from composer.callbacks.checkpoint_saver import CheckpointSaver
    from composer.loggers import MLFlowLogger
    # Example algorithms to train with
    from composer.algorithms import CutOut, LabelSmoothing

    setup()
 
    #########################################################
    ## TRAINING CODE 
    #########################################################

    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
 
    print(f"Device rank {rank}")
    print(f"Loading datasets")

    data_path = "/mnist"
    checkpoint_path = "/checkpoints"

    train_ds = datasets.MNIST(f'{data_path}/datasets/', train=True, transform=transforms.ToTensor(), download=True)
    val_ds = datasets.MNIST(f'{data_path}/datasets/', train=False, transform=transforms.ToTensor())

    print(f"Creating samplers")

    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(),
        shuffle=True,            # True for train
        drop_last=True           # optional: helps equal-size batches
    )
    val_sampler = DistributedSampler(
        val_ds,
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(),
        shuffle=False,
        drop_last=False
    )
   
    print(f"Creating dataloaders")
    # Your custom train dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_ds,
        drop_last=False,
        shuffle=False,
        batch_size=5,
        num_workers=32,
        pin_memory=True,
        sampler=train_sampler
    )

    # Your custom eval dataloader
    eval_dataloader = torch.utils.data.DataLoader(
        dataset=val_ds,
        drop_last=False,
        shuffle=False,
        batch_size=5,
        num_workers=32,
        pin_memory=True,
        sampler=val_sampler
    )

    print(f"Creating Trainer with checkpoint path as {checkpoint_path} ")
    trainer = Trainer(
        model=SimpleModel(num_hidden=128, num_classes=10),
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        max_duration="1ep",
        python_log_level="debug"
        run_name=run_name,
        save_folder=checkpoint_path,
        save_interval="1ep", 
        save_overwrite=True,
        autoresume=True,
        algorithms=[CutOut(num_holes=1, length=0.5), LabelSmoothing(0.1)]
    )

    ####################################################
    #STEP 5: Gracefully shutdown distributed process group 
    ####################################################

    try:
        trainer.fit()
    
    finally:
        cleanup()

if __name__ == '__main__':
    get_dataset()
    train()