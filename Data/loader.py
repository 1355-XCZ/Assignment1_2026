from torch.utils.data import DataLoader

from .squad import SQuADDataset


def make_loader(dataset: SQuADDataset, batch_size: int,
                shuffle: bool = False, pin_memory: bool = False) -> DataLoader:
    """Return a DataLoader over a SQuADDataset.

    Args:
        dataset:    A SQuADDataset instance.
        batch_size: Number of samples per batch.
        shuffle:    Whether to shuffle each epoch (True for training).
        pin_memory: Pin host memory for faster H2D transfers (True when using CUDA).
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
    )
