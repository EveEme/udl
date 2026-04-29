"""Data loading utilities for laplace experiments."""

import logging
import math
from collections.abc import Iterator, Sequence
from functools import partial
from pathlib import Path
from types import TracebackType

import numpy as np
import torch
import torchvision
from torch import Tensor
from torch.utils.data import (
    DataLoader,
    Dataset,
    Sampler,
)

from bul.utils.constants import CIFAR_IMG_SIZE, DATASET_CONFIGS, INTERPOLATION
from bul.utils.transforms import (
    transforms_cifar_eval,
    transforms_cifar_train,
)

logger = logging.getLogger(__name__)


class DefaultContext:
    """Identity context manager that does nothing."""

    def __enter__(self) -> None:
        """Enters the context manager."""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exits the context manager."""
        del exc_type, exc_value, traceback

    def __call__(self) -> "DefaultContext":
        """Call method that returns self.

        Returns:
            Itself.
        """
        return self


class PrefetchLoader:
    """Data loader that prefetches and preprocesses data on GPU for faster training."""

    def __init__(
        self,
        loader: DataLoader,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        device: torch.device | str,
    ) -> None:
        normalization_shape = (1, 3, 1, 1)

        if isinstance(device, str):
            device = torch.device(device)

        self.loader = loader
        self.device = device
        self.mean = torch.tensor(
            [x * 255 for x in mean], device=device, dtype=torch.float32
        ).view(normalization_shape)
        self.std = torch.tensor(
            [x * 255 for x in std], device=device, dtype=torch.float32
        ).view(normalization_shape)
        self.has_cuda = torch.cuda.is_available() and device.type == "cuda"

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        """Returns an iterator over the data, prefetching and preprocessing batches.

        Yields:
            The (input, target) tuple.
        """
        first = True
        if self.has_cuda:
            stream = torch.cuda.Stream()
            stream_context = partial(torch.cuda.stream, stream=stream)
        else:
            stream = None
            stream_context = DefaultContext()

        # Iterate inner loader; guard zero-batch case to avoid uninitialized vars
        for next_input, next_target in self.loader:
            with stream_context():
                next_input = next_input.to(device=self.device, non_blocking=True)
                next_target = next_target.to(device=self.device, non_blocking=True)
                next_input = next_input.to(torch.float32).sub_(self.mean).div_(self.std)

            if not first:
                # Yield the previous batch while we prefetch the next
                yield input, target  # noqa: F821, F823
            else:
                first = False

            if stream is not None:
                torch.cuda.current_stream().wait_stream(stream)

            input = next_input
            target = next_target

        # Only yield if at least one batch was seen
        if not first:
            yield input, target

    def __len__(self) -> int:
        """Returns the number of batches in the loader."""
        return len(self.loader)

    @property
    def sampler(self) -> Sampler:
        """Returns the sampler used by the underlying loader."""
        return self.loader.sampler

    @property
    def dataset(self) -> Dataset:
        """Returns the dataset used by the underlying loader."""
        return self.loader.dataset

    @property
    def batch_size(self) -> int:
        """Returns the batch size used by the underlying loader."""
        return self.loader.batch_size

    @property
    def drop_last(self) -> bool:
        """Returns whether the underlying loader drops the last incomplete batch."""
        return self.loader.drop_last


def calculate_total_dim(loader: DataLoader | PrefetchLoader, num_classes: int) -> int:
    """Calculate total dimension (C x N) without iterating through loader.

    Args:
        loader: DataLoader to calculate dimensions for.
        num_classes: Number of classes in the model output.

    Returns:
        Total dimension (dataset_size * num_classes), accounting for drop_last.
    """
    dataset_size = len(loader.dataset)
    batch_size = loader.batch_size

    if loader.drop_last:
        # If drop_last=True, incomplete final batch is dropped
        num_complete_batches = dataset_size // batch_size
        effective_size = num_complete_batches * batch_size
    else:
        # All samples are included
        effective_size = dataset_size

    return effective_size * num_classes


def calculate_num_samples(loader: DataLoader | PrefetchLoader) -> int:
    """Calculate number of samples without iterating through loader.

    Args:
        loader: DataLoader to calculate number of samples for.

    Returns:
        Number of samples, accounting for drop_last.
    """
    dataset_size = len(loader.dataset)
    batch_size = loader.batch_size

    if loader.drop_last:
        # If drop_last=True, incomplete final batch is dropped
        num_complete_batches = dataset_size // batch_size
        effective_size = num_complete_batches * batch_size
    else:
        # All samples are included
        effective_size = dataset_size

    return effective_size


def create_base_datasets(
    dataset: str,
    data_path: Path,
    img_size: tuple[int, int] = CIFAR_IMG_SIZE,
    interpolation: str = INTERPOLATION,
    *,
    use_prefetcher: bool = False,
) -> tuple[Dataset, Dataset]:
    """Create train and test datasets with appropriate transforms.

    Args:
        dataset: Dataset name ('cifar10').
        data_path: Path to store/load data.
        img_size: Image size for transforms.
        interpolation: Interpolation method for transforms.
        use_prefetcher: When True, outputs uint8 arrays and expects PrefetchLoader.

    Returns:
        tuple of (training_set, test_set).

    Raises:
        ValueError: The dataset is unknown.
    """
    config = DATASET_CONFIGS[dataset]

    if dataset == "cifar10":
        transform = transforms_cifar_eval(
            img_size=img_size,
            interpolation=interpolation,
            use_prefetcher=use_prefetcher,
            mean=config["mean"],
            std=config["std"],
        )
        training_set = torchvision.datasets.CIFAR10(
            root=data_path, train=True, download=True, transform=transform
        )
        test_set = torchvision.datasets.CIFAR10(
            root=data_path, train=False, download=True, transform=transform
        )
    else:
        msg = f"Unsupported dataset: {dataset}"
        raise ValueError(msg)

    return training_set, test_set


def create_base_loaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    *,
    use_prefetcher: bool = False,
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
    device: torch.device | None = None,
    drop_last_training: bool = False,
) -> tuple[DataLoader | PrefetchLoader, DataLoader | PrefetchLoader]:
    """Create basic train and test loaders, optionally with GPU prefetching.

    Args:
        train_dataset: Training dataset.
        test_dataset: Test dataset.
        batch_size: Batch size for loaders.
        num_workers: Number of worker processes.
        pin_memory: Whether to pin memory.
        persistent_workers: Whether to use persistent workers.
        use_prefetcher: If True, enable fast uint8 collate and GPU prefetching.
        mean: Channel means (RGB in [0, 1]) for normalization when prefetching.
        std: Channel standard deviations (RGB in [0, 1]) when prefetching.
        device: Target device for prefetch and normalization.
        drop_last_training: If True, drop the last incomplete batch for training
            loaders (both CPU and PrefetchLoader paths). Test/eval loaders never
            drop the last batch.

    Returns:
        tuple of (training_loader, test_loader).

    Raises:
        ValueError: If `use_prefetcher=True` but `mean`, `std`, or `device` is missing.
    """
    if use_prefetcher:
        if mean is None or std is None or device is None:
            msg = "mean/std/device must be provided when use_prefetcher=True"
            raise ValueError(msg)

        training_loader = create_loader(
            dataset=train_dataset,
            batch_size=batch_size,
            is_training_dataset=True,
            use_prefetcher=True,
            mean=mean,
            std=std,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            device=device,
            drop_last=drop_last_training,
        )
        test_loader = create_loader(
            dataset=test_dataset,
            batch_size=batch_size,
            is_training_dataset=False,
            use_prefetcher=True,
            mean=mean,
            std=std,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            device=device,
        )
    else:
        training_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last_training,
            persistent_workers=persistent_workers,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    return training_loader, test_loader


def get_dataset_labels(dataset: Dataset) -> list[int]:
    """Return class labels for a dataset without invoking transforms.

    Handles the one dataset we use with full certainty and includes
    conservative fallbacks:

    - CIFAR10 (torchvision): exposes ``targets`` (list[int]).

    If none of the above attributes exist, falls back to reading from
    ``samples``/``imgs`` or, as a last resort, to a slow enumeration.

    Args:
        dataset: Torch dataset instance.

    Returns:
        A list of integer class labels aligned with the dataset index order.

    Raises:
        ValueError: If the dataset does not expose any supported label attribute.
    """
    # 1) Common public attribute used by CIFAR
    if hasattr(dataset, "targets"):
        labels = dataset.targets
        return list(labels)

    # 3) Some datasets expose "labels" as a public alias
    if hasattr(dataset, "labels"):
        return list(dataset.labels)

    msg = "Dataset not supported"
    raise ValueError(msg)


def clone_eval_dataset(
    base_dataset: Dataset,
    *,
    img_size: tuple[int, int],
    interpolation: str,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> Dataset:
    """Clone a supported dataset with CPU evaluation transforms.

    This function is intentionally strict and supports only the datasets used in
    this project. It never falls back to defaults or guesses.

    Supported types and exact split propagation:
    - torchvision.datasets.CIFAR10 / CIFAR100: uses ``dataset.train`` (bool)

    Args:
        base_dataset: Original dataset instance to clone.
        img_size: Target evaluation image size.
        interpolation: Interpolation name for resizing.
        mean: Channel means (RGB in [0, 1]).
        std: Channel stds (RGB in [0, 1]).

    Returns:
        A new dataset instance with CPU evaluation transforms and the same split.

    Raises:
        ValueError: If the dataset type is unsupported.
    """
    if isinstance(base_dataset, torchvision.datasets.CIFAR10):
        transform = transforms_cifar_eval(
            img_size=img_size,
            interpolation=interpolation,
            use_prefetcher=False,
            mean=mean,
            std=std,
        )
        return torchvision.datasets.CIFAR10(
            root=base_dataset.root,
            train=base_dataset.train,
            download=False,
            transform=transform,
        )
    
    msg = f"Unsupported dataset type for cloning: {type(base_dataset)}"
    raise ValueError(msg)


def clone_train_dataset_with_train_aug(
    base_dataset: Dataset,
    dataset_name: str,
    *,
    interpolation: str,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    use_prefetcher: bool,
) -> Dataset:
    """Clone dataset with training-time augmentations for the given split.

    Returns:
        Dataset configured with the project's training-time transforms.

    Raises:
        ValueError: If ``dataset_name`` is not supported.
    """
    img_size = DATASET_CONFIGS[dataset_name]["img_size"]

    if dataset_name == "cifar10":
        transform = transforms_cifar_train(
            img_size=img_size,
            interpolation=interpolation,
            padding=4,
            hflip=0.5,
            max_rotation=0.0,
            color_jitter=0.0,
            use_prefetcher=use_prefetcher,
            mean=mean,
            std=std,
        )
        return torchvision.datasets.CIFAR10(
            root=base_dataset.root,
            train=True,
            download=False,
            transform=transform,
        )
    
    msg = f"Unsupported dataset for training augmentations: {dataset_name}"
    raise ValueError(msg)


def build_loader_for_dataset(
    dataset_to_load: Dataset,
    is_training: bool,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    use_prefetcher: bool,
    mean: tuple[float, float, float] | None,
    std: tuple[float, float, float] | None,
    device: torch.device | None,
    drop_last_training: bool = False,
) -> DataLoader | PrefetchLoader:
    """Create a loader for a dataset with optional GPU prefetching.

    If `use_prefetcher` is True, `mean`, `std`, and `device` must be provided.

    Returns:
        A DataLoader or a PrefetchLoader-wrapped DataLoader.

    Raises:
        ValueError: If `use_prefetcher=True` but stats or device are missing.
    """
    if use_prefetcher:
        if mean is None or std is None or device is None:
            msg = "mean/std/device must be provided when use_prefetcher=True"
            raise ValueError(msg)
        return create_loader(
            dataset=dataset_to_load,
            batch_size=batch_size,
            is_training_dataset=is_training,
            use_prefetcher=True,
            mean=mean,
            std=std,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            device=device,
            drop_last=drop_last_training if is_training else False,
        )
    return DataLoader(
        dataset_to_load,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_training if is_training else False,
        persistent_workers=persistent_workers,
    )


class RemappedDataset(Dataset):
    """Dataset wrapper that remaps class labels."""

    def __init__(self, dataset: Dataset, label_mapping: dict[int, int]):
        self.dataset = dataset
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        new_label = self.label_mapping[label]
        return data, new_label


def remap_class_labels(
    dataset: Dataset,
    original_labels: list[int],
    target_labels: list[int],
) -> RemappedDataset:
    """Remap class labels in dataset.

    Args:
        dataset: Dataset to remap.
        original_labels: Original class indices.
        target_labels: Target class indices.

    Returns:
        Dataset with remapped labels.
    """
    label_mapping = dict(zip(original_labels, target_labels, strict=True))
    return RemappedDataset(dataset, label_mapping)


def fast_collate(
    batch: Sequence[tuple[np.ndarray | Tensor | tuple, int]],
) -> tuple[Tensor, Tensor]:
    """A fast collation function optimized for uint8 images and int64 targets.

    Args:
        batch: A sequence of tuples containing image data and target labels.

    Returns:
        A tuple containing the collated image tensor and target tensor.

    Raises:
        TypeError: If the input is not a sequence of tuples.
        ValueError: If the input tensors are not of consistent shape or type.
    """
    if not isinstance(batch[0], tuple):
        msg = f"Tuple expected at batch[0], got {type(batch[0])}"
        raise TypeError(msg)

    batch_size = len(batch)
    if isinstance(batch[0][0], tuple):
        # This branch 'deinterleaves' and flattens tuples of input tensors into one
        # tensor ordered by position such that all tuple of position n will end up in a
        # torch.split(tensor, batch_size) in nth position
        inner_tuple_size = len(batch[0][0])
        flattened_batch_size = batch_size * inner_tuple_size
        targets = torch.zeros(flattened_batch_size, dtype=torch.int64)
        tensor = torch.zeros(
            (flattened_batch_size, *batch[0][0][0].shape), dtype=torch.uint8
        )
        for i in range(batch_size):
            if len(batch[i][0]) != inner_tuple_size:
                msg = "All input tensor tuples must be the same length"
                raise ValueError(msg)

            for j in range(inner_tuple_size):
                targets[i + j * batch_size] = batch[i][1]
                tensor[i + j * batch_size] += torch.from_numpy(batch[i][0][j])
        return tensor, targets

    if isinstance(batch[0][0], np.ndarray):
        targets = torch.from_numpy(np.array([b[1] for b in batch], dtype=np.int64))

        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)

        for i in range(batch_size):
            tensor[i] += torch.from_numpy(batch[i][0])

        return tensor, targets

    if isinstance(batch[0][0], Tensor):
        targets = torch.from_numpy(np.array([b[1] for b in batch], dtype=np.int64))

        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)

        for i in range(batch_size):
            tensor[i].copy_(batch[i][0])

        return tensor, targets

    msg = f"batch[0][0] has an invalid type {type(batch[0][0])}"
    raise ValueError(msg)


def make_deterministic_loader(
    dataloader: DataLoader | PrefetchLoader,
) -> DataLoader | PrefetchLoader:
    """Takes a loader and returns one that perfectly preserves iteration order.

    Assumes batch_size is always set.

    Args:
        dataloader: A torch.utils.data.DataLoader or PrefetchLoader instance

    Returns:
        A new DataLoader/PrefetchLoader with deterministic iteration order
    """
    # Check if it's a PrefetchLoader wrapper and unwrap if needed
    if isinstance(dataloader, PrefetchLoader):
        # Store wrapper info for reconstruction
        wrapper_info = {
            "mean": tuple(dataloader.mean.squeeze().cpu().numpy() / 255.0),
            "std": tuple(dataloader.std.squeeze().cpu().numpy() / 255.0),
            "device": dataloader.device,
        }
        inner_loader = dataloader.loader
    else:
        wrapper_info = None
        inner_loader = dataloader

    # Create deterministic loader
    deterministic_loader = DataLoader(
        dataset=inner_loader.dataset,
        batch_size=inner_loader.batch_size,
        shuffle=False,  # Forces SequentialSampler creation
        num_workers=inner_loader.num_workers,
        collate_fn=inner_loader.collate_fn,
        pin_memory=inner_loader.pin_memory,
        drop_last=inner_loader.drop_last,
        timeout=inner_loader.timeout,
        worker_init_fn=inner_loader.worker_init_fn,
        multiprocessing_context=inner_loader.multiprocessing_context,
        generator=torch.Generator().manual_seed(42),
        prefetch_factor=inner_loader.prefetch_factor,
        persistent_workers=inner_loader.persistent_workers,
        pin_memory_device=inner_loader.pin_memory_device,
        in_order=True,  # Ensures FIFO batch order with multiple workers
    )

    # Re-wrap if needed
    if wrapper_info:
        return PrefetchLoader(deterministic_loader, **wrapper_info)

    return deterministic_loader


def create_loader(
    dataset: Dataset,
    batch_size: int,
    is_training_dataset: bool,
    use_prefetcher: bool,
    mean: tuple[float, float, float] | None,
    std: tuple[float, float, float] | None,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    device: torch.device,
    drop_last: bool | None = None,
) -> DataLoader | PrefetchLoader:
    """Creates a DataLoader or PrefetchLoader based on the given parameters.

    Args:
        dataset: The dataset to wrap.
        batch_size: Batch size per iteration.
        is_training_dataset: If True, shuffles and drops last batch.
        use_prefetcher: If True, uses fast uint8 collate and GPU prefetching.
        mean: Channel means for normalization (RGB in [0,1]).
        std: Channel stds for normalization (RGB in [0,1]).
        num_workers: Number of worker processes.
        pin_memory: Whether to pin memory for faster host-device transfer.
        persistent_workers: Whether to keep workers alive between iterations.
        device: Target device for prefetching/normalization.
        drop_last: Optional override for drop_last behavior. When None, defaults
            to `is_training_dataset`; otherwise, uses the provided boolean.

    Returns:
        A standard DataLoader or a PrefetchLoader wrapper.

    Raises:
        ValueError: If `use_prefetcher=True` but `mean` or `std` is missing.
    """
    collate_fn = (
        fast_collate if use_prefetcher else torch.utils.data.dataloader.default_collate
    )

    effective_drop_last = is_training_dataset if drop_last is None else drop_last

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_training_dataset,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=effective_drop_last,
        persistent_workers=persistent_workers,
    )

    if use_prefetcher:
        if mean is None or std is None:
            msg = "mean/std must be provided when use_prefetcher=True"
            raise ValueError(msg)
        loader = PrefetchLoader(
            loader=loader,
            mean=mean,
            std=std,
            device=device,
        )

    return loader
