"""Constants and configuration defaults."""

import logging
import os
from pathlib import Path

import torch

BASE_PRIOR_PRECISION = 1e-3
HASH_LENGTH = 5
BATCH_SIZE = 128
NUM_WORKERS = 12
PERSISTENT_WORKERS = True
DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.mps.is_available()
    else torch.device("cpu")
)
PIN_MEMORY = True and DEVICE.type != "mps"

logger = logging.getLogger(__name__)


def DATA_PATH(dataset: str) -> Path:
    """Return the root data path for a dataset.

    For ImageNet, prefer the shared ImageNet2012 location when running under SLURM.
    For other datasets, prefer a per-job scratch directory under SLURM.
    """
    if dataset.lower() == "imagenet":
        efficient_imagenet_path = Path("/scratch_local/datasets/ImageNet2012")
        inefficient_imagenet_path_lustre = Path("/mnt/lustre/datasets/ImageNet2012")
        inefficient_imagenet_path_ferranti = Path("/weka/datasets/ImageNet2012")

        imagenet_path = Path("../data")
        if efficient_imagenet_path.exists():
            imagenet_path = efficient_imagenet_path
        elif inefficient_imagenet_path_lustre.exists():
            imagenet_path = inefficient_imagenet_path_lustre
        elif inefficient_imagenet_path_ferranti.exists():
            imagenet_path = inefficient_imagenet_path_ferranti

        logger.info("Chose ImageNet path %s", imagenet_path)
        return imagenet_path
    return (
        Path(
            f"/scratch_local/{os.environ.get('SLURM_JOB_USER')}-"
            f"{os.environ.get('SLURM_JOBID')}"
        )
        if os.environ.get("SLURM_JOB_USER") and os.environ.get("SLURM_JOBID")
        else Path("../data")
    )


CIFAR_IMG_SIZE = (32, 32)
IMAGENET_IMG_SIZE = (224, 224)
INTERPOLATION = "bicubic"

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FOOD101_MEAN = (0.5574, 0.4420, 0.3269)
FOOD101_STD = (0.2606, 0.2647, 0.2673)
DATASET_CONFIGS = {
    "cifar10": {
        "num_classes": 10,
        "default_model_name": "wide_resnet_c_preact_26_3",
        "checkpoint_name": {
            "wide_resnet_c_preact_26_3": "cifar10_wide_resnet_c_preact_26_3_last.pt",
            "resnet_c_preact_26": "cifar10_last.pt",
        },
        "result_prefix": "cifar10",
        "mean": CIFAR10_MEAN,
        "std": CIFAR10_STD,
        "img_size": CIFAR_IMG_SIZE,
    },
    "cifar100": {
        "num_classes": 100,
        "default_model_name": "wide_resnet_c_preact_26_5",
        "checkpoint_name": {"wide_resnet_c_preact_26_5": "cifar100_last.pt"},
        "result_prefix": "cifar100",
        "mean": CIFAR100_MEAN,
        "std": CIFAR100_STD,
        "img_size": CIFAR_IMG_SIZE,
    },
    "imagenet": {
        "num_classes": 1000,
        "default_model_name": "resnet_50",
        "checkpoint_name": {
            "resnet_50": "resnet_50_imagenet_last.pt",
            (
                "timm:vit_little_patch16_reg4_gap_256.sbb_in1k"
            ): "vit_little_imagenet_last.pt",
        },
        "result_prefix": "imagenet",
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "img_size": IMAGENET_IMG_SIZE,
    },
    "food101": {
        "num_classes": 101,
        "default_model_name": "resnet_50",
        "checkpoint_name": {
            "resnet_50": "resnet_50_food101_last.pt",
            (
                "timm:vit_little_patch16_reg4_gap_256.sbb_in1k"
            ): "vit_little_food101_last.pt",
        },
        "result_prefix": "food101",
        "mean": FOOD101_MEAN,
        "std": FOOD101_STD,
        "img_size": IMAGENET_IMG_SIZE,
    },
}
LAPLACE_GRID_LOG10_FROM = 2.0  # 100
LAPLACE_GRID_LOG10_TO = 6.0  # 1000000
LAPLACE_GRID_NUM_POINTS = 50
