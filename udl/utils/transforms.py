"""Transform utilities."""

import math
import random

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch import Tensor
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

STR_TO_INTERPOLATION = {
    "nearest": InterpolationMode.NEAREST,
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
}


class Resize:
    """Resize image transform.

    Args:
        img_size: Desired output size.
        interpolation: Desired interpolation method.
    """

    def __init__(self, img_size: int | tuple[int, int], interpolation: str) -> None:
        self.img_size = img_size

        self.interpolation = STR_TO_INTERPOLATION[interpolation]

    def __call__(self, img: Tensor) -> Tensor:
        """Applies the transform.

        Args:
            img: Image tensor to be resized.

        Returns:
            Resized image tensor.
        """
        return F.resize(img=img, size=self.img_size, interpolation=self.interpolation)


class ResizeKeepRatio:
    """Resizes the input while keeping its ratio.

    Args:
        size: Desired output size.
        longest: Weight for the longest side in ratio calculation.
        interpolation: Desired interpolation method.
        fill: Pixel fill value for the area outside the image.
    """

    def __init__(
        self,
        size: tuple[int, int],
        longest: float = 0.0,
        interpolation: str = "bilinear",
        fill: int = 0,
    ) -> None:
        self.size = size
        self.interpolation = STR_TO_INTERPOLATION[interpolation]
        self.longest = longest
        self.fill = fill

    @staticmethod
    def get_params(
        img: Tensor, target_size: tuple[int, int], longest: float
    ) -> list[int]:
        """Calculates the new size while keeping the aspect ratio.

        Args:
            img: Input image tensor.
            target_size: Desired output size.
            longest: Weight for the longest side in ratio calculation.

        Returns:
            New size that keeps the aspect ratio.
        """
        source_size = img.size[::-1]  # H, W
        h, w = source_size
        target_h, target_w = target_size
        ratio_h = h / target_h
        ratio_w = w / target_w
        ratio = max(ratio_h, ratio_w) * longest + min(ratio_h, ratio_w) * (
            1.0 - longest
        )
        size = [round(x / ratio) for x in source_size]

        return size

    def __call__(self, img: Tensor) -> Tensor:
        """Applies the transform.

        Args:
            img: Image tensor to be resized.

        Returns:
            Resized image tensor.
        """
        size = self.get_params(img, self.size, self.longest)
        img = F.resize(img, size, self.interpolation)

        return img


class RandomResizedCropAndInterpolation:
    """Crops the PIL Image to random size and aspect ratio with random interpolation.

    A crop of random size of the original size and a random
    aspect ratio of the original aspect ratio is made. This
    crop is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: Expected output size of each edge.
        scale: Range of size of the origin size cropped.
        ratio: Range of aspect ratio of the origin aspect ratio cropped.
        interpolation: Desired interpolation method.
    """

    def __init__(
        self,
        size: tuple[int, int],
        scale: tuple[float, float],
        ratio: tuple[float, float],
        interpolation: str,
    ) -> None:
        self.size = size
        self.interpolation = STR_TO_INTERPOLATION[interpolation]
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(
        img: Image.Image, scale: tuple[float, float], ratio: tuple[float, float]
    ) -> tuple[int, int, int, int]:
        """Gets parameters for ``crop`` for a random sized crop.

        Args:
            img: Image to be cropped.
            scale: Range of size of the origin size cropped.
            ratio: Range of aspect ratio of the origin aspect ratio cropped.

        Returns:
            Tuple of params (i, j, h, w) to be passed to ``crop`` for a random sized
            crop.
        """
        # PIL Image has size (width, height)
        area = img.size[0] * img.size[1]

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = round(math.sqrt(target_area * aspect_ratio))
            h = round(math.sqrt(target_area / aspect_ratio))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]

        if in_ratio < min(ratio):
            w = img.size[0]
            h = round(w / min(ratio))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = round(h * max(ratio))
        else:  # Whole image
            w = img.size[0]
            h = img.size[1]

        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2

        return i, j, h, w

    def __call__(self, img: Image.Image) -> Image.Image:
        """Applies the transform.

        Args:
            img: Image to be cropped and resized.

        Returns:
            Cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        interpolation = self.interpolation

        return F.resized_crop(img, i, j, h, w, self.size, interpolation)


class ToNumpy:
    """Transform that converts PIL images into NumPy arrays."""

    def __call__(self, pil_img: Image.Image) -> np.ndarray:
        """Converts a PIL image to a NumPy array.

        Args:
            pil_img: PIL image to be converted.

        Returns:
            NumPy array representation of the image.
        """
        np_img = np.array(pil_img, dtype=np.uint8)

        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)

        np_img = np.rollaxis(np_img, 2)  # HWC to CHW

        return np_img


def transforms_cifar_train(
    img_size: tuple[int, int],
    interpolation: str,
    padding: int,
    hflip: float,
    max_rotation: float,
    color_jitter: float,
    use_prefetcher: bool,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> transforms.Compose:
    """Creates a transform for CIFAR training.

    Args:
        img_size: Size of the image.
        interpolation: Interpolation method.
        padding: Padding size for random crop.
        hflip: Probability of horizontal flip.
        max_rotation: Maximum number of degrees to rotate image by.
        color_jitter: How much to jitter brightness, contrast and saturation.
        use_prefetcher: Whether to use prefetcher.
        mean: Mean values for normalization.
        std: Standard deviation values for normalization.

    Returns:
        A composition of transforms.
    """
    tfl = []

    if img_size != (32, 32):
        tfl += [
            Resize(img_size, interpolation),
        ]

    tfl += [
        transforms.RandomCrop(img_size, padding=padding),
    ]

    if hflip > 0:
        tfl += [transforms.RandomHorizontalFlip(p=hflip)]

    if max_rotation > 0:
        tfl += [transforms.RandomRotation(max_rotation)]

    if color_jitter != 0.0:
        # Duplicate for brightness, contrast, and saturation, no hue
        color_jitter = (color_jitter,) * 3
        tfl += [transforms.ColorJitter(*color_jitter)]

    if use_prefetcher:
        tfl += [ToNumpy()]
    else:
        tfl += [
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        ]

    return transforms.Compose(tfl)


def transforms_imagenet_train(
    img_size: tuple[int, int],
    scale: tuple[float, float],
    ratio: tuple[float, float],
    hflip: float,
    color_jitter: float,
    interpolation: str,
    use_prefetcher: bool,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> transforms.Compose:
    """Creates a transform for ImageNet training.

    Args:
        img_size: Size of the image.
        scale: Scale range for random resized crop.
        ratio: Aspect ratio range for random resized crop.
        hflip: Probability of horizontal flip.
        color_jitter: Strength of color jitter.
        interpolation: Interpolation method.
        use_prefetcher: Whether to use prefetcher.
        mean: Mean values for normalization.
        std: Standard deviation values for normalization.

    Returns:
        A composition of transforms.
    """
    primary_tfl = [
        RandomResizedCropAndInterpolation(
            img_size, scale=scale, ratio=ratio, interpolation=interpolation
        )
    ]
    if hflip > 0.0:
        primary_tfl += [transforms.RandomHorizontalFlip(p=hflip)]

    secondary_tfl = []

    if color_jitter is not None:
        # Duplicate for brightness, contrast, and saturation, no hue
        color_jitter = (float(color_jitter),) * 3
        secondary_tfl += [transforms.ColorJitter(*color_jitter)]

    final_tfl = []
    if use_prefetcher:
        # Prefetcher and collate will handle tensor conversion and norm
        final_tfl += [ToNumpy()]
    else:
        final_tfl += [
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        ]

    return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)


def transforms_cifar_eval(
    img_size: tuple[int, int],
    interpolation: str,
    use_prefetcher: bool,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> transforms.Compose:
    """Creates a transform for CIFAR evaluation.

    Args:
        img_size: Size of the image.
        interpolation: Interpolation method.
        use_prefetcher: Whether to use prefetcher.
        mean: Mean values for normalization.
        std: Standard deviation values for normalization.

    Returns:
        A torchvision.transforms.Compose of transforms.
    """
    tfl = []

    if img_size != (32, 32):
        tfl += [
            Resize(img_size, interpolation),
        ]

    if use_prefetcher:
        tfl += [ToNumpy()]
    else:
        tfl += [
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        ]

    return transforms.Compose(tfl)


def transforms_imagenet_eval(
    img_size: int | tuple[int, int],
    crop_pct: float,
    interpolation: str,
    use_prefetcher: bool,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> transforms.Compose:
    """Creates a transform for ImageNet evaluation.

    Args:
        img_size: Size of the image.
        crop_pct: Percentage of image to crop.
        interpolation: Interpolation method.
        use_prefetcher: Whether to use prefetcher.
        mean: Mean values for normalization.
        std: Standard deviation values for normalization.

    Returns:
        A torchvision.transforms.Compose of transforms.

    Raises:
        ValueError: When the image size is invalid.
    """
    tfl = []

    if isinstance(img_size, tuple | list):
        if len(img_size) != 2:
            msg = "Invalid image size provided"
            raise ValueError(msg)
        scale_size = tuple(math.floor(x / crop_pct) for x in img_size)
    else:
        scale_size = math.floor(img_size / crop_pct)
        scale_size = (scale_size, scale_size)

    # Default crop model is center
    # Aspect ratio is preserved, crops center within image, no borders are added,
    # image is lost
    if scale_size[0] == scale_size[1]:
        # Simple case, use torchvision built-in Resize w/ shortest edge mode
        # (scalar size arg)
        tfl += [
            transforms.Resize(
                scale_size[0], interpolation=STR_TO_INTERPOLATION[interpolation]
            )
        ]
    else:
        # Resize shortest edge to matching target dim for non-square target
        tfl += [ResizeKeepRatio(scale_size)]
    tfl += [transforms.CenterCrop(img_size)]

    if use_prefetcher:
        # Prefetcher and collate will handle tensor conversion and norm
        tfl += [ToNumpy()]
    else:
        tfl += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std),
            ),
        ]

    return transforms.Compose(tfl)
