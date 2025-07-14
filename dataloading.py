import torch
import torchvision.transforms as T
import json
import numpy as np
import os
import pathlib
from typing import Any, Callable, Optional, Tuple
import PIL.Image
from torchvision.datasets import VisionDataset, ImageFolder

PREPROCESSING = {
    "Default": T.Compose(
        [
            T.Resize((256, 256)),
            T.CenterCrop((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "Basic": T.Compose(
        [
            T.Resize((256, 256)),
            T.ToTensor(),
        ]
    ),
}


class BetterImageFolder(ImageFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        return_paths=False,
        split_folder="imagenet-val/",
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.return_paths = return_paths
        self.split_folder = (
            split_folder + "/" if split_folder[-1] != "/" else split_folder
        )

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_file, label = self.samples[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            try:
                image = self.transform(image)
            except:
                image = self.transform([image_file])
            # check if the batch dimension is present, if so, squeeze it
            if len(image.shape) == 4:
                image = image.squeeze(0)
        if self.target_transform:
            label = self.target_transform(label)
        if self.return_paths:
            return image, label, str(image_file).split(self.split_folder)[-1]
        return image, label


def get_dataloader(
    dataset,
    subset=None,
    batch_size=128,
    shuffle_pre_subset=True,
    dataloader_shuffle=False,
    disable_batch_size_adjustment=False,
    data_kwargs={},
    shuffle_seed=42,
):
    # data_kwargs = {}
    if subset is not None:
        if len(subset) == 1:
            subset = (subset[0], len(dataset))
        if shuffle_pre_subset:
            np.random.seed(shuffle_seed)
            subset_idx = np.random.permutation(len(dataset))[subset[0] : subset[1]]
            print("Subset indices:", subset_idx)
        else:
            subset_idx = list(range(subset[0], subset[1]))
        dataset = torch.utils.data.Subset(dataset, subset_idx)
        batch_size = min(batch_size, len(dataset))
    if len(dataset) % batch_size != 0 and not disable_batch_size_adjustment:
        # calculate a batch size that will evenly divide the data
        batch_size = len(dataset) // (len(dataset) // batch_size)
        print("Batch size adjusted to", batch_size)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=dataloader_shuffle,
        **data_kwargs,
    )
    return dataloader


def load_single_sample_from_path(
    img_path: str, transform: Optional[Callable] = None
) -> Tuple[Any, Any]:
    image = load_image(img_path, resize=False)
    return load_single_sample(image, transform), img_path.split("/")[-2]


def load_single_sample(
    image: PIL.Image.Image, transform: Optional[Callable] = None
) -> Any:
    if transform:
        image = transform(image).unsqueeze(0)
    return image


def load_image(image_path: str, resize: bool = True) -> PIL.Image.Image:
    image = PIL.Image.open(image_path).convert("RGB")
    if resize:
        return image.resize((256, 256))
    return image
