from pathlib import Path

import numpy as np
import os

from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms as tv_transforms
from scipy.ndimage import gaussian_filter

class MetDataLoader:
    def __init__(self, episode_paths: list):
        self.episode_paths = episode_paths

    def __call__(self, sample: dict):
        data_path = Path(self.episode_paths[sample["_idx"]])

        sample["image"] = np.load(data_path)['image']
        # normalize (standardize) the input
        sample["image"] = (sample["image"] - sample["_mean"][:,None,None]) / sample["_std"][:,None,None]
        # add topo data
        sample["image"] = np.concatenate([sample["image"], sample["_topo"][None,:,:]], axis=0).astype(np.float32)

        # get case (datetime)
        sample["case"] = int(str(data_path).split("set.")[1])

        # pre-process the label
        sample["label"] = np.load(data_path)['label']
        sample["mask"] = np.where(sample["label"] < 0, False, True)
        sample["label"] = np.where(sample["label"] < 0, 0, sample["label"])
        constant = 0.1
        sample["label"] = np.log10(sample["label"] + constant)
        sample["label"] = (sample["label"] - sample["_label_mean"]) / sample["_label_std"]

        return sample

class Compose(tv_transforms.Compose):
    def __call__(self, sample: dict):
        for t in self.transforms:
            sample = t(sample)
        return sample

class MetDataset(Dataset):
    """
    meteorology dataset for classification
    """

    def __init__(
        self,
        mean,
        std,
        topo,
        label_mean,
        label_std,
        episode_paths: list,
        transform_pipeline: str = "default",
    ):
        super().__init__()

        self.episode_paths = episode_paths
        self.transform = self.get_transform(transform_pipeline)
        self.mean = mean
        self.std = std
        self.topo = topo
        self.label_mean = label_mean
        self.label_std = label_std

    def get_transform(self, transform_pipeline: str):
        """
        Creates a pipeline for processing data.
        """
        xform = None

        if transform_pipeline == "default":
            xform = Compose(
                [
                    MetDataLoader(episode_paths=self.episode_paths),
                ]
            )                       

        if xform is None:
            raise ValueError(f"Invalid transform {transform_pipeline} specified!")

        return xform

    def __len__(self):
        return len(self.episode_paths)

    def __getitem__(self, idx: int):
        sample = {"_idx": idx, "_mean": self.mean, "_std": self.std, "_topo": self.topo, "_label_mean": self.label_mean, "_label_std": self.label_std}
        sample = self.transform(sample)

        # remove private keys
        for key in list(sample.keys()):
            if key.startswith("_"):
                sample.pop(key)
 
        return sample


def load_data(
    mean,
    std,
    topo,
    label_mean,
    label_std,
    dataset_path: str,
    transform_pipeline: str = "default",
    return_dataloader: bool = True,
    num_workers: int = 2,
    batch_size: int = 32,
    shuffle: bool = False,
):
    """
    Constructs the dataset/dataloader.

    Args:
        transform_pipeline (str): 'default', 'aug', or other custom transformation pipelines
        return_dataloader (bool): returns either DataLoader or Dataset
        num_workers (int): data workers, set to 0 for VSCode debugging
        batch_size (int): batch size
        shuffle (bool): should be true for train and false for val

    Returns:
        DataLoader or Dataset
    """
    dataset_path = Path(dataset_path)
    filepaths = [x for x in dataset_path.iterdir() if not x.is_dir()]
    filepaths = sorted(filepaths)

    datasets = []

    datasets.append(MetDataset(mean=mean, std=std, topo=topo, label_mean=label_mean, label_std=label_std, episode_paths=filepaths, transform_pipeline=transform_pipeline))
    dataset = ConcatDataset(datasets)

    if not return_dataloader:
        return dataset

    return DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
    )
