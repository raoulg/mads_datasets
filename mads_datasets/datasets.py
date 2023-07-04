import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, List, Protocol, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

Tensor = torch.Tensor

if TYPE_CHECKING:
    import pandas as pd


class DatasetProtocol(Protocol):
    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> Tuple:
        ...


class ProcessingDatasetProtocol(DatasetProtocol):
    def process_data(self) -> None:
        ...


class AbstractDataset(ABC, ProcessingDatasetProtocol):
    """The main responsibility of the Dataset class is to load the data from disk
    and to offer a __len__ method and a __getitem__ method
    """

    def __init__(self, paths: List[Path]) -> None:
        self.paths = paths
        random.shuffle(self.paths)
        self.dataset: List = []
        self.process_data()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple:
        return self.dataset[idx]

    @abstractmethod
    def process_data(self) -> None:
        raise NotImplementedError


class PdDataset(DatasetProtocol):
    def __init__(
        self,
        df: "pd.DataFrame",  # noqa: F821 type: ignore
        target: str,
        features: List[str],  # noqa: F821 type: ignore
    ) -> None:  # noqa: F821
        self.df = df
        self.target = target
        self.features = features

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(
        self, idx: int
    ) -> Tuple["pd.Series", "pd.Series"]:  # noqa: F821 type: ignore
        x = self.df[self.features].iloc[idx]
        y = self.df[self.target].iloc[idx]
        return x, y


class SunspotDataset(DatasetProtocol):
    def __init__(self, data: Tensor, horizon: int) -> None:
        self.data = data
        self.size = len(data)
        self.horizon = horizon

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        # get a single item
        item = self.data[idx]
        # slice off the horizon
        x = item[: -self.horizon, :]
        y = item[-self.horizon :, :].squeeze(
            -1
        )  # squeeze will remove the last dimension if possible.
        return x, y


class TensorDataset(DatasetProtocol):
    """The main responsibility of the Dataset class is to
    offer a __len__ method and a __getitem__ method
    """

    def __init__(self, data: Tensor, targets: Tensor) -> None:
        self.data = data
        self.targets = targets
        assert len(data) == len(targets)


class MNISTDataset(DatasetProtocol):
    """MNIST dataset
    Args:
        data (torch.Tensor): images
        labels (torch.Tensor): labels
        transform (callable, optional): Optional transform to be applied
            on a sample.

    Returns:
        torch.Tensor: image
        torch.Tensor: label
    """

    def __init__(self, data: Tensor, labels: Tensor):
        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        # add single batch dimension
        image = self.data[idx].unsqueeze(0)
        # scale to [0, 1] and cast to torch.float32
        image = image / 255.0
        image = image.type(torch.float32)
        label = self.labels[idx].type(torch.uint8)
        return image, label


class FacesDataset(AbstractDataset):
    def __init__(self, paths: List[Path]) -> None:
        super().__init__(paths)

    def process_data(self) -> None:
        for path in self.paths:
            img = self.load_image(path)
            self.dataset.append((img, path.name))

    def load_image(self, path: Path) -> Image.Image:
        img = Image.open(path)
        return img


class ImgDataset(AbstractDataset):
    def __init__(
        self, paths: List[Path], class_names: List[str], img_size: Tuple[int, int]
    ) -> None:
        self.img_size = img_size
        self.class_names = class_names
        super().__init__(paths)

    def process_data(self) -> None:
        for file in self.paths:
            img = self.load_image(file, self.img_size)
            x_ = np.reshape(img, (1,) + img.shape)
            x = torch.tensor(x_ / 255.0).type(torch.float32)
            y = self.class_names.index(file.parent.name)
            self.dataset.append((x, y))

    def load_image(self, path: Path, image_size: Tuple[int, int]) -> np.ndarray:
        # load file
        img_ = Image.open(path).resize(image_size, Image.LANCZOS)
        return np.asarray(img_)

    def __repr__(self) -> str:
        return f"ImgDataset (imgsize {self.img_size}, #classes {len(self.class_names)})"


class TSDataset(AbstractDataset):
    """This assume a txt file with numeric data
    Dropping the first columns is hardcoded
    y label is name-1, because the names start with 1

    """

    def process_data(self) -> None:
        for file in tqdm(self.paths, colour="#1e4706"):
            x_ = np.genfromtxt(file)[:, 3:]
            x = torch.tensor(x_).type(torch.float32)
            y = torch.tensor(int(file.parent.name) - 1)
            self.dataset.append((x, y))

    def __repr__(self) -> str:
        return f"TSDataset (size {len(self)})"


class TextDataset(AbstractDataset):
    """This assumes textual data, one line per file"""

    def process_data(self) -> None:
        for file in tqdm(self.paths, colour="#1e4706"):
            with open(file) as f:
                x = f.readline()
            y = file.parent.name
            self.dataset.append((x, y))

    def __repr__(self) -> str:
        return f"TextDataset (len {len(self)})"
