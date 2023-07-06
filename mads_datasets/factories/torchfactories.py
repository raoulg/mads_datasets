import random
import shutil
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from mads_datasets.base import AbstractDatasetFactory, DatasetProtocol
from mads_datasets.datasets import ImgDataset, MNISTDataset, SunspotDataset, TSDataset
from mads_datasets.datatools import (
    import_torch,
    iter_valid_paths,
    keep_subdirs_only,
    walk_dir,
    window,
)
from mads_datasets.settings import (
    DatasetSettings,
    ImgDatasetSettings,
    WindowedDatasetSettings,
)

if TYPE_CHECKING:
    import torch

    Tensor = torch.Tensor


class SunspotsDatasetFactory(AbstractDatasetFactory[WindowedDatasetSettings]):
    """
    Data from https://www.sidc.be/SILSO/datafiles
    """

    def create_dataset(
        self, *args: Any, **kwargs: Any
    ) -> Mapping[str, DatasetProtocol]:
        self.download_data()
        with import_torch() as torch:  # type: ignore
            spots = np.genfromtxt(str(self.filepath), usecols=(3))  # type: ignore
            tensors = torch.from_numpy(spots).type(torch.float32)  # type: ignore

            split = kwargs.pop("split", 0.8)
            idx = int(len(tensors) * split)
            train = tensors[:idx]
            valid = tensors[idx:]

            norm = max(train)
            train = train / norm
            valid = valid / norm

            window_size = self.settings.window_size
            horizon = self.settings.horizon
            trainset = SunspotDataset(self._window(train, window_size), horizon)
            validset = SunspotDataset(self._window(valid, window_size), horizon)
            return {"train": trainset, "valid": validset}

    @staticmethod
    def _window(data: "Tensor", window_size: int) -> "Tensor":
        idx = window(data, window_size)
        dataset = data[idx]
        dataset = dataset[..., None]
        return dataset


class FashionDatasetFactory(AbstractDatasetFactory[DatasetSettings]):
    def create_dataset(
        self, *args: Any, **kwargs: Any
    ) -> Mapping[str, DatasetProtocol]:
        self.download_data()
        with import_torch() as torch:  # type: ignore
            data = torch.load(self.filepath)  # type: ignore
            training_data = data["traindata"]
            training_labels = data["trainlabels"]
            test_data = data["testdata"]
            test_labels = data["testlabels"]

            train = MNISTDataset(training_data, training_labels)
            test = MNISTDataset(test_data, test_labels)

            return {"train": train, "valid": test}


class GesturesDatasetFactory(AbstractDatasetFactory[DatasetSettings]):
    def __init__(
        self,
        settings: DatasetSettings,
        **kwargs: Any,
    ) -> None:
        super().__init__(settings, **kwargs)
        self._created = False
        self.datasets: Mapping[str, DatasetProtocol]

    def create_dataset(
        self, *args: Any, **kwargs: Any
    ) -> Mapping[str, DatasetProtocol]:
        self.download_data()

        if self._created:
            return deepcopy(self.datasets)
        formats = [f.value for f in self._settings.formats]
        datadir = self.subfolder / "gestures-dataset"
        img = datadir / "gestures.png"
        if img.exists():
            shutil.move(img, datadir.parent / "gestures.png")
        keep_subdirs_only(datadir)
        paths = [path for path in walk_dir(datadir) if path.suffix in formats]
        random.shuffle(paths)

        split = kwargs.pop("split", 0.8)
        idx = int(len(paths) * split)
        trainpaths = paths[:idx]
        validpaths = paths[idx:]

        traindataset = TSDataset(trainpaths)
        validdataset = TSDataset(validpaths)
        datasets = {
            "train": traindataset,
            "valid": validdataset,
        }
        self.datasets = datasets  # type: ignore
        self._created = True
        return datasets


class FlowersDatasetFactory(AbstractDatasetFactory[ImgDatasetSettings]):
    def create_dataset(
        self, *args: Any, **kwargs: Any
    ) -> Mapping[str, DatasetProtocol]:
        self.download_data()
        formats = self._settings.formats
        paths_, class_names = iter_valid_paths(
            self.subfolder / "flower_photos", formats=formats
        )
        paths = [*paths_]
        random.shuffle([paths])
        trainidx = int(len(paths) * self._settings.trainfrac)
        train = paths[:trainidx]
        valid = paths[trainidx:]
        traindataset = ImgDataset(train, class_names, img_size=self._settings.img_size)
        validdataset = ImgDataset(valid, class_names, img_size=self._settings.img_size)
        return {"train": traindataset, "valid": validdataset}
