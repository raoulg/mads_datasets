from pathlib import Path

from mads_datasets.base import AbstractDatasetFactory
from mads_datasets.factories import (
    FashionDatasetFactory,
    FlowersDatasetFactory,
    GesturesDatasetFactory,
    IMDBDatasetFactory,
    IrisDatasetFactory,
    PenguinsDatasetFactory,
    SecureDatasetFactory,
    SunspotsDatasetFactory,
)
from mads_datasets.settings import (
    DatasetType,
    fashionmnistsettings,
    flowersdatasetsettings,
    garbagesettings,
    gesturesdatasetsettings,
    imdbdatasetsettings,
    irissettings,
    penguinssettings,
    sunspotsettings,
)

__all__ = ["DatasetFactoryProvider", "DatasetType"]

__version__ = "0.3.1"


class DatasetFactoryProvider:
    @staticmethod
    def create_factory(dataset_type: DatasetType, **kwargs) -> AbstractDatasetFactory:
        datadir = Path(kwargs.get("datadir", Path.home() / ".cache/mads_datasets"))
        if dataset_type == DatasetType.FLOWERS:
            return FlowersDatasetFactory(flowersdatasetsettings, datadir=datadir)
        if dataset_type == DatasetType.IMDB:
            return IMDBDatasetFactory(imdbdatasetsettings, datadir=datadir)
        if dataset_type == DatasetType.GESTURES:
            return GesturesDatasetFactory(gesturesdatasetsettings, datadir=datadir)
        if dataset_type == DatasetType.FASHION:
            return FashionDatasetFactory(fashionmnistsettings, datadir=datadir)
        if dataset_type == DatasetType.SUNSPOTS:
            return SunspotsDatasetFactory(sunspotsettings, datadir=datadir)
        if dataset_type == DatasetType.IRIS:
            return IrisDatasetFactory(irissettings, datadir=datadir)
        if dataset_type == DatasetType.PENGUINS:
            return PenguinsDatasetFactory(penguinssettings, datadir=datadir)
        if dataset_type == DatasetType.GARBAGE:
            return SecureDatasetFactory(garbagesettings, datadir=datadir)

        raise ValueError(f"Invalid dataset type: {dataset_type}")
