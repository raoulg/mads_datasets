from pathlib import Path
from loguru import logger

from mads_datasets.base import AbstractDatasetFactory
from mads_datasets.factories import (
    FashionDatasetFactory,
    FavoritaDatasetFactory,
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
    favoritasettings,
    flowersdatasetsettings,
    gesturesdatasetsettings,
    imdbdatasetsettings,
    irissettings,
    penguinssettings,
    sunspotsettings,
    SecureDatasetSettings,
)

__all__ = ["DatasetFactoryProvider", "DatasetType"]

__version__ = "0.3.7"


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
        if dataset_type == DatasetType.FAVORITA:
            return FavoritaDatasetFactory(favoritasettings, datadir=datadir)
        if dataset_type == DatasetType.SECURE:
            securesettings = kwargs.get("settings", None)
            if not securesettings:
                logger.warning(
                    "No settings provided for SecureDatasetFactory."
                )
            if not isinstance(securesettings, SecureDatasetSettings):
                raise ValueError(
                    f"Invalid settings type: {type(securesettings)}. "
                    "Expected SecureDatasetSettings."
                )
            return SecureDatasetFactory(securesettings, datadir=datadir)

        raise ValueError(f"Invalid dataset type: {dataset_type}")
