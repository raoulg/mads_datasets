from mads_datasets.factories.basicfactories import (
    FavoritaDatasetFactory,
    IMDBDatasetFactory,
    IrisDatasetFactory,
    PenguinsDatasetFactory,
    SecureDatasetFactory,
)
from mads_datasets.factories.torchfactories import (
    FashionDatasetFactory,
    FlowersDatasetFactory,
    GesturesDatasetFactory,
    SunspotsDatasetFactory,
)

__all__ = [
    "PenguinsDatasetFactory",
    "FavoritaDatasetFactory",
    "IrisDatasetFactory",
    "IMDBDatasetFactory",
    "SecureDatasetFactory",
    "SunspotsDatasetFactory",
    "FashionDatasetFactory",
    "GesturesDatasetFactory",
    "FlowersDatasetFactory",
]
