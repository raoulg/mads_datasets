from mads_datasets.factories.basicfactories import (
    IMDBDatasetFactory,
    IrisDatasetFactory,
    PenguinsDatasetFactory,
)
from mads_datasets.factories.torchfactories import (
    FashionDatasetFactory,
    FlowersDatasetFactory,
    GesturesDatasetFactory,
    SunspotsDatasetFactory,
)

__all__ = [
    "PenguinsDatasetFactory",
    "IrisDatasetFactory",
    "IMDBDatasetFactory",
    "SunspotsDatasetFactory",
    "FashionDatasetFactory",
    "GesturesDatasetFactory",
    "FlowersDatasetFactory",
]
