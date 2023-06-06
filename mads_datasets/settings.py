from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional, Tuple, cast

from pydantic import BaseModel, HttpUrl

from mads_datasets import tokenizer


class FileTypes(Enum):
    JPG = ".jpg"
    PNG = ".png"
    TXT = ".txt"
    ZIP = ".zip"
    TGZ = ".tgz"
    TAR = ".tar.gz"
    GZ = ".gz"
    PT = ".pt"


class ReportTypes(Enum):
    GIN = 1
    TENSORBOARD = 2
    MLFLOW = 3
    RAY = 4


class DatasetType(Enum):
    FLOWERS = 1
    IMDB = 2
    GESTURES = 3
    FASHION = 4
    SUNSPOTS = 5


class BaseSettings(BaseModel):
    def __str__(self) -> str:
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())

    def __repr__(self) -> str:
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())


class DatasetSettings(BaseSettings):
    dataset_url: HttpUrl
    filename: Path
    name: str
    formats: List[FileTypes]
    digest: Optional[str] = None


class ImgDatasetSettings(DatasetSettings):
    trainfrac: float
    img_size: Tuple[int, int]


class TextDatasetSettings(DatasetSettings):
    maxvocab: int
    maxtokens: int
    clean_fn: Callable


class WindowedDatasetSettings(DatasetSettings):
    horizon: int
    window_size: int


sunspotsettings = WindowedDatasetSettings(
    dataset_url=cast(HttpUrl, "https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.txt"),
    filename=Path("sunspots.txt"),
    name="sunspots",
    formats=[],
    horizon=3,
    window_size=26,
    digest="4ba2c195441e535045699468dae40fb0",
)

fashionmnistsettings = DatasetSettings(
    dataset_url=cast(
        HttpUrl, "https://github.com/raoulg/data_assets/raw/main/fashionmnist.pt"
    ),
    filename=Path("fashionmnist.pt"),
    name="fashionmnist",
    formats=[FileTypes.PT],
    digest="c4f1c3f76673fe3802f579773267163a",
)

gesturesdatasetsettings = DatasetSettings(
    dataset_url=cast(
        HttpUrl, "https://github.com/raoulg/gestures/raw/main/gestures-dataset.zip"
    ),
    filename=Path("gestures-dataset.zip"),
    name="gestures",
    formats=[FileTypes.TXT],
    digest="7966323b95154f314e831c312e5cc33b",
)

imdbdatasetsettings = TextDatasetSettings(
    dataset_url=cast(
        HttpUrl, "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    ),
    filename=Path("aclImdb_v1.tar.gz"),
    name="imdb",
    formats=[FileTypes.TXT],
    maxvocab=10000,
    maxtokens=100,
    clean_fn=tokenizer.clean,
    digest="7c2ac02c03563afcf9b574c7e56c153a",
)

flowersdatasetsettings = ImgDatasetSettings(
    dataset_url=cast(
        HttpUrl,
        "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
    ),
    filename=Path("flowers.tgz"),
    name="flowers",
    formats=[FileTypes.JPG],
    trainfrac=0.8,
    img_size=(224, 224),
    digest="6f87fb78e9cc9ab41eff2015b380011d",
)
