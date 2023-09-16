import hashlib
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
)

import numpy as np
from loguru import logger

from mads_datasets.datatools import create_headers, get_file
from mads_datasets.settings import DatasetSettings, SecureDatasetSettings


class DatasetProtocol(Protocol):
    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> Any:
        ...


class ProcessingDatasetProtocol(DatasetProtocol):
    def process_data(self) -> None:
        ...


class DatastreamerProtocol(Protocol):
    def stream(self) -> Iterator:
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


T = TypeVar("T", bound=DatasetSettings)


class AbstractDatasetFactory(ABC, Generic[T]):
    def __init__(self, settings: T, datadir: Path) -> None:
        self._settings = settings
        self.datadir = datadir

        if type(settings) == SecureDatasetSettings:
            self.secure = True
        else:
            self.secure = False

    @property
    def settings(self) -> T:
        return self._settings

    def download_data(self) -> None:
        url = self._settings.dataset_url
        filename = self._settings.filename
        self.subfolder = Path(self.datadir) / self.settings.name

        if self.secure:
            headers = create_headers(self._settings)  # type: ignore
        else:
            headers = None

        if not self.subfolder.exists():
            logger.info("Start download...")
            self.subfolder.mkdir(parents=True)
            self.filepath = get_file(
                self.subfolder,
                filename,
                url=str(url),
                unzip=self._settings.unzip,
                overwrite=False,
                headers=headers,
            )
            digest = self.calculate_md5(self.filepath)
            if self.settings.digest is not None:
                if digest != self.settings.digest:
                    raise ValueError(
                        f"Digest of {self.filepath} does not match expected digest"
                        f"\nExpected: {self.settings.digest}\nGot: {digest}"
                    )
                else:
                    logger.info(f"Digest of {self.filepath} matches expected digest")
            else:
                logger.info(f"Digest of downloaded {self.filepath} is {digest}")

            if self._settings.unzip:
                logger.info(f"Removing unzipped file {self.filepath}")
                self.filepath.unlink()

        else:
            logger.info(f"Folder already exists at {self.subfolder}")
            self.filepath = self.subfolder / filename
            if self.filepath.exists():
                logger.info(f"File already exists at {self.filepath}")
            else:
                if not self._settings.unzip:
                    logger.warning(f"Expected file does not exist at {self.filepath}")


    @staticmethod
    def calculate_md5(file_path: Path, block_size: int = 2**16) -> str:
        md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(block_size), b""):
                md5.update(block)
        return md5.hexdigest()

    @abstractmethod
    def create_dataset(self, *args, **kwargs) -> Mapping[str, DatasetProtocol]:
        raise NotImplementedError

    def create_datastreamer(
        self, batchsize: int, **kwargs
    ) -> Mapping[str, DatastreamerProtocol]:
        datasets = self.create_dataset()
        traindataset = datasets["train"]
        validdataset = datasets["valid"]
        # get preprocessor from kwargs
        preprocessor: Optional[Callable] = kwargs.pop("preprocessor", None)

        trainstreamer = BaseDatastreamer(
            traindataset, batchsize=batchsize, preprocessor=preprocessor
        )
        validstreamer = BaseDatastreamer(
            validdataset, batchsize=batchsize, preprocessor=preprocessor
        )
        return {"train": trainstreamer, "valid": validstreamer}


class BaseDatastreamer:
    """This datastreamer wil never stop
    The dataset should have a:
        __len__ method
        __getitem__ method

    """

    def __init__(
        self,
        dataset: DatasetProtocol,
        batchsize: int,
        preprocessor: Optional[Callable] = None,
    ) -> None:
        self.dataset = dataset
        self.batchsize = batchsize

        if preprocessor is None:
            self.preprocessor = lambda x: zip(*x)
        else:
            self.preprocessor = preprocessor

        self.size = len(self.dataset)
        self.reset_index()

    def __len__(self) -> int:
        return int(len(self.dataset) / self.batchsize)

    def __repr__(self) -> str:
        return f"BasetDatastreamer: {self.dataset} (streamerlen {len(self)})"

    def reset_index(self) -> None:
        self.index_list = np.random.permutation(self.size)
        self.index = 0

    def batchloop(self) -> Sequence[Tuple]:
        batch = []
        for _ in range(self.batchsize):
            x, y = self.dataset[int(self.index_list[self.index])]
            batch.append((x, y))
            self.index += 1
        return batch

    def stream(self) -> Iterator:
        while True:
            if self.index > (self.size - self.batchsize):
                self.reset_index()
            batch = self.batchloop()
            X, Y = self.preprocessor(batch)  # noqa N806
            yield X, Y
