from __future__ import annotations

import hashlib
import random
import shutil
from abc import ABC, abstractmethod
from collections import Counter, OrderedDict
from copy import deepcopy
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
    Type,
    TypeVar,
)

import numpy as np
import torch
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import Vocab, vocab

from mads_datasets.datasets import (
    DatasetProtocol,
    ImgDataset,
    MNISTDataset,
    SunspotDataset,
    TextDataset,
    TSDataset,
)
from mads_datasets.datatools import (
    get_file,
    iter_valid_paths,
    keep_subdirs_only,
    walk_dir,
    window,
)
from mads_datasets.settings import (
    DatasetSettings,
    DatasetType,
    ImgDatasetSettings,
    TextDatasetSettings,
    WindowedDatasetSettings,
    fashionmnistsettings,
    flowersdatasetsettings,
    gesturesdatasetsettings,
    imdbdatasetsettings,
    sunspotsettings,
)

Tensor = torch.Tensor


class DatastreamerProtocol(Protocol):
    def stream(self) -> Iterator:
        ...


T = TypeVar("T", bound=DatasetSettings)


class AbstractDatasetFactory(ABC, Generic[T]):
    def __init__(
        self, settings: T, preprocessor: Type[PreprocessorProtocol], **kwargs: Any
    ) -> None:
        self._settings = settings
        self.data_dir = Path(
            kwargs.get("datadir", Path.home() / ".cache/mads_datasets")
        )
        self.preprocessor = preprocessor

    @property
    def settings(self) -> T:
        return self._settings

    def download_data(self) -> None:
        url = self._settings.dataset_url
        filename = self._settings.filename
        datadir = self.data_dir
        self.subfolder = Path(datadir) / self.settings.name
        if not self.subfolder.exists():
            logger.info("Start download...")
            self.subfolder.mkdir(parents=True)
            self.filepath = get_file(self.subfolder, filename, url=url, overwrite=False)
        else:
            logger.info(f"Dataset already exists at {self.subfolder}")
            self.filepath = self.subfolder / filename

        digest = self.calculate_md5(self.filepath)
        if self.settings.digest is not None:
            if digest != self.settings.digest:
                raise ValueError(
                    f"Digest of downloaded file {self.filepath} does not match expected digest"
                )
            else:
                logger.info(
                    f"Digest of downloaded {self.filepath} matches expected digest"
                )
        else:
            logger.info(f"Digest of downloaded {self.filepath} is {digest}")

    def get_preprocessor(self, **kwargs: Any) -> PreprocessorProtocol:
        return self.preprocessor(**kwargs)

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
        preprocessor = self.get_preprocessor(**kwargs)

        trainstreamer = BaseDatastreamer(
            traindataset, batchsize=batchsize, preprocessor=preprocessor
        )
        validstreamer = BaseDatastreamer(
            validdataset, batchsize=batchsize, preprocessor=preprocessor
        )
        return {"train": trainstreamer, "valid": validstreamer}


class SunspotsDatasetFactory(AbstractDatasetFactory[WindowedDatasetSettings]):
    """
    Data from https://www.sidc.be/SILSO/datafiles
    """

    def create_dataset(
        self, *args: Any, **kwargs: Any
    ) -> Mapping[str, DatasetProtocol]:
        self.download_data()
        spots = np.genfromtxt(str(self.filepath), usecols=(3))  # type: ignore
        tensors = torch.from_numpy(spots).type(torch.float32)

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
    def _window(data: Tensor, window_size: int) -> Tensor:
        idx = window(data, window_size)
        dataset = data[idx]
        dataset = dataset[..., None]
        return dataset


class FashionDatasetFactory(AbstractDatasetFactory[DatasetSettings]):
    def create_dataset(
        self, *args: Any, **kwargs: Any
    ) -> Mapping[str, DatasetProtocol]:
        self.download_data()
        data = torch.load(self.filepath)
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
        preprocessor: Type[PreprocessorProtocol],
        **kwargs: Any,
    ) -> None:
        super().__init__(settings, preprocessor, **kwargs)
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


class IMDBDatasetFactory(AbstractDatasetFactory[TextDatasetSettings]):
    def create_dataset(
        self, *args: Any, **kwargs: Any
    ) -> Mapping[str, DatasetProtocol]:
        self.download_data()
        testdir = self.subfolder / "aclImdb/test"
        traindir = self.subfolder / "aclImdb/train"
        keep_subdirs_only(testdir)
        keep_subdirs_only(traindir)

        # remove dir with unlabeled reviews
        unsup = traindir / "unsup"
        if unsup.exists():
            shutil.rmtree(traindir / "unsup")

        formats = [f.value for f in self._settings.formats]
        trainpaths = [path for path in walk_dir(traindir) if path.suffix in formats]
        testpaths = [path for path in walk_dir(testdir) if path.suffix in formats]
        logger.info(
            f"Creating TextDatasets from {len(trainpaths)} trainfiles and {len(testpaths)} testfiles."
        )

        traindataset = TextDataset(paths=trainpaths)
        testdataset = TextDataset(paths=testpaths)
        return {"train": traindataset, "valid": testdataset}

    def create_datastreamer(
        self, batchsize: int, **kwargs
    ) -> Mapping[str, DatastreamerProtocol]:
        datasets = self.create_dataset()
        traindataset = datasets["train"]
        validdataset = datasets["valid"]
        preprocessor = self.get_preprocessor(
            traindataset=traindataset, settings=self.settings
        )

        trainstreamer = BaseDatastreamer(
            traindataset, batchsize=batchsize, preprocessor=preprocessor
        )
        validstreamer = BaseDatastreamer(
            validdataset, batchsize=batchsize, preprocessor=preprocessor
        )
        return {"train": trainstreamer, "valid": validstreamer}


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


class PreprocessorProtocol(Protocol):
    def __call__(self, batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
        ...


class BasePreprocessor(PreprocessorProtocol):
    def __call__(self, batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
        X, y = zip(*batch)
        return torch.stack(X), torch.tensor(y)


class PaddedPreprocessor(PreprocessorProtocol):
    def __call__(self, batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
        X, y = zip(*batch)
        X_ = pad_sequence(X, batch_first=True, padding_value=0)  # noqa N806
        return X_, torch.tensor(y)


class BaseTokenizer(PreprocessorProtocol):
    def __init__(
        self, traindataset: TextDataset, settings: TextDatasetSettings
    ) -> None:
        self.maxvocab = settings.maxvocab
        self.maxtokens = settings.maxtokens
        self.clean = settings.clean_fn
        self.vocab = self.build_vocab(self.build_corpus(traindataset))

    @staticmethod
    def split_and_flat(corpus: List[str]) -> List[str]:
        """
        Split a list of strings on spaces into a list of lists of strings
        and then flatten the list of lists into a single list of strings.
        eg ["This is a sentence"] -> ["This", "is", "a", "sentence"]
        """
        corpus_ = [x.split() for x in corpus]
        corpus = [x for y in corpus_ for x in y]
        return corpus

    def build_corpus(self, dataset) -> List[str]:
        corpus = []
        for i in range(len(dataset)):
            x = self.clean(dataset[i][0])
            corpus.append(x)
        return corpus

    def build_vocab(
        self, corpus: List[str], oov: str = "<OOV>", pad: str = "<PAD>"
    ) -> Vocab:
        data = self.split_and_flat(corpus)
        counter = Counter(data).most_common()
        logger.info(f"Found {len(counter)} tokens")
        counter = counter[: self.maxvocab - 2]
        ordered_dict = OrderedDict(counter)
        v1 = vocab(ordered_dict, specials=[pad, oov])
        v1.set_default_index(v1[oov])
        return v1

    def cast_label(self, label: str) -> int:
        raise NotImplementedError

    def __call__(self, batch: List) -> Tuple[Tensor, Tensor]:
        labels, text = [], []
        for x, y in batch:
            if self.clean is not None:
                x = self.clean(x)  # type: ignore
            x = x.split()[: self.maxtokens]
            tokens = torch.tensor([self.vocab[word] for word in x], dtype=torch.int32)
            text.append(tokens)
            labels.append(self.cast_label(y))

        text_ = pad_sequence(text, batch_first=True, padding_value=0)
        return text_, torch.tensor(labels)


class IMDBTokenizer(BaseTokenizer):
    def __init__(self, traindataset, settings):
        super().__init__(traindataset, settings)

    def cast_label(self, label: str) -> int:
        if label == "neg":
            return 0
        else:
            return 1


class DatasetFactoryProvider:
    @staticmethod
    def create_factory(dataset_type: DatasetType, **kwargs) -> AbstractDatasetFactory:
        if dataset_type == DatasetType.FLOWERS:
            preprocessor = kwargs.get("preprocessor", BasePreprocessor)
            return FlowersDatasetFactory(
                flowersdatasetsettings, preprocessor=preprocessor, **kwargs
            )
        if dataset_type == DatasetType.IMDB:
            preprocessor = kwargs.get("preprocessor", IMDBTokenizer)
            return IMDBDatasetFactory(
                imdbdatasetsettings, preprocessor=preprocessor, **kwargs
            )
        if dataset_type == DatasetType.GESTURES:
            preprocessor = kwargs.get("preprocessor", PaddedPreprocessor)
            return GesturesDatasetFactory(
                gesturesdatasetsettings, preprocessor=preprocessor, **kwargs
            )
        if dataset_type == DatasetType.FASHION:
            preprocessor = kwargs.get("preprocessor", BasePreprocessor)
            return FashionDatasetFactory(
                fashionmnistsettings, preprocessor=preprocessor, **kwargs
            )
        if dataset_type == DatasetType.SUNSPOTS:
            preprocessor = kwargs.get("preprocessor", BasePreprocessor)
            return SunspotsDatasetFactory(
                sunspotsettings, preprocessor=preprocessor, **kwargs
            )

        raise ValueError(f"Invalid dataset type: {dataset_type}")


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
            if self.preprocessor is not None:
                X, Y = self.preprocessor(batch)  # noqa N806
            else:
                X, Y = zip(*batch)  # noqa N806
            yield X, Y
