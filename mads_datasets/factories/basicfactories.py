import re
import shutil
from typing import Any, Mapping

import pandas as pd
import polars as pl
from loguru import logger

from mads_datasets.base import (
    AbstractDatasetFactory,
    DatasetProtocol,
    DatastreamerProtocol,
)
from mads_datasets.datasets import PdDataset, PolarsDataset, TextDataset
from mads_datasets.datatools import keep_subdirs_only, walk_dir
from mads_datasets.settings import (
    DatasetSettings,
    PdDatasetSettings,
    SecureDatasetSettings,
    TextDatasetSettings,
)


class PenguinsDatasetFactory(AbstractDatasetFactory[DatasetSettings]):
    def create_dataset(self, *args, **kwargs) -> Mapping[str, DatasetProtocol]:
        pass


class IrisDatasetFactory(AbstractDatasetFactory[PdDatasetSettings]):
    def create_dataset(
        self, *args: Any, **kwargs: Any
    ) -> Mapping[str, DatasetProtocol]:
        self.download_data()
        df = pd.read_csv(
            self.filepath,
            header=None,
            names=self._settings.features + [self._settings.target],
        )
        df["sepal_width"].fillna(value=df["sepal_width"].mean(), inplace=True)
        df[self._settings.target] = df[self._settings.target].apply(
            lambda x: re.sub(r"setsoa", "setosa", x)
        )
        df.drop_duplicates(inplace=True)
        df["petal_width"] = df["petal_width"].apply(
            lambda x: float(re.findall(r"(\d+\.?\d*) mm", x)[0]) / 10
        )
        idx = df[df["sepal_length"] == 58].index
        df.loc[idx, "sepal_length"] = 5.8

        train = df.sample(frac=0.6)
        valid = df.drop(train.index)

        trainset = PdDataset(
            train, features=self._settings.features, target=self._settings.target
        )
        validset = PdDataset(
            valid, features=self._settings.features, target=self._settings.target
        )
        return {"train": trainset, "valid": validset}

    def create_datastreamer(
        self, batchsize: int, **kwargs
    ) -> Mapping[str, DatastreamerProtocol]:
        raise NotImplementedError("Not implemented yet...")


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
            f"Creating TextDatasets from {len(trainpaths)} trainfiles"
            f"and {len(testpaths)} testfiles."
        )

        traindataset = TextDataset(paths=trainpaths)
        testdataset = TextDataset(paths=testpaths)
        return {"train": traindataset, "valid": testdataset}


class SecureDatasetFactory(AbstractDatasetFactory[SecureDatasetSettings]):
    def create_dataset(
        self, *args: Any, **kwargs: Any
    ) -> Mapping[str, DatasetProtocol]:
        self.download_data()

        df = pl.read_parquet(self.filepath)
        dataset = PolarsDataset(df)

        return {"train": dataset}

class FavoritaDatasetFactory(AbstractDatasetFactory[DatasetSettings]):
    def create_dataset(
        self, *args: Any, **kwargs: Any
    ) -> Mapping[str, DatasetProtocol]:
        self.download_data()

        trainfile = self.subfolder / "train.parq"

        df = pl.read_parquet(trainfile)
        dataset = PolarsDataset(df)

        return {"train": dataset}
