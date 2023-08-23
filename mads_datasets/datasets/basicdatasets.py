from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

from PIL import Image
from tqdm import tqdm

from mads_datasets.base import AbstractDataset, DatasetProtocol

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl


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


class PolarsDataset(DatasetProtocol):
    def __init__(self, df: "pl.DataFrame"):
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> "pl.DataFrame":
        return self.df[idx]


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
